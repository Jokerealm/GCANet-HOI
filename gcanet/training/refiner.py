"""Training-only coupled adaptation; deployment uses the separate 6-DoF BCR."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import ImageRead, PairRelation, skew, so3_exp

def project(points, camera):
    xyz = points + camera[:, None]
    return xyz[..., :2] / xyz[..., 2:].clamp_min(0.05) * (1000.0 / 256.0)

def projection_jacobian(points, camera):
    """Derivative of the actual crop-camera projection, including z clamp."""
    xyz = points + camera[:, None]
    z = xyz[..., 2].clamp_min(0.05)
    active = (xyz[..., 2] > 0.05).to(points.dtype)
    zero = torch.zeros_like(z)
    scale = 1000.0 / 256.0
    row_x = torch.stack((scale / z, zero, -scale * xyz[..., 0] / z.square() * active), -1)
    row_y = torch.stack((zero, scale / z, -scale * xyz[..., 1] / z.square() * active), -1)
    return torch.stack((row_x, row_y), -2)

@torch.no_grad()
def lbs_vertex_jacobian(layer, pose, shape, indices, controls=22, parents=None):
    """Exact SMPL-H LBS vertex Jacobian for left local rotation increments.

    Includes both pose blend-shape and full kinematic-chain derivatives. Only
    requested vertices are skinned, rather than running 132 finite differences.
    Shape and current rotations are held fixed during this GN linearization.
    Returns B x selected_vertices x 3 x (controls*3), in metres / radian.
    """
    batch, joint_count = pose.shape[:2]
    if pose.shape[-2:] != (3, 3) or not 0 < controls <= joint_count:
        raise ValueError('Malformed LBS rotations or control count')
    indices = torch.as_tensor(indices, device=pose.device, dtype=torch.long)
    dtype = pose.dtype
    vt = layer.v_template.to(pose)
    sd = layer.shapedirs.to(pose)
    pd = layer.posedirs.to(pose)
    jr = layer.J_regressor.to(pose)
    weights = layer.lbs_weights.to(pose)[indices]
    parents = layer.parents.detach().cpu().tolist() if parents is None else parents
    if len(parents) != joint_count:
        raise ValueError('Cached LBS parent topology has the wrong size')
    shaped = vt[None] + torch.einsum('bl,vcl->bvc', shape, sd)
    joints = torch.einsum('jv,bvc->bjc', jr, shaped)
    identity = torch.eye(3, dtype=dtype, device=pose.device)
    posedirs = pd.reshape(joint_count - 1, 3, 3, len(vt), 3)[..., indices, :]
    posed = shaped[:, indices] + torch.einsum('bjik,jikvc->bvc', pose[:, 1:] - identity, posedirs)
    axes = skew(identity)
    drot = axes[None, None] @ pose[:, :controls, None]
    dimension = controls * 3
    chain, dchain = ([], [])
    for joint in range(joint_count):
        local = pose.new_zeros((batch, 4, 4))
        local[:, :3, :3] = pose[:, joint]
        local[:, :3, 3] = joints[:, joint] - (joints[:, parents[joint]] if joint else 0)
        local[:, 3, 3] = 1
        dlocal = pose.new_zeros((batch, dimension, 4, 4))
        if joint < controls:
            dlocal[:, joint * 3:(joint + 1) * 3, :3, :3] = drot[:, joint]
        if joint == 0:
            chain.append(local)
            dchain.append(dlocal)
        else:
            parent = parents[joint]
            if not 0 <= parent < joint:
                raise ValueError('LBS parents must be in topological order')
            chain.append(chain[parent] @ local)
            dchain.append(dchain[parent] @ local[:, None] + chain[parent][:, None] @ dlocal)
    transforms = torch.stack(chain, 1)
    derivatives = torch.stack(dchain, 1)
    transforms[:, :, :3, 3] -= (transforms[:, :, :3, :3] @ joints[..., None]).squeeze(-1)
    derivatives[:, :, :, :3, 3] -= (derivatives[:, :, :, :3, :3] @ joints[:, :, None, :, None]).squeeze(-1)
    skin = torch.einsum('vj,bjkl->bvkl', weights, transforms)
    dskin = torch.einsum('vj,bjdkl->bvdkl', weights, derivatives)
    homogeneous = torch.cat((posed, torch.ones_like(posed[..., :1])), -1)
    result = torch.einsum('bvdkl,bvl->bvkd', dskin[:, :, :, :3], homogeneous)
    if controls > 1:
        dv = torch.einsum('bjaik,jikvc->bvcja', drot[:, 1:], posedirs[:controls - 1])
        dv = torch.cat((dv.new_zeros((*dv.shape[:3], 1, 3)), dv), -2).reshape(batch, len(indices), 3, dimension)
        result = result + skin[:, :, :3, :3] @ dv
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError('Nonfinite analytic LBS Jacobian')
    return result

def damped_normal_solve(jacobian, residual, weight, stiffness, damping=0.001):
    """Batched positive-definite weighted GN solve with explicit diagnostics."""
    if jacobian.ndim != 3 or residual.shape != jacobian.shape[:2] or weight.shape != residual.shape:
        raise ValueError('GN row dimensions differ')
    if stiffness.shape != (len(jacobian), jacobian.shape[-1]):
        raise ValueError('GN stiffness dimension differs')
    if bool((weight < 0).any()) or bool((stiffness <= 0).any()) or damping <= 0:
        raise ValueError('GN weights must be nonnegative; stiffness/damping positive')
    j, r, w, p = (jacobian.double(), residual.double(), weight.double(), stiffness.double())
    normal = j.transpose(-1, -2) @ (j * w[..., None])
    normal = normal + torch.diag_embed(p + float(damping))
    rhs = -(j.transpose(-1, -2) @ (w * r)[..., None])
    answer = torch.linalg.solve(normal, rhs)[..., 0]
    if not bool(torch.isfinite(answer).all()):
        raise FloatingPointError('GN update is nonfinite')
    with torch.no_grad():
        eigenvalues = torch.linalg.eigvalsh(normal.detach())
        condition = eigenvalues[:, -1] / eigenvalues[:, 0]
    return (answer.to(jacobian.dtype), condition.to(jacobian.dtype))

def object_point_jacobian(rotation, anchors):
    """World point derivative for Exp(omega)R, t + delta_t."""
    rotated = anchors @ rotation.transpose(1, 2)
    identity = torch.eye(3, dtype=rotated.dtype, device=rotated.device)
    return torch.cat((-skew(rotated), identity.expand(*rotated.shape[:-1], 3, 3)), -1)

@torch.no_grad()
def tangent_coordinate_scales(human_jacobian, object_jacobian):
    """Scale tangent units by measured surface response, with explicit clamps.

    A normalized rotation coordinate nominally moves its sampled surface by
    5 cm RMS. Weakly observable axes use a 2 cm/rad denominator floor and all
    angular scales are bounded to [0.02, 0.5] rad/unit. Translation is 5 cm/unit.
    These are numerical coordinates, not a probabilistic precision estimate.
    """
    human_rms = human_jacobian.square().sum(-2).mean(-2).sqrt()
    object_rms = object_jacobian[..., :3].square().sum(-2).mean(-2).sqrt()
    human_scale = (0.05 / human_rms.clamp_min(0.02)).clamp(0.02, 0.5)
    object_scale = (0.05 / object_rms.clamp_min(0.02)).clamp(0.02, 0.5)
    translation_scale = object_scale.new_full((len(object_scale), 3), 0.05)
    return torch.cat((human_scale, object_scale, translation_scale), -1)

class ContactAdaptationRefiner(nn.Module):
    """Coupled body and object adaptation with learned stiffness."""

    def __init__(self, width=256, edge_chunk=32, rounds=3, speed_equivalent_optimizations=True):
        super().__init__()
        self.round_count = rounds
        if rounds != 3:
            raise ValueError('IACP adaptation uses three rounds')
        self.speed_equivalent_optimizations = bool(speed_equivalent_optimizations)
        self.pair_projection = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, width), nn.GELU())
        self.contact_projection = nn.Linear(2, width)
        self.anchor_geometry = nn.Linear(6, width)
        self.position = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.entity = nn.Embedding(3, width)
        self.relation = PairRelation(width, 64, edge_chunk)
        self.image_read = ImageRead(width)
        self.relation_message = nn.Sequential(nn.Linear(8, width), nn.GELU(), nn.Linear(width, width))
        self.body_attention = nn.MultiheadAttention(width, 8, batch_first=True, dropout=0)
        self.update = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.body_update = nn.Linear(width, 3)
        self.object_update = nn.Linear(width, 6)
        self.body_stiffness = nn.Linear(width, 3)
        self.object_stiffness = nn.Linear(width, 6)
        self.update_gain = nn.Parameter(torch.tensor(-4.0))
        nn.init.zeros_(self.body_update.weight)
        nn.init.zeros_(self.body_update.bias)
        nn.init.zeros_(self.object_update.weight)
        nn.init.zeros_(self.object_update.bias)
        nn.init.zeros_(self.body_stiffness.weight)
        nn.init.zeros_(self.body_stiffness.bias)
        nn.init.zeros_(self.object_stiffness.weight)
        nn.init.zeros_(self.object_stiffness.bias)
        xy = (torch.arange(32).float() + 0.5) / 32.0 - 0.5
        y, x = torch.meshgrid(xy, xy, indexing='ij')
        self.register_buffer('patch_uv', torch.stack((x, y), -1).reshape(-1, 2), persistent=False)

    def initial_features(self, native, tokens, asset, contact, geometry):
        indices, weights = (asset['interp64_idx'].long(), asset['interp64_weights'])
        batch_index = torch.arange(len(tokens), device=tokens.device)[:, None, None]
        object_z = (tokens[:, -64:][batch_index, indices] * weights[..., None]).sum(2)
        object_z = object_z + self.anchor_geometry(torch.cat((asset['anchors'], asset['normals']), -1))
        z = torch.cat((tokens[:, :504], object_z), 1)
        entity = torch.tensor([0] * 73 + [1] * 431 + [2] * len(asset['anchors'][0]), device=z.device)
        z = z + self.entity(entity)[None]
        pair = self.pair_projection(contact['pair_features'].float())
        p = contact['probabilities'][:, geometry.human_indices].float()
        p = p.detach()
        presence = contact['presence'].detach().float().reshape(-1, 1).expand_as(p)
        p_features = torch.stack((p, presence), -1)
        human = z[:, 73:504] + pair[:, None] + self.contact_projection(p_features)
        z = torch.cat((z[:, :73] + pair[:, None], human, z[:, 504:] + pair[:, None]), 1)
        return z

    @staticmethod
    def initial_state(native, geometry, reuse_native_body=False):
        pose = geometry.pose_matrices(native['initial_smpl_pose'])
        shape = native['initial_smpl_shape']
        body = native['initial_smpl_mesh'] if reuse_native_body else geometry.posed_body(pose, shape)
        return dict(human=native['smpl_verts6000'], rotation=native['obj_pose'], translation=native['obj_trans'], pose_matrices=pose, shape=shape, body=body, camera=native['cam_trans'])

    def prepare_context(self, native, tokens, memory, asset, geometry, contact):
        """Reuse only values independent of refinement state, with autograd."""
        if not self.speed_equivalent_optimizations:
            raise ValueError('Shared forward context requires explicit equivalent-optimization flag')
        return dict(native_state=self.initial_state(native, geometry, reuse_native_body=True), initial_features=self.initial_features(native, tokens, asset, contact, geometry), image_memory=self.image_read.prepare_memory(memory))

    def _support(self, relation, contact, geometry):
        difference = relation['current_relation'] - relation['relation_vectors']
        logit = relation['relation_logits_h']
        penalty = (difference / 0.05).square().sum(-1).clamp_max(200.0) * 0.5
        posterior = torch.cat((logit[..., :-1] - penalty, logit[..., -1:]), -1).softmax(-1)[..., :-1]
        p = contact['probabilities'][:, geometry.human_indices].detach().float()
        robust = torch.rsqrt(1 + (difference / 0.05).square().sum(-1))
        weights = posterior * robust * p[..., None]
        return (difference, weights)

    def _gn(self, state, z, relation, trace, contact, asset, geometry, round_index, human_jacobian, object_jacobian, scales, prepared_support=None):
        human = geometry.coarse(state['human'])
        rotation, translation, camera = (state['rotation'], state['translation'], state['camera'])
        anchors = asset['anchors']
        object_points = anchors @ rotation.transpose(1, 2) + translation[:, None]
        difference, pair_weights = self._support(relation, contact, geometry) if prepared_support is None else prepared_support
        with torch.no_grad():
            j_h = human_jacobian
            local_h = human - translation[:, None]
            rt = rotation.transpose(-1, -2)
            jc_h = rt[:, None] @ j_h
            jc_o = torch.cat((rt[:, None] @ skew(local_h), -rt[:, None].expand(-1, len(human[0]), -1, -1)), -1)
            jc = torch.cat((jc_h, jc_o), -1) * scales[:, None, None] / 0.1
            j_o = object_jacobian
            jvh = projection_jacobian(human, camera) @ j_h
            jvo = projection_jacobian(object_points, camera) @ j_o
            jvh = torch.cat((jvh, jvh.new_zeros((*jvh.shape[:-1], 6))), -1)
            jvo = torch.cat((jvo.new_zeros((*jvo.shape[:-1], 66)), jvo), -1)
            jv = torch.cat((jvh, jvo), 1) * scales[:, None, None] / 0.1
        weight_sum = pair_weights.sum(-1)
        relation_residual = (pair_weights[..., None] * difference).sum(-2) / weight_sum.clamp_min(1e-08)[..., None]
        relation_residual = relation_residual / 0.1
        weights_c = weight_sum[..., None].expand(-1, -1, 3) / human.shape[1]
        probabilities = trace['log_probs'].exp()[:, 73:]
        image_mass = probabilities[..., :-1].sum(-1).clamp_min(1e-08)
        image_target = probabilities[..., :-1] @ self.patch_uv.to(probabilities) / image_mass[..., None]
        points = torch.cat((human, object_points), 1)
        residual_v = (project(points, camera) - image_target) / 0.1
        weights_v = (image_mass * trace['visibility'][:, 73:])[..., None].expand(-1, -1, 2) / points.shape[1]
        j = torch.cat((jc.flatten(1, 2), jv.flatten(1, 2)), 1)
        residual = torch.cat((relation_residual.flatten(1), residual_v.flatten(1)), 1)
        weight = torch.cat((weights_c.flatten(1), weights_v.flatten(1)), 1)
        raw_stiffness = torch.cat((self.body_stiffness(z[:, :22]).flatten(1), self.object_stiffness(z[:, 504:].mean(1))), -1)
        stiffness = 0.01 * torch.exp(math.log(100.0) * raw_stiffness.tanh())
        delta, condition = damped_normal_solve(j, residual, weight, stiffness)
        delta = delta * scales * self.update_gain.sigmoid()
        diagnostic = dict(condition=condition, stiffness=stiffness, relation_mass=weight_sum.mean(-1), image_mass=image_mass.mean(-1), raw_stiffness=raw_stiffness, linearization='detached_exact_LBS_and_projection_Jacobian')
        return (delta[:, :66].reshape(-1, 22, 3), delta[:, 66:], diagnostic)

    def forward(self, native, tokens, memory, asset, geometry, contact, initial=None, context=None):
        if self.speed_equivalent_optimizations:
            context = self.prepare_context(native, tokens, memory, asset, geometry, contact) if context is None else context
            state = context['native_state'] if initial is None else initial
            z = context['initial_features']
        else:
            if context is not None:
                raise ValueError('Baseline refinement must not consume a shared context')
            state = self.initial_state(native, geometry) if initial is None else initial
            z = self.initial_features(native, tokens, asset, contact, geometry)
        base_human = state['human']
        base_joints = native['refh3d_keypoints'] if initial is None else geometry.joints(base_human)
        rounds = []
        for index in range(self.round_count):
            human = state['human']
            coarse = geometry.coarse(human)
            joints = base_joints + geometry.joints(human - base_human)
            objects = asset['anchors'] @ state['rotation'].transpose(1, 2) + state['translation'][:, None]
            relation = self.relation(z[:, 73:504], z[:, 504:], coarse, state['rotation'], state['translation'], asset['anchors'])
            target_error, support = self._support(relation, contact, geometry)
            normalizer = support.sum(-1).clamp_min(1e-08)
            message_h = torch.cat(((support[..., None] * target_error).sum(2) / normalizer[..., None] / 0.1, (support[..., None] * relation['relation_vectors']).sum(2) / normalizer[..., None] / 0.1, normalizer[..., None], relation['contact_logits_h'].sigmoid()[..., None]), -1)
            normalizer_o = support.sum(1).clamp_min(1e-08)
            message_o = torch.cat((-(support[..., None] * target_error).sum(1) / normalizer_o[..., None] / 0.1, -(support[..., None] * relation['relation_vectors']).sum(1) / normalizer_o[..., None] / 0.1, normalizer_o[..., None], relation['contact_logits_o'].sigmoid()[..., None]), -1)
            mh, mo = (self.relation_message(message_h), self.relation_message(message_o))
            mj, _ = self.body_attention(z[:, :73], z[:, 73:504], mh, need_weights=False)
            message = torch.cat((mj, mh, mo), 1)
            positions = torch.cat((joints, coarse, objects), 1)
            observation, image_trace = self.image_read(z + self.position(positions / 0.1) + message, memory, prepared_memory=context['image_memory'] if context is not None else None)
            z = z + observation + message
            z = z + self.update(z)
            with torch.no_grad():
                j_h = lbs_vertex_jacobian(geometry._body_layer, state['pose_matrices'].detach(), state['shape'].detach(), geometry.human_indices, parents=getattr(geometry, '_speed_body_parent_indices', None) if self.speed_equivalent_optimizations else None)
                j_o = object_point_jacobian(state['rotation'].detach(), asset['anchors'])
                scales = tangent_coordinate_scales(j_h, j_o)
            delta_h, delta_o, diagnostic = self._gn(state, z, relation, image_trace, contact, asset, geometry, index, j_h, j_o, scales, prepared_support=(target_error, support) if self.speed_equivalent_optimizations else None)
            pose = torch.cat((so3_exp(delta_h) @ state['pose_matrices'][:, :22], state['pose_matrices'][:, 22:]), 1)
            body = geometry.posed_body(pose, state['shape'])
            next_human = human + body - state['body']
            rotation = so3_exp(delta_o[:, :3]) @ state['rotation']
            translation = state['translation'] + delta_o[:, 3:]
            state = dict(human=next_human, rotation=rotation, translation=translation, pose_matrices=pose, body=body, shape=state['shape'], camera=state['camera'])
            result = dict(state, human_coarse=geometry.coarse(next_human), joints=base_joints + geometry.joints(next_human - base_human), image_trace=image_trace, round_index=index, delta_h=delta_h, delta_o=delta_o, solver=diagnostic, **{k: v for k, v in relation.items() if k != 'current_relation'})
            rounds.append(result)
        return rounds

@torch.no_grad()
def perturb_training_state(native, target, geometry, ann_ids, step, seed=123, native_state=None):
    """One balanced deterministic intervention per image; training only.

    Clean H/R/t are labels; the controllable LBS coordinates are initialized
    from native pose/shape, so the intervention exactly matches our update
    degrees of freedom without pretending native H is a parametric mesh.
    """
    if native_state is None:
        pose = geometry.pose_matrices(native['initial_smpl_pose'].detach())
        shape = native['initial_smpl_shape'].detach()
        body = geometry.posed_body(pose, shape)
    else:
        pose, shape, body = (native_state[key].detach() for key in ('pose_matrices', 'shape', 'body'))
    human = target['smpl_verts'].detach()
    rotation, translation = (target['obj_pose'].detach(), target['obj_trans'].detach())
    perturb_h, perturb_o, modes = ([], [], [])
    for ann_id in ann_ids.detach().cpu().reshape(-1).tolist():
        rng = torch.Generator(device='cpu').manual_seed(int(seed) + int(ann_id) * 1009 + int(step) * 1000003)
        mode = int(torch.randint(0, 4, (), generator=rng))
        h = torch.randn((22, 3), generator=rng)
        active = int(torch.randint(1, 22, (), generator=rng))
        h_mask = torch.zeros((22, 1))
        h_mask[active] = 1
        h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-08) * h_mask
        h *= (5 + 15 * float(torch.rand((), generator=rng))) * math.pi / 180.0
        o = torch.randn((6,), generator=rng)
        o[:3] = o[:3] / o[:3].norm().clamp_min(1e-08) * (10 + 35 * float(torch.rand((), generator=rng))) * math.pi / 180.0
        o[3:] = o[3:] / o[3:].norm().clamp_min(1e-08) * (0.02 + 0.13 * float(torch.rand((), generator=rng)))
        perturb_h.append(h if mode in (1, 3) else h * 0)
        perturb_o.append(o if mode in (2, 3) else o * 0)
        modes.append(mode)
    dh = torch.stack(perturb_h).to(human)
    do = torch.stack(perturb_o).to(human)
    next_pose = torch.cat((so3_exp(dh) @ pose[:, :22], pose[:, 22:]), 1)
    next_body = geometry.posed_body(next_pose, shape)
    mode = torch.tensor(modes, device=human.device)
    state = dict(human=human + next_body - body, rotation=so3_exp(do[:, :3]) @ rotation, translation=translation + do[:, 3:], pose_matrices=next_pose, shape=shape, body=next_body, camera=native['cam_trans'].detach())
    masks = dict(mode=mode, protect_human=(mode == 0) | (mode == 2), protect_object=(mode == 0) | (mode == 1), human_perturbation=dh, object_perturbation=do)
    return (state, masks)
