"""Ground-truth geometry supervision for training-only IACP adaptation."""
import torch
from torch import nn
from torch.nn import functional as F
from gcanet.surface import closest_surface

def _masked_mean(value, mask):
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1)

def _soft_surface_labels(nearest, candidates, distance, sigma=.02, null_distance=.10):
    squared = (nearest[:, :, None] - candidates[:, None]).square().sum(-1)
    probability = (-squared / (2 * sigma ** 2)).softmax(-1)
    is_null = distance > null_distance
    probability = probability * (~is_null)[..., None]
    return torch.cat((probability, is_null[..., None].to(probability.dtype)), -1)

def deterministic_far_pairs(squared_distance, ann_ids, count=8, seed=20261024):
    """Uniform-hash-ranked far endpoints; no global torch/NumPy RNG changes."""
    b, n, m = squared_distance.shape
    if count <= 0:
        raise ValueError('Far-pair count must be positive')
    ids = torch.as_tensor(ann_ids, device=squared_distance.device, dtype=torch.int64).reshape(b, 1, 1)
    row = torch.arange(n, device=squared_distance.device, dtype=torch.int64)[None, :, None]
    column = torch.arange(m, device=squared_distance.device, dtype=torch.int64)[None, None]
    # 31-bit integer mixing is reproducible on CPU/CUDA and independent of
    # batches, global RNG, epoch, and the order in which assets were loaded.
    key = ((ids + 1) * 1000003 + (row + 1) * 9176 + (column + 1) * 6361 + seed) & 2147483647
    key = ((key ^ (key >> 16)) * 73244475) & 2147483647
    key = ((key ^ (key >> 16)) * 73244475) & 2147483647
    key = key ^ (key >> 16)
    far = squared_distance > .10 ** 2
    rank = key.masked_fill(~far, 2147483647)
    indices = rank.topk(min(count, m), largest=False, sorted=True).indices
    valid = far.gather(-1, indices)
    return indices, valid

def balanced_contact_bce(logits, target):
    if logits.shape == target.shape + (1,):
        logits = logits.squeeze(-1)
    if logits.shape != target.shape:
        raise ValueError('Contact logits/targets have different shapes')
    error = F.binary_cross_entropy_with_logits(logits, target.to(logits), reduction='none')
    positive = target.bool()
    return .5 * (_masked_mean(error, positive) + _masked_mean(error, ~positive))

def image_auxiliary_loss(trace, image_target):
    """The donor's three .02 terms, also usable for the new surface queries."""
    if 'log_probs' in trace:
        score = trace['log_probs']
    elif 'logits' in trace:
        score = trace['logits']
    else:
        score = torch.cat((trace['score'], trace['null']), -1)
    logprob = score.float().log_softmax(-1)
    visibility = trace.get('visibility', trace.get('visible'))
    if visibility is None:
        visibility = trace['visibility_logits'].sigmoid()
    visibility = visibility.float()
    if visibility.ndim == 3:
        visibility = visibility.squeeze(-1)
    label, valid, visible = image_target['label'], image_target['valid'], image_target['visible']
    if logprob.shape[:2] != label.shape or visibility.shape != label.shape:
        raise ValueError('Image query ordering/shape does not match supervised labels')
    good, hidden = valid & visible, valid & ~visible
    ce = -logprob.gather(-1, label[..., None]).squeeze(-1)
    loss_visible = _masked_mean(ce, good)
    loss_null = _masked_mean(-logprob[..., -1], hidden)
    bce = F.binary_cross_entropy(visibility[:, 73:], visible[:, 73:].float(), reduction='none')
    loss_visibility = _masked_mean(bce, valid[:, 73:])
    return .02 * (loss_visible + loss_null + loss_visibility)

class JointSupervision(nn.Module):
    def __init__(self, assets, point_chunk=64, raster_function=None):
        super().__init__()
        # Assets already belong to the model/runner; avoid duplicate checkpoint
        # registration while retaining the same movable tensor object.
        object.__setattr__(self, 'assets', assets)
        self.point_chunk = int(point_chunk)
        self.raster_function = raster_function

    @torch.no_grad()
    def build_targets(self, targets, object_ids, ann_ids, include_image=True):
        human = targets.get('smpl_verts', targets.get('smpl_mesh_cam')).float()
        coarse = targets['smpl_verts431'].float()
        rotation, translation = targets['obj_pose'].float(), targets['obj_trans'].float()
        b = len(human)
        if coarse.ndim != 3 or coarse.shape[-1] != 3 or rotation.shape != (b, 3, 3) or translation.shape != (b, 3):
            raise ValueError('Malformed metric ground-truth geometry')
        ids = object_ids.detach().cpu().reshape(-1).tolist() if torch.is_tensor(object_ids) else list(object_ids)
        if len(ids) != b:
            raise ValueError('Object identity batch mismatch')
        batch = {key: value.to(human.device) for key, value in self.assets.batch(ids).items()}
        canonical_human = torch.bmm(coarse - translation[:, None], rotation)
        anchors = batch['anchors']
        object_world = torch.bmm(anchors, rotation.transpose(1, 2)) + translation[:, None]
        closest_h = torch.empty_like(coarse)
        distance_h = coarse.new_empty(coarse.shape[:2])
        for oid in sorted(set(int(x) for x in ids)):
            indices = torch.tensor([i for i, current in enumerate(ids) if int(current) == oid], device=human.device)
            vertices, faces = self.assets.mesh(oid)
            result = closest_surface(canonical_human[indices], vertices, faces, self.point_chunk)
            closest_h[indices] = result['points']
            distance_h[indices] = result['squared_distance'].clamp_min(0).sqrt()
        closest_o = closest_surface(object_world, human, self.assets.human_faces, self.point_chunk)
        distance_o = closest_o['squared_distance'].clamp_min(0).sqrt()
        labels_h = _soft_surface_labels(closest_h, anchors, distance_h)
        labels_o = _soft_surface_labels(closest_o['points'], coarse, distance_o)
        relation = canonical_human[:, :, None] - anchors[:, None]
        squared = relation.square().sum(-1)
        far_h, far_valid_h = deterministic_far_pairs(squared, ann_ids)
        far_o, far_valid_o = deterministic_far_pairs(squared.transpose(1, 2), ann_ids, seed=20261025)
        prepared = dict(human=human, human_coarse=coarse, rotation=rotation, translation=translation,
            anchors=anchors, loss_points=batch['loss_points'], relation_vectors=relation,
            relation_target_h=labels_h, relation_target_o=labels_o,
            contact_target_h=(distance_h < .05).float(), contact_target_o=(distance_o < .05).float(),
            relation_distance_h=distance_h, relation_distance_o=distance_o,
            closest_h_canonical=closest_h, closest_o_world=closest_o['points'],
            far_indices_h=far_h, far_valid_h=far_valid_h, far_indices_o=far_o, far_valid_o=far_valid_o,
            object_ids=ids)
        if include_image:
            prepared.update(self.build_image_targets(targets, prepared, batch))
        return prepared

    @torch.no_grad()
    def build_image_targets(self, targets, prepared, batch=None):
        if batch is None:
            batch = self.assets.batch(prepared['object_ids'])
        human, coarse = prepared['human'], prepared['human_coarse']
        rotation, translation = prepared['rotation'], prepared['translation']
        joints = targets['h3d_keypoints'][..., :3].float()
        uv = targets['h2d_keypoints'][..., :2].float()
        nf = 1000. / 256.
        zero = torch.zeros_like(uv[..., 0]); focal = torch.full_like(zero, nf)
        ax = torch.stack((focal, zero, -uv[..., 0]), -1)
        ay = torch.stack((zero, focal, -uv[..., 1]), -1)
        rhsx = uv[..., 0] * joints[..., 2] - nf * joints[..., 0]
        rhsy = uv[..., 1] * joints[..., 2] - nf * joints[..., 1]
        camera = torch.linalg.lstsq(torch.cat((ax, ay), 1), torch.cat((rhsx, rhsy), 1)[..., None]).solution[..., 0]
        def projection(points):
            xyz = points + camera[:, None]
            return xyz[..., :2] / xyz[..., 2:].clamp_min(.05) * nf
        fit_error = (projection(joints) - uv).norm(dim=-1).mean(-1)
        native_object = targets['obj_verts'][..., :3].float()
        anchors_world = torch.bmm(prepared['anchors'], rotation.transpose(1, 2)) + translation[:, None]
        points = torch.cat((joints, coarse, native_object, anchors_world), 1)
        target_uv = projection(points)
        valid = (target_uv.abs() < .49).all(-1) & (points[..., 2] + camera[:, None, 2] > .05) & (fit_error[:, None] < .03)
        index = ((target_uv + .5) * 32).long().clamp(0, 31)
        labels = index[..., 1] * 32 + index[..., 0]
        raster = self.raster_function
        if raster is None:
            from gcanet.gsf_supervision import raster_depth
            raster = raster_depth
        depths = raster(human, self.assets.human_faces, camera)
        for i, oid in enumerate(prepared['object_ids']):
            vertices, faces = self.assets.mesh(oid)
            vertices = vertices.to(human) @ rotation[i].T + translation[i]
            depths[i:i + 1] = torch.minimum(depths[i:i + 1], raster(vertices[None], faces, camera[i:i + 1]))
        bi = torch.arange(len(human), device=human.device)[:, None]
        nearest = depths[:, 0][bi, index[..., 1], index[..., 0]]
        visible = (points[..., 2] + camera[:, None, 2] <= nearest + .05) & torch.isfinite(nearest)
        visible[:, :73] = True
        native_count = joints.shape[1] + coarse.shape[1] + native_object.shape[1]
        shared_count = joints.shape[1] + coarse.shape[1]
        native_index = torch.arange(native_count, device=human.device)
        joint_index = torch.cat((torch.arange(shared_count, device=human.device),
                                 torch.arange(native_count, points.shape[1], device=human.device)))
        def select(indices):
            return dict(label=labels[:, indices], valid=valid[:, indices], visible=visible[:, indices])
        return dict(native_image_target=select(native_index), image_target=select(joint_index),
                    target_camera=camera, target_camera_fit_error=fit_error)

    def native_image_loss(self, traces, prepared):
        if not traces:
            raise ValueError('Native evidence trace is empty')
        return sum(image_auxiliary_loss(trace, prepared['native_image_target']) for trace in traces) / len(traces)

    def relation_loss(self, state, prepared):
        labels_h, labels_o = prepared['relation_target_h'], prepared['relation_target_o']
        logits_h, logits_o = state['relation_logits_h'].float(), state['relation_logits_o'].float()
        if logits_h.shape != labels_h.shape or logits_o.shape != labels_o.shape:
            raise ValueError('Relation direction/null shapes do not match')
        ce_h = -(labels_h * logits_h.log_softmax(-1)).sum(-1).mean()
        ce_o = -(labels_o * logits_o.log_softmax(-1)).sum(-1).mean()
        prediction = state['relation_vectors'].float()
        target = prepared['relation_vectors']
        if prediction.shape != target.shape:
            raise ValueError('Relation vector shape does not match complete pair targets')
        error = F.smooth_l1_loss(prediction / .1, target / .1, reduction='none', beta=.1).mean(-1)
        def vector_direction(value, probability, indices, far_valid):
            near_valid = probability[..., -1] == 0
            near = _masked_mean((value * probability[..., :-1]).sum(-1), near_valid)
            far = _masked_mean(value.gather(-1, indices), far_valid)
            return .5 * (near + far)
        vector_h = vector_direction(error, labels_h, prepared['far_indices_h'], prepared['far_valid_h'])
        vector_o = vector_direction(error.transpose(1, 2), labels_o, prepared['far_indices_o'], prepared['far_valid_o'])
        return .5 * (ce_h + ce_o) + .5 * (vector_h + vector_o)

    def surface_loss(self, human, donor_human):
        human, baseline = human.float(), donor_human.detach().float()
        if human.shape != baseline.shape:
            raise ValueError('Surface regularization requires matching dense vertices')
        edges = self.assets.human_edges.to(human.device)
        old_length = (baseline[:, edges[:, 0]] - baseline[:, edges[:, 1]]).norm(dim=-1)
        new_length = (human[:, edges[:, 0]] - human[:, edges[:, 1]]).norm(dim=-1)
        edge_loss = (new_length - old_length).abs().mean()
        displacement = human - baseline
        source = self.assets.human_adjacency_source.to(human.device)
        target = self.assets.human_adjacency_target.to(human.device)
        neighbor_sum = torch.zeros_like(displacement)
        neighbor_sum.index_add_(1, source, displacement[:, target])
        degree = self.assets.human_degree.to(human)
        if len(degree) != human.shape[1]:
            raise ValueError('Human topology and dense vertices differ')
        laplacian = displacement - neighbor_sum / degree.clamp_min(1)[None, :, None]
        return edge_loss + laplacian.abs().mean() + .1 * displacement.abs().mean()

    def losses(self, rounds, donor_human, targets, object_ids, ann_ids, prepared=None, ramp=1.):
        if len(rounds) != 2:
            raise ValueError('The approved recipe uses exactly two updates')
        if not 0 <= float(ramp) <= 1:
            raise ValueError('Loss ramp must be in [0,1]')
        if prepared is None:
            prepared = self.build_targets(targets, object_ids, ann_ids)
        truth_object = torch.bmm(prepared['loss_points'], prepared['rotation'].transpose(1, 2)) + prepared['translation'][:, None]
        result = {}
        for index, (state, weight) in enumerate(zip(rounds, (1. / 3., 2. / 3.))):
            prediction = torch.bmm(prepared['loss_points'], state['rotation'].float().transpose(1, 2)) + state['translation'].float()[:, None]
            final = F.l1_loss(state['human'].float(), prepared['human']) + F.l1_loss(prediction, truth_object)
            relation = self.relation_loss(state, prepared)
            contact = .5 * (balanced_contact_bce(state['contact_logits_h'], prepared['contact_target_h']) +
                            balanced_contact_bce(state['contact_logits_o'], prepared['contact_target_o']))
            image = image_auxiliary_loss(state['image_trace'], prepared['image_target'])
            surface = self.surface_loss(state['human'], donor_human)
            for name, value, coefficient in (('final', final, 1.), ('relation', relation, .1),
                    ('contact', contact, .1), ('image', image, 1.), ('surface', surface, .01)):
                result['joint%d_%s' % (index + 1, name)] = value * (float(ramp) * weight * coefficient)
        return result

    forward = losses