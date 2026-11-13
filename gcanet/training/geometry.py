"""Geometry and relation features used only for IACP joint adaptation."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

def skew(v):
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(v.shape[:-1] + (3, 3))

def vee(m):
    return torch.stack((m[..., 2, 1], m[..., 0, 2], m[..., 1, 0]), -1)

def so3_exp(omega):
    """SO(3) exponential with a finite, correct derivative at exactly zero."""
    angle = torch.linalg.vector_norm(omega, dim=-1)
    k = skew(omega)
    a = torch.sinc(angle / math.pi)[..., None, None]
    b = (0.5 * torch.sinc(angle / (2 * math.pi)).square())[..., None, None]
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device)
    return eye + a * k + b * (k @ k)

class _ProperPolar(torch.autograd.Function):
    """Proper polar rotation with an implicit backward, not SVD-vector backward.

Repeated positive singular values (spheres, cubes) are harmless: the derivative
uses sums of singular values. A genuinely ambiguous rank-one/reflection fit is
rejected instead of silently substituting a different pose or gradient.
"""
    @staticmethod
    def forward(ctx, covariance):
        u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
        sign = torch.linalg.det(u @ vh)
        diagonal = torch.ones(covariance.shape[:-2] + (3,), dtype=covariance.dtype, device=covariance.device)
        diagonal[..., -1] = torch.where(sign < 0, -torch.ones_like(sign), torch.ones_like(sign))
        rotation = (u * diagonal[..., None, :]) @ vh
        symmetric = rotation.transpose(-1, -2) @ covariance
        symmetric = (symmetric + symmetric.transpose(-1, -2)) * 0.5
        eye = torch.eye(3, dtype=covariance.dtype, device=covariance.device)
        trace = symmetric.diagonal(dim1=-2, dim2=-1).sum(-1)
        system = trace[..., None, None] * eye - symmetric
        values = torch.linalg.eigvalsh(system)
        relative_gap = values[..., 0] / values.abs().amax(-1).clamp_min(torch.finfo(covariance.dtype).tiny)
        if not bool(torch.isfinite(system).all()) or bool((relative_gap < 1e-10).any()):
            raise FloatingPointError('Ambiguous historical object pose fit: polar derivative gap below 1e-10')
        ctx.save_for_backward(rotation, system)
        return rotation

    @staticmethod
    def backward(ctx, gradient):
        rotation, system = ctx.saved_tensors
        rtg = rotation.transpose(-1, -2) @ gradient
        rhs = vee(rtg - rtg.transpose(-1, -2))
        tangent = torch.linalg.solve(system.transpose(-1, -2), rhs[..., None]).squeeze(-1)
        return rotation @ skew(tangent)

def historical_pose_fit(canonical, predicted, return_scale=False):
    """Reproduce donor R/t, including scale in t but NOT in output CAD.

Donor: C=Bc^T Ac/n, R=proper_polar(C), c=tr(R^T C)/var(A),
       t=mean(B)-c R mean(A).  It discards c when generating the final mesh.
"""
    if canonical.shape != predicted.shape or canonical.shape[-1] != 3:
        raise ValueError('Historical pose fit expects matching (..., N, 3) arrays')
    dtype = predicted.dtype
    a, b = canonical.double(), predicted.double()
    ma, mb = a.mean(-2), b.mean(-2)
    ac, bc = a - ma[..., None, :], b - mb[..., None, :]
    covariance = bc.transpose(-1, -2) @ ac / a.shape[-2]
    variance = ac.square().sum(-1).mean(-1)
    if bool((variance <= 1e-16).any()):
        raise ValueError('Canonical object points have zero variance')
    rotation = _ProperPolar.apply(covariance)
    scale = (rotation * covariance).sum((-1, -2)) / variance
    translation = mb - scale[..., None] * (rotation @ ma[..., None]).squeeze(-1)
    result = (rotation.to(dtype), translation.to(dtype))
    return result + (scale.to(dtype),) if return_scale else result

def compare_legacy_pose_mapping(rotation, translation, legacy_rotation, legacy_translation,
                                object_ids, assets, tolerance=2e-6):
    """Compare both readouts on complete, unscaled CADs, not just the 64 points."""
    if tolerance <= 0:
        raise ValueError('Pose-readout audit tolerance must be positive')
    rotation_error = (rotation - legacy_rotation).abs().amax()
    translation_error = (translation - legacy_translation).abs().amax()
    records = []
    for index, oid in enumerate(object_ids):
        vertices, _ = assets.mesh(int(oid))
        vertices = vertices.to(device=rotation.device, dtype=rotation.dtype)
        current = vertices @ rotation[index].T + translation[index]
        legacy = vertices @ legacy_rotation[index].T + legacy_translation[index]
        difference = torch.linalg.vector_norm(current - legacy, dim=-1)
        records.append(dict(object_id=int(oid), complete_cad_vertices=len(vertices),
                            max_vertex_difference_m=float(difference.max())))
    maximum = max(record['max_vertex_difference_m'] for record in records)
    passed = (bool(torch.isfinite(rotation_error)) and bool(torch.isfinite(translation_error))
              and math.isfinite(maximum) and float(rotation_error) <= tolerance
              and float(translation_error) <= tolerance and maximum <= tolerance)
    return dict(passed=passed, reference='original bound donor kps_to_Rt using NumPy rigid_transform_3D',
                tolerance_absolute=tolerance, rotation_max_absolute=float(rotation_error),
                translation_max_absolute_m=float(translation_error),
                complete_cad_max_vertex_difference_m=maximum, samples=records,
                same_actual_raw64_input=True, final_cad_scaled=False)

def _checkpoint(function, *values):
    if torch.is_grad_enabled() and any(v.requires_grad for v in values if torch.is_tensor(v)):
        # Shared layers are checkpointed repeatedly across chunks and both
        # trajectories. Non-reentrant checkpointing supports DDP unused-parameter
        # discovery without marking the same parameter ready multiple times.
        return checkpoint(function, *values, preserve_rng_state=False, use_reentrant=False)
    return function(*values)

class AdaptationGeometry(nn.Module):
    """Fixed mesh operators; donor parameters remain owned by the native model."""
    def __init__(self, native_model, device):
        super().__init__()
        from gcanet.reconstruction.models.templates import smplh, obj_dict
        object.__setattr__(self, '_body_layer', native_model.smplh_layer)
        down0, down1 = [d.coalesce() for d in smplh._D]
        indices0 = self._selection_indices(down0)
        indices1 = self._selection_indices(down1)
        self.register_buffer('human_indices', indices0[indices1].to(device), persistent=False)
        self.register_buffer('human_mid_indices', indices0.to(device), persistent=False)
        self.register_buffer('joint_regressor', torch.as_tensor(smplh.joint_regressor, device=device), persistent=False)
        self.register_buffer('up_coarse', smplh._U[1].to(device), persistent=False)
        self.register_buffer('up_dense', smplh._U[0].to(device), persistent=False)
        # The original getupsample now reuses these same immutable tensors.
        smplh._U = [self.up_dense, self.up_coarse]
        ids = list(range(len(obj_dict.obj_names)))
        self.register_buffer('canonical64', torch.stack([torch.as_tensor(obj_dict[i].load_verts()) for i in ids]).to(device), persistent=False)

    @staticmethod
    def _selection_indices(matrix):
        indices, values = matrix.indices(), matrix.values()
        if matrix._nnz() != matrix.shape[0] or not bool((values == 1).all()):
            raise ValueError('Human downsampling is not the expected fixed vertex selection')
        if not torch.equal(indices[0], torch.arange(matrix.shape[0], device=indices.device)):
            raise ValueError('Human downsampling row order differs')
        return indices[1]

    def coarse(self, human):
        return human[:, self.human_indices]

    def mid(self, human):
        return human[:, self.human_mid_indices]

    def joints(self, human):
        return self.joint_regressor @ human

    def upsample(self, delta):
        return self.up_dense @ (self.up_coarse @ delta)

    def pose_matrices(self, pose6d):
        from gcanet.reconstruction.utils.funcs_utils import rot6d_to_aa
        from smplx.lbs import batch_rodrigues
        layer = self._body_layer
        aa = rot6d_to_aa(pose6d.reshape(-1, 6)).reshape(len(pose6d), -1)
        # Native SMPLH.forward adds its nonzero hand pose_mean before LBS.
        return batch_rodrigues((aa + layer.pose_mean).reshape(-1, 3)).reshape(len(pose6d), 52, 3, 3)

    def posed_body(self, matrices, shape):
        from smplx.lbs import lbs
        layer = self._body_layer
        # smplx indexes a Python transform list with parents[i]. Keeping this
        # immutable integer topology on CPU avoids one CUDA scalar read for
        # every joint; matrix/vertex arithmetic remains on the original device.
        parents = getattr(self, '_speed_body_parents_cpu', layer.parents)
        vertices, joints = lbs(shape, matrices, layer.v_template, layer.shapedirs,
                              layer.posedirs, layer.J_regressor, parents,
                              layer.lbs_weights, pose2rot=False)
        return vertices - joints[:, :1]

    def enable_speed_equivalent_optimizations(self):
        parents = self._body_layer.parents.detach().cpu().clone()
        # Deliberately not buffers: model.to(cuda) must not move this cache.
        object.__setattr__(self, '_speed_body_parents_cpu', parents)
        object.__setattr__(self, '_speed_body_parent_indices', tuple(parents.tolist()))

class PairRelation(nn.Module):
    """All human-object edges, recomputed in chunks without distance pruning."""
    def __init__(self, width=256, edge_width=64, chunk=32):
        super().__init__()
        self.chunk = chunk
        self.human = nn.Linear(width, edge_width)
        self.object = nn.Linear(width, edge_width)
        self.geometry = nn.Linear(4, edge_width)
        self.history = nn.Linear(4, edge_width, bias=False)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(edge_width), nn.Linear(edge_width, 2 * edge_width),
                                                  nn.GELU(), nn.Linear(2 * edge_width, edge_width)) for _ in range(2)])
        self.score = nn.Linear(edge_width, 1)
        self.vector = nn.Linear(edge_width, 3)
        self.null_h = nn.Linear(width, 1)
        self.null_o = nn.Linear(width, 1)
        self.contact_h = nn.Linear(width, 1)
        self.contact_o = nn.Linear(width, 1)

    def _edges(self, h, o, relative, previous):
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        edge = h[:, :, None] + o[:, None] + self.geometry(torch.cat((relative / .1, distance / .1), -1))
        edge = edge + self.history(previous)
        for block in self.blocks:
            edge = edge + block(edge)
        return self.score(edge).squeeze(-1), .1 * self.vector(edge)

    def forward(self, human_features, object_features, human, rotation, translation, anchors, previous=None):
        h, o = self.human(human_features), self.object(object_features)
        local_human = (human - translation[:, None]) @ rotation
        relative = local_human[:, :, None] - anchors[:, None]
        scores, vectors = [], []
        edge_function = getattr(self, '_compiled_edges', self._edges)
        for start in range(0, human.shape[1], self.chunk):
            stop = min(start + self.chunk, human.shape[1])
            if previous is None:
                hist = relative.new_zeros((len(human), stop-start, anchors.shape[1], 4))
            else:
                hist = torch.cat((previous['relation_vectors'][:, start:stop] / .1,
                                  previous['relation_logits_h'][:, start:stop, :-1, None]), -1)
            score, vector = _checkpoint(edge_function, h[:, start:stop], o, relative[:, start:stop], hist)
            scores.append(score)
            vectors.append(vector)
        score = torch.cat(scores, 1)
        vector = torch.cat(vectors, 1)
        return dict(relation_logits_h=torch.cat((score, self.null_h(human_features)), -1),
                    relation_logits_o=torch.cat((score.transpose(1, 2), self.null_o(object_features)), -1),
                    relation_vectors=vector, current_relation=relative,
                    contact_logits_h=self.contact_h(human_features).squeeze(-1),
                    contact_logits_o=self.contact_o(object_features).squeeze(-1))

class ImageRead(nn.Module):
    def __init__(self, width=256, heads=8, chunk=128):
        super().__init__()
        self.heads, self.chunk = heads, chunk
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.null = nn.Linear(width, heads)
        self.visible = nn.Linear(width, 1)
        self.output = nn.Linear(width, width)

    def _read(self, query, key, memory):
        b, n, d = query.shape
        q = self.query(query).reshape(b, n, self.heads, d // self.heads)
        k = key.reshape(b, -1, self.heads, d // self.heads)
        score = torch.einsum('bnhd,bphd->bhnp', q, k) / math.sqrt(d // self.heads)
        null = self.null(query).permute(0, 2, 1)[..., None]
        probabilities = torch.cat((score, null), -1).softmax(-1).mean(1)
        observed = probabilities[..., :-1] @ memory
        return probabilities, observed

    def prepare_memory(self, memory):
        """Same differentiable memory projections shared within one forward."""
        return self.key(memory), self.value(memory)

    def forward(self, query, memory, prepared_memory=None):
        keys, values = self.prepare_memory(memory) if prepared_memory is None else prepared_memory
        probabilities, observations = [], []
        read_function = getattr(self, '_compiled_read', self._read)
        for start in range(0, query.shape[1], self.chunk):
            p, observed = _checkpoint(read_function, query[:, start:start+self.chunk], keys, values)
            probabilities.append(p)
            observations.append(observed)
        probabilities = torch.cat(probabilities, 1)
        visible = self.visible(query).sigmoid().squeeze(-1)
        # Native convention: joints are not surface-occlusion classification targets.
        visibility = torch.cat((torch.ones_like(visible[:, :73]), visible[:, 73:]), 1)
        observation = self.output(torch.cat(observations, 1)) * visibility[..., None]
        return observation, dict(log_probs=probabilities.clamp_min(1e-8).log(),
                                 visibility=visibility, query=query)

def _device_asset(asset, device):
    return {name: torch.as_tensor(value, device=device) if not torch.is_tensor(value)
            else value.to(device) for name, value in asset.items()}
