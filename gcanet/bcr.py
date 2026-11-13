"""Body-guided Contact Refinement (BCR), GCANet Sec. 3.4.

Only object rotation and translation change. All distances are in meters.
The main method uses confidence-weighted contact and an initial-pose prior.
"""
import math
import numpy as np
import torch

DEFAULTS = dict(
    probability_threshold=.7, presence_threshold=.5, max_initial_distance=.08,
    minimum_contacts=6, max_contacts=128, max_dense_contacts=64,
    iterations=6, robust_scale=.02, baseline_anchor_weight=.25, damping=1e-6,
    max_step_translation=.005, max_step_rotation_degrees=1.,
    camera_focal=1000., camera_epsilon=1e-4,
    minimum_camera_depth=.05, line_search=(1., .5, .25, .125),
    improvement_tolerance=1e-12,
)


def configuration(config=None):
    value = dict(DEFAULTS)
    if config:
        unknown = set(config) - set(value)
        if unknown:
            raise ValueError('Unknown BCR configuration: ' + repr(sorted(unknown)))
        value.update(config)
    for key in ('minimum_contacts', 'max_contacts', 'max_dense_contacts', 'iterations'):
        if not isinstance(value[key], int) or value[key] <= 0:
            raise ValueError('Invalid positive integer: ' + key)
    if not value['minimum_contacts'] <= value['max_contacts']:
        raise ValueError('Contact minimum exceeds maximum')
    if not value['max_dense_contacts'] < value['max_contacts']:
        raise ValueError('Dense budget must leave a coarse contact budget')
    for key in ('probability_threshold', 'presence_threshold'):
        if not 0 <= value[key] <= 1:
            raise ValueError('Probability threshold outside [0,1]')
    for key in ('max_initial_distance', 'robust_scale', 'damping',
                'max_step_translation', 'max_step_rotation_degrees',
                'camera_focal', 'minimum_camera_depth'):
        if not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError('Invalid positive bound: ' + key)
    if not math.isfinite(value['baseline_anchor_weight']) or value['baseline_anchor_weight'] < 0:
        raise ValueError('Invalid baseline anchor weight')
    if not value['line_search'] or any(not 0 < a <= 1 for a in value['line_search']):
        raise ValueError('Invalid line-search schedule')
    return value


def _skew(v):
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*v.shape[:-1], 3, 3)


def _exp(v):
    angle = torch.linalg.vector_norm(v)
    k = _skew(v)
    # torch.sinc is stable at the identity; no random/learned operations.
    a = torch.sinc(angle / math.pi)
    b = .5 * torch.sinc(angle / (2 * math.pi)).square()
    return torch.eye(3, device=v.device, dtype=v.dtype) + a * k + b * (k @ k)


def _angle(rotation):
    # atan2 of skew/trace is stable near identity (unlike acos of rounded trace).
    vector = torch.stack((rotation[2, 1] - rotation[1, 2],
                          rotation[0, 2] - rotation[2, 0],
                          rotation[1, 0] - rotation[0, 1])) * .5
    return torch.atan2(torch.linalg.vector_norm(vector), ((torch.trace(rotation) - 1) / 2).clamp(-1, 1))


def _spread_indices(points, probabilities, indices, count):
    """Deterministic confidence-weighted farthest sampling, stable ID ties."""
    indices = torch.unique(indices, sorted=True)
    if len(indices) <= count:
        return indices
    xyz, p = points[indices], probabilities[indices]
    slot = int(torch.argmax(p))
    selected = []
    distances = p.new_full((len(indices),), float('inf'))
    available = torch.ones(len(indices), device=p.device, dtype=torch.bool)
    for _ in range(count):
        selected.append(indices[slot])
        available[slot] = False
        distances = torch.minimum(distances, (xyz - xyz[slot]).square().sum(-1))
        score = (distances * p).masked_fill(~available, -1.)
        slot = int(torch.argmax(score))
    return torch.stack(selected)


def select_contact_indices(human, probabilities, human_indices, config=None):
    """At most64 coarse +64 extra dense predicted contacts under defaults."""
    cfg = configuration(config)
    coarse = torch.unique(torch.as_tensor(human_indices, device=human.device, dtype=torch.long), sorted=True)
    if coarse.ndim != 1 or bool((coarse < 0).any()) or bool((coarse >= len(human)).any()):
        raise ValueError('Invalid coarse human indices')
    coarse = coarse[probabilities[coarse] >= cfg['probability_threshold']]
    coarse = _spread_indices(human, probabilities, coarse,
                             cfg['max_contacts'] - cfg['max_dense_contacts'])
    dense = torch.where(probabilities >= cfg['probability_threshold'])[0]
    if len(coarse):
        remaining = torch.ones(len(human), device=human.device, dtype=torch.bool)
        remaining[coarse] = False
        dense = dense[remaining[dense]]
    dense = _spread_indices(human, probabilities, dense, cfg['max_dense_contacts'])
    return torch.cat((coarse, dense)), len(coarse), len(dense)


def _projection(points, camera, cfg):
    xyz = points + camera
    depth = xyz[:, 2] + cfg['camera_epsilon']
    if not bool(torch.isfinite(xyz).all()) or bool((depth <= cfg['minimum_camera_depth']).any()):
        return None
    return cfg['camera_focal'] * xyz[:, :2] / depth[:, None]


def _surface(human, rotation, translation, vertices, faces):
    from .surface import closest_surface
    local = (human - translation) @ rotation
    answer = closest_surface(local[None], vertices, faces)
    local_points = answer['points'][0]
    world = local_points @ rotation.T + translation
    return local_points, world, answer['squared_distance'][0]


def _objective(human, weights, rotation, translation, vertices, faces,
               controls, baseline_controls, cfg):
    local, closest, distance2 = _surface(human, rotation, translation, vertices, faces)
    scale2 = cfg['robust_scale'] ** 2
    robust_distance = scale2 * (torch.sqrt(1 + distance2 / scale2) - 1)
    contact = (weights * robust_distance).sum() / weights.sum()
    control_world = controls @ rotation.T + translation
    prior = .5 * cfg['baseline_anchor_weight'] * (control_world - baseline_controls).square().sum(-1).mean()
    return contact + prior, contact, local, closest, distance2, control_world


@torch.no_grad()
def refine_object(human, rotation, translation, probabilities, coverage, presence,
                  vertices, faces, human_indices, object_control_points, camera,
                  config=None):
    """Return (R,t,diagnostics), without gradients or input mutation.

    Inputs are one sample: Hx3 human, 3x3 object R, 3 object t, H probabilities,
    Vx3 canonical CAD, Fx3 faces, Kx3 canonical control points, 3 camera t.
    Original tensors are returned verbatim when no update is accepted.
    """
    cfg = configuration(config)
    original_rotation, original_translation = rotation, translation
    if human.ndim != 2 or human.shape[1] != 3 or probabilities.shape != (len(human),):
        raise ValueError('Human and probability shapes differ')
    if rotation.shape != (3, 3) or translation.shape != (3,) or camera.shape != (3,):
        raise ValueError('Pose/camera shape mismatch')
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('Malformed CAD triangles')
    if object_control_points.ndim != 2 or object_control_points.shape[1] != 3 or not len(object_control_points):
        raise ValueError('Object controls must be nonempty Kx3')
    tensors = (human, rotation, translation, probabilities, vertices, object_control_points, camera)
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError('Nonfinite input geometry or probability')
    if bool((probabilities < 0).any()) or bool((probabilities > 1).any()):
        raise ValueError('Probabilities outside [0,1]')
    report = dict(eligible=False, accepted_iterations=0, selected_contacts=0,
                  initial_valid_contacts=0, candidate_trace=[],
                  human_geometry_unchanged=True, network_training_updates=0,
                  optimized_parameters='object_rotation_translation_only',
                  correspondence='triangles',
                  anchor_weight=cfg['baseline_anchor_weight'], solver_configuration=cfg,
                  objective='predicted_fixed_support_pseudo_Huber_surface_distance_plus_initial_control_anchor',
                  penetration_certified=False,
                  projection_units='native_normalized_projection_times256_focal1000_virtual_pixels',
                  total_translation_m=0., total_rotation_degrees=0., max_projection_displacement=0.)
    if not bool(coverage) or float(presence) < cfg['presence_threshold']:
        report['reason'] = 'unmatched_instance_or_low_presence'
        return original_rotation, original_translation, report
    indices, coarse_count, dense_count = select_contact_indices(human, probabilities, human_indices, cfg)
    report.update(selected_contacts=len(indices), selected_coarse_contacts=coarse_count,
                  selected_extra_dense_contacts=dense_count)
    if len(indices) < cfg['minimum_contacts']:
        report['reason'] = 'insufficient_high_confidence_contacts'
        return original_rotation, original_translation, report
    device = human.device
    h = human[indices].double()
    weights = probabilities[indices].double()
    r0, t0 = rotation.double(), translation.double()
    r, t = r0.clone(), t0.clone()
    vertices = vertices.to(device=device, dtype=torch.float64)
    faces = faces.to(device=device, dtype=torch.long)
    controls = object_control_points.to(device=device, dtype=torch.float64)
    camera = camera.to(device=device, dtype=torch.float64)
    baseline_controls = controls @ r0.T + t0
    baseline_projection = _projection(baseline_controls, camera, cfg)
    if baseline_projection is None:
        report['reason'] = 'invalid_initial_projection_depth'
        return original_rotation, original_translation, report
    _, _, distance2 = _surface(h, r, t, vertices, faces)
    valid = distance2 <= cfg['max_initial_distance'] ** 2
    report['initial_valid_contacts'] = int(valid.sum())
    if int(valid.sum()) < cfg['minimum_contacts']:
        report['reason'] = 'insufficient_near_surface_contacts'
        return original_rotation, original_translation, report
    # The same indices and probabilities define every candidate objective.
    h, weights, indices = h[valid], weights[valid], indices[valid]
    report['support_vertex_ids'] = indices.cpu().tolist()
    current = _objective(h, weights, r, t, vertices, faces, controls, baseline_controls, cfg)
    report.update(eligible=True, initial_objective=float(current[0]),
                  initial_contact_objective=float(current[1]),
                  initial_mean_surface_distance_m=float(torch.sqrt(current[4]).mean()),
                  initial_max_surface_distance_m=float(torch.sqrt(current[4]).max()))
    identity = torch.eye(3, device=device, dtype=torch.float64)
    for iteration in range(cfg['iterations']):
        objective, contact, local, nearest, distance2, control_world = current
        contact_jacobian = torch.cat((-_skew(local @ r.T), identity.expand(len(local), -1, -1)), -1)
        robust = torch.rsqrt(1 + distance2 / cfg['robust_scale'] ** 2)
        w = weights * robust / weights.sum()
        residual = nearest - h
        normal = torch.einsum('nki,n,nkj->ij', contact_jacobian, w, contact_jacobian)
        rhs = -torch.einsum('nki,n,nk->i', contact_jacobian, w, residual)
        prior_j = torch.cat((-_skew(controls @ r.T), identity.expand(len(controls), -1, -1)), -1)
        prior_w = cfg['baseline_anchor_weight'] / len(controls)
        normal += prior_w * torch.einsum('nki,nkj->ij', prior_j, prior_j)
        rhs -= prior_w * torch.einsum('nki,nk->i', prior_j, control_world - baseline_controls)
        normal += cfg['damping'] * torch.eye(6, device=device, dtype=torch.float64)
        delta = torch.linalg.solve(normal, rhs)
        if not bool(torch.isfinite(delta).all()):
            report['reason'] = 'nonfinite_step_rejected'
            break
        angle = torch.linalg.vector_norm(delta[:3])
        move = torch.linalg.vector_norm(delta[3:])
        factor = min(1., math.radians(cfg['max_step_rotation_degrees']) / max(float(angle), 1e-30),
                     cfg['max_step_translation'] / max(float(move), 1e-30))
        delta = delta * factor
        accepted = False
        for alpha in cfg['line_search']:
            rr = _exp(delta[:3] * alpha) @ r
            tt = t + delta[3:] * alpha
            rotation_size = float(_angle(rr @ r0.T))
            translation_size = float(torch.linalg.vector_norm(tt - t0))
            entry = dict(iteration=iteration + 1, alpha=alpha, accepted=False,
                         translation_m=translation_size, rotation_degrees=math.degrees(rotation_size))
            projected = _projection(controls @ rr.T + tt, camera, cfg)
            if projected is None:
                entry['reason'] = 'invalid_projection_depth'
                report['candidate_trace'].append(entry)
                continue
            projection_displacement = float(torch.linalg.vector_norm(projected - baseline_projection, dim=-1).max())
            entry['max_projection_displacement'] = projection_displacement
            candidate = _objective(h, weights, rr, tt, vertices, faces, controls, baseline_controls, cfg)
            entry.update(objective=float(candidate[0]), contact_objective=float(candidate[1]),
                         mean_surface_distance_m=float(torch.sqrt(candidate[4]).mean()))
            if float(candidate[1]) > float(contact) + cfg['improvement_tolerance']:
                entry['reason'] = 'contact_objective_increased'
            elif float(candidate[0]) >= float(objective) - cfg['improvement_tolerance']:
                entry['reason'] = 'no_objective_improvement'
            elif bool((candidate[4] > cfg['max_initial_distance'] ** 2).any()):
                entry['reason'] = 'fixed_support_outside_distance_gate'
            else:
                r, t, current = rr, tt, candidate
                entry.update(accepted=True, reason='accepted')
                accepted = True
                report['accepted_iterations'] += 1
                report['max_projection_displacement'] = projection_displacement
            report['candidate_trace'].append(entry)
            if accepted:
                break
        if not accepted:
            report.setdefault('reason', 'no_acceptable_step')
            break
    report.update(final_objective=float(current[0]), final_contact_objective=float(current[1]),
                  final_mean_surface_distance_m=float(torch.sqrt(current[4]).mean()),
                  final_max_surface_distance_m=float(torch.sqrt(current[4]).max()),
                  total_translation_m=float(torch.linalg.vector_norm(t - t0)),
                  total_rotation_degrees=math.degrees(float(_angle(r @ r0.T))))
    if report['accepted_iterations']:
        report['reason'] = 'accepted_update'
        return r.to(original_rotation), t.to(original_translation), report
    return original_rotation, original_translation, report


class BodyGuidedContactRefinement:
    """GCANet object-pose refinement with the paper's default configuration."""

    def __init__(self, config=None):
        self.config = configuration(config)

    def __call__(self, human, rotation, translation, probabilities, coverage, presence,
                 vertices, faces, human_indices, object_control_points, camera):
        return refine_object(human, rotation, translation, probabilities, coverage,
                             presence, vertices, faces, human_indices,
                             object_control_points, camera, config=self.config)
