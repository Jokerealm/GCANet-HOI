"""Exact closest points on a complete object triangle surface."""
import torch

def _closest_on_triangles(points, triangles):
    """Broadcast P[...,3] against triangles[...,3,3]; return closest points."""
    a, b, c = triangles.unbind(-2)
    ab, ac, ap = b - a, c - a, points - a
    normal = torch.cross(ab, ac, dim=-1)
    denominator = (normal * normal).sum(-1)
    d00, d01, d11 = (ab * ab).sum(-1), (ab * ac).sum(-1), (ac * ac).sum(-1)
    d20, d21 = (ap * ab).sum(-1), (ap * ac).sum(-1)
    # Scale-relative degeneracy: collapsed triangles are handled by their edges.
    tiny = torch.finfo(points.dtype).tiny
    safe = denominator.clamp_min(tiny)
    v, w = (d11 * d20 - d01 * d21) / safe, (d00 * d21 - d01 * d20) / safe
    inside = (denominator > torch.finfo(points.dtype).eps ** 2 * (d00 * d11).clamp_min(tiny))
    inside = inside & (v >= 0) & (w >= 0) & (v + w <= 1)
    projected = a + v[..., None] * ab + w[..., None] * ac
    choices = []
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        fraction = ((points - start) * edge).sum(-1) / (edge * edge).sum(-1).clamp_min(tiny)
        choices.append(start + fraction.clamp(0, 1)[..., None] * edge)
    candidates = torch.stack(choices, dim=-2)
    squared = ((points[..., None, :] - candidates) ** 2).sum(-1)
    index = squared.argmin(-1)
    nearest_edge = candidates.gather(-2, index[..., None, None].expand(*index.shape, 1, 3)).squeeze(-2)
    return torch.where(inside[..., None], projected, nearest_edge)


@torch.no_grad()
def closest_surface(points, vertices, faces, point_chunk=64, initial_candidates=32):
    """Exact batched closest points on a complete triangle mesh.

    points BxNx3, vertices BxVx3 or Vx3; faces shared Fx3.  Temporary
    lower bounds are Bxpoint_chunkxF, not BxNxFx3.  CPU is supported for tests.
    Degenerate faces remain valid line/point primitives.  Ties choose a valid
    minimum, without imposing a semantic label on the face index.
    """
    if points.ndim != 3 or points.shape[-1] != 3 or len(points) == 0:
        raise ValueError('points must be nonempty BxNx3')
    if not points.is_floating_point() or points.dtype not in (torch.float32, torch.float64):
        raise ValueError('Closest-surface labels require float32/float64')
    if point_chunk <= 0 or initial_candidates <= 0:
        raise ValueError('Chunk sizes must be positive')
    if vertices.ndim == 2:
        vertices = vertices[None].expand(len(points), -1, -1)
    if vertices.ndim != 3 or vertices.shape[0] != points.shape[0] or vertices.shape[-1] != 3:
        raise ValueError('vertices must share the query batch')
    if faces.ndim != 2 or faces.shape[-1] != 3 or len(faces) == 0:
        raise ValueError('faces must be nonempty Fx3')
    if not torch.isfinite(points).all() or not torch.isfinite(vertices).all():
        raise ValueError('Nonfinite complete-surface inputs')
    vertices = vertices.to(device=points.device, dtype=points.dtype)
    faces = faces.to(device=points.device, dtype=torch.long)
    triangles = vertices[:, faces]
    low, high = triangles.amin(-2), triangles.amax(-2)
    answers, distances, face_ids = [], [], []
    batch_count, face_count = len(points), len(faces)
    for offset in range(0, points.shape[1], point_chunk):
        query = points[:, offset:offset + point_chunk]
        count = query.shape[1]
        lower = query.new_zeros((batch_count, count, face_count))
        for axis in range(3):
            coordinate = query[:, :, axis, None]
            outside = (low[:, None, :, axis] - coordinate).clamp_min(0)
            outside = outside + (coordinate - high[:, None, :, axis]).clamp_min(0)
            lower.add_(outside.square())
        flat_query = query.reshape(-1, 3)
        flat_lower = lower.reshape(-1, face_count)
        batch_indices = torch.arange(batch_count, device=points.device).repeat_interleave(count)
        unresolved = torch.arange(len(flat_query), device=points.device)
        best_points = torch.empty_like(flat_query)
        best_distance = query.new_full((len(flat_query),), float('inf'))
        best_face = torch.empty((len(flat_query),), dtype=torch.long, device=points.device)
        candidates = min(initial_candidates, face_count)
        while unresolved.numel():
            bounds, indices = flat_lower[unresolved].topk(min(candidates + 1, face_count), largest=False, sorted=True)
            selected = indices[:, :candidates]
            selected_triangles = triangles[batch_indices[unresolved, None], selected]
            nearest = _closest_on_triangles(flat_query[unresolved, None], selected_triangles)
            squared = (nearest - flat_query[unresolved, None]).square().sum(-1)
            value, slot = squared.min(-1)
            best_points[unresolved] = nearest[torch.arange(len(slot), device=points.device), slot]
            best_distance[unresolved] = value
            best_face[unresolved] = selected.gather(1, slot[:, None]).squeeze(1)
            if candidates == face_count:
                break
            # Conservative rounding margin avoids pruning a potentially closer
            # face.  It affects work only, not the returned exact distance.
            margin = 64 * torch.finfo(query.dtype).eps * (1 + value)
            unresolved = unresolved[bounds[:, candidates] <= value + margin]
            candidates = min(face_count, candidates * 2)
        answers.append(best_points.reshape(batch_count, count, 3))
        distances.append(best_distance.reshape(batch_count, count))
        face_ids.append(best_face.reshape(batch_count, count))
    return dict(points=torch.cat(answers, 1), squared_distance=torch.cat(distances, 1),
                face_index=torch.cat(face_ids, 1))


