"""Deterministic surface identities for the IACP joint adaptation.

No dataset or model imports: construction happens only after the native runtime
has loaded its audited meshes.  Sampling is area-uniform on complete triangles,
and every sample retains its face and barycentric coordinates.
"""
import hashlib
import json

import numpy as np
import torch
from torch import nn


def sample_surface(vertices, faces, count, seed):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Surface vertices must be finite Nx3')
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError('Surface faces must be nonempty Fx3')
    if np.any(faces < 0) or np.any(faces >= len(vertices)) or count <= 0:
        raise ValueError('Invalid surface indices or sample count')
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_area = np.linalg.norm(cross, axis=-1)
    total_area = twice_area.sum() * .5
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError('A surface must have positive finite area')
    rng = np.random.RandomState(int(seed) % (2 ** 32))
    face_index = rng.choice(len(faces), size=count, p=twice_area / twice_area.sum())
    random = rng.random_sample((count, 2))
    root = np.sqrt(random[:, 0])
    barycentric = np.stack((1 - root, root * (1 - random[:, 1]), root * random[:, 1]), axis=-1)
    points = (triangles[face_index] * barycentric[:, :, None]).sum(1)
    normals = cross[face_index] / twice_area[face_index, None]
    return dict(points=points.astype(np.float32), face_index=face_index,
                barycentric=barycentric.astype(np.float32), normals=normals.astype(np.float32),
                area_weights=np.full(count, total_area / count, dtype=np.float32),
                total_area=float(total_area))


def _interpolation(points, canonical, neighbors=4):
    squared = ((points[:, None].astype(np.float64) - canonical[None].astype(np.float64)) ** 2).sum(-1)
    neighbors = min(neighbors, len(canonical))
    indices = np.argsort(squared, axis=-1, kind='stable')[:, :neighbors]
    distance = np.sqrt(np.take_along_axis(squared, indices, axis=-1))
    weights = 1. / np.maximum(distance, 1e-8)
    exact = distance[:, 0] <= 1e-8
    weights[exact] = 0
    weights[exact, 0] = 1
    weights /= weights.sum(-1, keepdims=True)
    return indices, weights.astype(np.float32)


class SurfaceAssets(nn.Module):
    """Device-movable, checkpointed fixed sampling and complete mesh buffers."""
    def __init__(self, object_meshes, human_faces, seed=20261024, anchor_count=512, loss_count=2048):
        super().__init__()
        self.seed, self.anchor_count, self.loss_count = int(seed), int(anchor_count), int(loss_count)
        if hasattr(object_meshes, 'obj_num'):
            objects = {i: object_meshes[i] for i in range(object_meshes.obj_num)}
        else:
            objects = {int(k): v for k, v in object_meshes.items()}
        self.object_ids = sorted(objects)
        self.id_to_row = {oid: row for row, oid in enumerate(self.object_ids)}
        if not self.object_ids:
            raise ValueError('No object meshes supplied')
        batches = {key: [] for key in ('anchors', 'normals', 'area_weights', 'anchor_faces',
                   'anchor_barycentric', 'loss_points', 'loss_faces', 'loss_barycentric',
                   'loss_area_weights', 'canonical64', 'interp64_idx', 'interp64_weights')}
        hashes = {}
        for oid in self.object_ids:
            mesh = objects[oid]
            vertices = np.asarray(mesh['vertices'] if isinstance(mesh, dict) else mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh['faces'] if isinstance(mesh, dict) else mesh.faces, dtype=np.int64)
            canonical = np.asarray(mesh['canonical64'] if isinstance(mesh, dict) else mesh.load_verts(), dtype=np.float32)
            if canonical.ndim != 2 or canonical.shape[-1] != 3 or not np.isfinite(canonical).all():
                raise ValueError('Invalid canonical object keypoints')
            anchors = sample_surface(vertices, faces, anchor_count, seed + oid * 1009)
            loss = sample_surface(vertices, faces, loss_count, seed + oid * 1009 + 1000003)
            indices, weights = _interpolation(anchors['points'], canonical)
            values = dict(anchors=anchors['points'], normals=anchors['normals'],
                          area_weights=anchors['area_weights'], anchor_faces=anchors['face_index'],
                          anchor_barycentric=anchors['barycentric'], loss_points=loss['points'],
                          loss_faces=loss['face_index'], loss_barycentric=loss['barycentric'],
                          loss_area_weights=loss['area_weights'], canonical64=canonical,
                          interp64_idx=indices, interp64_weights=weights)
            for key, value in values.items():
                batches[key].append(value)
            self.register_buffer('vertices_%d' % oid, torch.from_numpy(vertices.astype(np.float32)))
            self.register_buffer('faces_%d' % oid, torch.from_numpy(faces.copy()))
            digest = hashlib.sha256()
            digest.update(vertices.astype('<f8').tobytes()); digest.update(faces.astype('<i8').tobytes())
            digest.update(canonical.astype('<f4').tobytes())
            hashes[str(oid)] = digest.hexdigest()
        for key, values in batches.items():
            self.register_buffer(key, torch.from_numpy(np.stack(values)))
        human_faces = np.asarray(human_faces, dtype=np.int64)
        if human_faces.ndim != 2 or human_faces.shape[1] != 3 or len(human_faces) == 0 or np.any(human_faces < 0):
            raise ValueError('Invalid human triangle topology')
        edges = np.concatenate((human_faces[:, [0, 1]], human_faces[:, [1, 2]], human_faces[:, [2, 0]]))
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        directed = np.concatenate((edges, edges[:, ::-1]))
        self.register_buffer('human_faces', torch.from_numpy(human_faces.copy()))
        self.register_buffer('human_edges', torch.from_numpy(edges.copy()))
        self.register_buffer('human_adjacency_source', torch.from_numpy(directed[:, 0].copy()))
        self.register_buffer('human_adjacency_target', torch.from_numpy(directed[:, 1].copy()))
        degree = np.bincount(directed[:, 0], minlength=int(human_faces.max()) + 1).astype(np.float32)
        self.register_buffer('human_degree', torch.from_numpy(degree))
        self.provenance = dict(seed=self.seed, anchor_count=self.anchor_count, loss_count=self.loss_count,
            sampling='independent fixed iid triangle-area CDF and sqrt-barycentric; no process-global RNG',
            interpolation='4 nearest canonical keypoints, inverse-distance convex weights',
            object_mesh_sha256=hashes, human_faces_sha256=hashlib.sha256(human_faces.astype('<i8').tobytes()).hexdigest())
        self.provenance['configuration_sha256'] = hashlib.sha256(
            json.dumps(self.provenance, sort_keys=True).encode()).hexdigest()

    def batch(self, object_ids):
        ids = object_ids.detach().cpu().reshape(-1).tolist() if torch.is_tensor(object_ids) else list(object_ids)
        try:
            rows = [self.id_to_row[int(oid)] for oid in ids]
        except KeyError as error:
            raise ValueError('Unknown object identity') from error
        rows = torch.tensor(rows, dtype=torch.long, device=self.anchors.device)
        keys = ('anchors', 'normals', 'area_weights', 'anchor_faces', 'anchor_barycentric',
                'loss_points', 'loss_faces', 'loss_barycentric', 'loss_area_weights',
                'canonical64', 'interp64_idx', 'interp64_weights')
        return {key: getattr(self, key).index_select(0, rows) for key in keys}

    @classmethod
    def from_native(cls, object_meshes, smplh, **kwargs):
        return cls(object_meshes, smplh.faces, **kwargs)

    def manifest(self):
        result = json.loads(json.dumps(self.provenance))
        result['buffers'] = {}
        for key, tensor in self.named_buffers():
            value = tensor.detach().cpu().contiguous().numpy()
            result['buffers'][key] = dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())
        result['all_buffers_sha256'] = hashlib.sha256(
            json.dumps(result['buffers'], sort_keys=True).encode()).hexdigest()
        return result

    def mesh(self, object_id):
        oid = int(object_id)
        if oid not in self.id_to_row:
            raise ValueError('Unknown object identity')
        return getattr(self, 'vertices_%d' % oid), getattr(self, 'faces_%d' % oid)
