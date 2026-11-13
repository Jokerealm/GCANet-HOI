"""Exact surface-query numerical tests."""
import numpy as np
import unittest
import torch
from gcanet.surface import closest_surface

class TestCompleteClosestSurface(unittest.TestCase):
    def test_face_interior_edge_vertex_and_degenerate(self):
        vertices = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]], dtype=torch.float64)
        faces = torch.tensor([[0, 1, 2]])
        points = torch.tensor([[[.5, .5, .3], [1.5, 1.5, 0], [-1, -1, 0]]], dtype=torch.float64)
        result = closest_surface(points, vertices, faces, point_chunk=1, initial_candidates=1)
        expected = torch.tensor([[[.5, .5, 0], [1., 1., 0], [0., 0., 0]]], dtype=torch.float64)
        torch.testing.assert_close(result['points'], expected, rtol=0, atol=1e-12)
        result = closest_surface(points[:, :1], vertices, torch.tensor([[0, 1, 1]]))
        torch.testing.assert_close(result['points'], torch.tensor([[[.5, 0, 0]]], dtype=torch.float64))

    def test_long_triangle_face_beats_all_its_vertices(self):
        vertices = torch.tensor([[-100., -1., 0.], [100., -1., 0.], [0., 100., 0.],
                                 [.1, .1, 1.], [.2, .1, 1.], [.1, .2, 1.]])
        faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
        points = torch.tensor([[[0., 0., .01]]])
        result = closest_surface(points, vertices, faces, initial_candidates=1)
        self.assertEqual(result['face_index'].item(), 0)
        self.assertAlmostEqual(result['squared_distance'].item(), .0001, places=6)
        self.assertGreater(torch.cdist(points, vertices[None]).min().item(), .9)

    def test_bound_expands_past_misleading_triangle_box(self):
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 1.], [0., 1., 1.],
                                 [.7, .7, .5], [1., .7, .5], [.7, 1., .5]], dtype=torch.float64)
        faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
        points = torch.tensor([[[.8, .8, .6]]], dtype=torch.float64)
        result = closest_surface(points, vertices, faces, initial_candidates=1)
        self.assertEqual(result['face_index'].item(), 1)
        torch.testing.assert_close(result['points'], torch.tensor([[[.8, .8, .5]]], dtype=torch.float64))

    def test_independent_trimesh_naive_reference_and_rigid_equivariance(self):
        import trimesh
        rng = np.random.RandomState(21)
        vertices = rng.normal(size=(24, 3))
        faces = np.arange(24).reshape(-1, 3)
        points = rng.normal(size=(19, 3))
        expected, distance, _ = trimesh.proximity.closest_point_naive(
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False), points)
        v, p = torch.tensor(vertices, dtype=torch.float64), torch.tensor(points, dtype=torch.float64)[None]
        f = torch.tensor(faces)
        result = closest_surface(p, v, f, point_chunk=3, initial_candidates=1)
        np.testing.assert_allclose(result['points'][0].numpy(), expected, atol=1e-10)
        np.testing.assert_allclose(result['squared_distance'][0].sqrt().numpy(), distance, atol=1e-10)
        rotation = torch.tensor([[0., -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float64)
        translation = torch.tensor([2., -3., .7], dtype=torch.float64)
        moved = closest_surface(p @ rotation.T + translation, v @ rotation.T + translation, f, point_chunk=7)
        torch.testing.assert_close(moved['points'], result['points'] @ rotation.T + translation)
        torch.testing.assert_close(moved['squared_distance'], result['squared_distance'])
