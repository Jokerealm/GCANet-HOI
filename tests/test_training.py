"""Numerical and gradient contracts for IACP's coupled training graph."""
import math
import types
import unittest
import numpy as np
import torch
from gcanet.training import geometry as MODEL
from gcanet.training.geometry import so3_exp, skew
from gcanet.training.refiner import (damped_normal_solve, lbs_vertex_jacobian,
    object_point_jacobian, project, projection_jacobian,
    ContactAdaptationRefiner, perturb_training_state, tangent_coordinate_scales)

def tiny_body(dtype=torch.float64):
    rng = torch.Generator().manual_seed(913)
    vertices, joints, shapes = 11, 4, 2
    layer = types.SimpleNamespace(
        v_template=torch.randn(vertices, 3, generator=rng, dtype=dtype) * .2,
        shapedirs=torch.randn(vertices, 3, shapes, generator=rng, dtype=dtype) * .02,
        posedirs=torch.randn((joints - 1) * 9, vertices * 3, generator=rng, dtype=dtype) * .01,
        J_regressor=torch.rand(joints, vertices, generator=rng, dtype=dtype),
        lbs_weights=torch.rand(vertices, joints, generator=rng, dtype=dtype),
        parents=torch.tensor([-1, 0, 1, 1]))
    layer.J_regressor /= layer.J_regressor.sum(-1, keepdim=True)
    layer.lbs_weights /= layer.lbs_weights.sum(-1, keepdim=True)
    shape = torch.randn(2, shapes, generator=rng, dtype=dtype) * .1
    pose = so3_exp(torch.randn(2, joints, 3, generator=rng, dtype=dtype) * .2)
    return layer, pose, shape

def independent_body(layer, pose, shape):
    from smplx.lbs import lbs
    vertices, joints = lbs(shape, pose, layer.v_template, layer.shapedirs,
        layer.posedirs, layer.J_regressor, layer.parents, layer.lbs_weights, pose2rot=False)
    return vertices - joints[:, :1]

class GeometryDerivativeTests(unittest.TestCase):
    def test_lbs_jacobian_against_actual_smplx_central_difference(self):
        layer, pose, shape = tiny_body()
        indices = torch.tensor([1, 4, 7, 9])
        calculated = lbs_vertex_jacobian(layer, pose, shape, indices, controls=4)
        eps = 1e-6
        columns = []
        for dimension in range(12):
            delta = torch.zeros(2, 4, 3, dtype=pose.dtype)
            delta[:, dimension // 3, dimension % 3] = eps
            plus = independent_body(layer, so3_exp(delta) @ pose, shape)
            minus = independent_body(layer, so3_exp(-delta) @ pose, shape)
            columns.append(((plus - minus) / (2 * eps))[:, indices])
        numerical = torch.stack(columns, -1)
        self.assertTrue(torch.allclose(calculated, numerical, atol=2e-9, rtol=2e-7),
                        float((calculated - numerical).abs().max()))

    def test_lbs_root_and_descendant_are_both_present(self):
        layer, pose, shape = tiny_body()
        jacobian = lbs_vertex_jacobian(layer, pose, shape, torch.arange(11), controls=4)
        self.assertGreater(float(jacobian[..., :3].abs().max()), .01)
        self.assertGreater(float(jacobian[..., 9:].abs().max()), .001)
        self.assertFalse(jacobian.requires_grad)

    def test_projection_jacobian_including_clamped_depth(self):
        points = torch.tensor([[[.2, -.1, 1.], [.1, .3, -.2]]], dtype=torch.float64)
        camera = torch.tensor([[.03, -.02, .1]], dtype=torch.float64)
        actual = projection_jacobian(points, camera)
        numeric = []
        for axis in range(3):
            step = torch.zeros_like(points); step[..., axis] = 1e-6
            numeric.append((project(points + step, camera) - project(points - step, camera)) / 2e-6)
        self.assertTrue(torch.allclose(actual, torch.stack(numeric, -1), atol=1e-8, rtol=1e-7))

    def test_object_point_jacobian_agrees_with_real_rotation_action(self):
        rotation = so3_exp(torch.tensor([[.2, -.1, .3]], dtype=torch.float64))
        anchors = torch.tensor([[[.3, -.5, .1], [-.1, .7, .2]]], dtype=torch.float64)
        analytic = object_point_jacobian(rotation, anchors)
        columns = []
        for axis in range(6):
            d = torch.zeros(1, 6, dtype=torch.float64); d[:, axis] = 1e-6
            plus = anchors @ (so3_exp(d[:, :3]) @ rotation).transpose(1, 2) + d[:, None, 3:]
            minus = anchors @ (so3_exp(-d[:, :3]) @ rotation).transpose(1, 2) - d[:, None, 3:]
            columns.append((plus - minus) / 2e-6)
        self.assertTrue(torch.allclose(analytic, torch.stack(columns, -1), atol=1e-9, rtol=1e-7))

    def test_canonical_contact_object_jacobian(self):
        rotation = so3_exp(torch.tensor([[.2, -.1, .3]], dtype=torch.float64))
        human = torch.tensor([[[.2, .5, -.1], [.3, .1, .2]]], dtype=torch.float64)
        translation = torch.tensor([[.1, -.1, .2]], dtype=torch.float64)
        rt = rotation.transpose(-1, -2)
        analytic = torch.cat((rt[:, None] @ skew(human - translation[:, None]),
                               -rt[:, None].expand(-1, 2, -1, -1)), -1)
        numeric = []
        for axis in range(6):
            d = torch.zeros(1, 6, dtype=torch.float64); d[:, axis] = 1e-6
            plus = (human - (translation + d[:, 3:])[:, None]) @ (so3_exp(d[:, :3]) @ rotation)
            minus = (human - (translation - d[:, 3:])[:, None]) @ (so3_exp(-d[:, :3]) @ rotation)
            numeric.append((plus - minus) / 2e-6)
        self.assertTrue(torch.allclose(analytic, torch.stack(numeric, -1), atol=1e-9, rtol=1e-7))

class SolverTests(unittest.TestCase):
    def test_solves_two_sided_gap_and_protects_stiff_side(self):
        # Gap + h - o = 0; human stiff means object must absorb the correction.
        j = torch.tensor([[[1., -1.]]], dtype=torch.float64)
        r = torch.tensor([[1.]], dtype=torch.float64)
        w = torch.ones_like(r)
        result, condition = damped_normal_solve(j, r, w,
            torch.tensor([[100., .01]], dtype=torch.float64))
        self.assertLess(abs(float(result[0, 0])), .001)
        self.assertGreater(float(result[0, 1]), .98)
        self.assertLess(abs(float(r + (j @ result[..., None])[..., 0])), .02)
        self.assertTrue(torch.isfinite(condition).all())

    def test_no_relationship_evidence_means_no_invented_update(self):
        j = torch.randn(2, 10, 6, dtype=torch.float64)
        r = torch.randn(2, 10, dtype=torch.float64)
        delta, _ = damped_normal_solve(j, r, torch.zeros_like(r), torch.ones(2, 6, dtype=torch.float64))
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))

    def test_gradients_flow_to_residual_and_stiffness(self):
        j = torch.tensor([[[1., -1.], [.3, .4]]], dtype=torch.float64)
        residual = torch.tensor([[.2, .1]], dtype=torch.float64, requires_grad=True)
        stiffness = torch.tensor([[.3, .2]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda r, p: damped_normal_solve(j, r, torch.ones_like(r), p)[0],
            (residual, stiffness), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_bad_weights_raise_instead_of_identity_fallback(self):
        with self.assertRaises(ValueError):
            damped_normal_solve(torch.ones(1, 1, 1), torch.ones(1, 1),
                                 -torch.ones(1, 1), torch.ones(1, 1))

class IntegrationGeometry:
    def __init__(self):
        rng = torch.Generator().manual_seed(59)
        v, j = 431, 52
        self._body_layer = types.SimpleNamespace(
            v_template=torch.randn(v, 3, generator=rng) * .1,
            shapedirs=torch.randn(v, 3, 10, generator=rng) * .005,
            posedirs=torch.randn((j - 1) * 9, v * 3, generator=rng) * .0001,
            J_regressor=torch.rand(j, v, generator=rng),
            lbs_weights=torch.rand(v, j, generator=rng),
            parents=torch.tensor([-1] + [0] * (j - 1)))
        self._body_layer.J_regressor /= self._body_layer.J_regressor.sum(-1, keepdim=True)
        self._body_layer.lbs_weights /= self._body_layer.lbs_weights.sum(-1, keepdim=True)
        self.human_indices = torch.arange(v)
    def coarse(self, human): return human
    def joints(self, human): return human[:, :73]
    def pose_matrices(self, pose): return pose
    def posed_body(self, pose, shape): return independent_body(self._body_layer, pose, shape)

class FullRefinerIntegrationTests(unittest.TestCase):

    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(15)
        self.geometry = IntegrationGeometry()
        pose = torch.eye(3)[None, None].expand(1, 52, 3, 3).clone()
        shape = torch.zeros(1, 10)
        human = self.geometry.posed_body(pose, shape)
        self.native = dict(smpl_verts6000=human, initial_smpl_mesh=human, obj_pose=torch.eye(3)[None], obj_trans=torch.tensor([[0.2, 0.0, 0.0]]), initial_smpl_pose=pose, initial_smpl_shape=shape, refh3d_keypoints=human[:, :73], cam_trans=torch.tensor([[0.0, 0.0, 3.0]]))
        self.tokens = torch.randn(1, 568, 256, requires_grad=True)
        self.memory = torch.randn(1, 1024, 256, requires_grad=True)
        self.contact = dict(probabilities=torch.rand(1, 6890, requires_grad=True), pair_features=torch.randn(1, 1024, requires_grad=True), presence=torch.tensor([0.8], requires_grad=True))
        self.asset = dict(anchors=torch.randn(1, 12, 3) * 0.1, normals=torch.randn(1, 12, 3), interp64_idx=torch.randint(64, (1, 12, 4)), interp64_weights=torch.full((1, 12, 4), 0.25))

    def run_refiner(self, model):
        return model(self.native, self.tokens, self.memory, self.asset, self.geometry, self.contact)

    def test_solver_runs_under_no_grad_and_is_rigid(self):
        model = ContactAdaptationRefiner(edge_chunk=128).eval()
        with torch.no_grad():
            rounds = self.run_refiner(model)
        self.assertEqual(len(rounds), 3)
        rotation = rounds[-1]['rotation']
        self.assertTrue(torch.allclose(rotation @ rotation.transpose(-1, -2), torch.eye(3)[None], atol=2e-06))
        self.assertTrue(torch.isfinite(rounds[-1]['human']).all())
        self.assertGreater(float(rounds[-1]['solver']['condition'].min()), 0)

    def test_geometry_gradient_reaches_native_features_and_pair_not_contact_weights(self):
        model = ContactAdaptationRefiner(edge_chunk=128)
        rounds = self.run_refiner(model)
        loss = rounds[-1]['human'].square().mean() + rounds[-1]['translation'].square().mean()
        loss.backward()
        for value in (self.tokens, self.memory, self.contact['pair_features']):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0)
        self.assertIsNone(self.contact['probabilities'].grad)
        self.assertIsNone(self.contact['presence'].grad)
        for layer in (model.body_stiffness, model.object_stiffness):
            self.assertIsNotNone(layer.weight.grad)
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(float(layer.weight.grad.abs().sum()), 0)

    def test_three_intervention_types_leave_the_unperturbed_side_exact(self):
        truth = dict(smpl_verts=self.native['smpl_verts6000'], obj_pose=self.native['obj_pose'], obj_trans=self.native['obj_trans'])
        modes = set()
        for step in range(30):
            state, masks = perturb_training_state(self.native, truth, self.geometry, torch.tensor([123]), step)
            mode = int(masks['mode'][0])
            modes.add(mode)
            if bool(masks['protect_human'][0]):
                self.assertTrue(torch.allclose(state['human'], truth['smpl_verts'], atol=2e-08, rtol=0))
            if bool(masks['protect_object'][0]):
                self.assertTrue(torch.equal(state['rotation'], truth['obj_pose']))
                self.assertTrue(torch.equal(state['translation'], truth['obj_trans']))
        self.assertEqual(modes, {0, 1, 2, 3})

class PoseFitTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_historical_scaled_translation_not_unit_scale(self):
        a = torch.randn(2, 64, 3, dtype=torch.double) * .1 + torch.tensor([.04, -.07, .03])
        r = MODEL.so3_exp(torch.tensor([[.2, -.3, .1], [-.1, .4, .6]], dtype=torch.double))
        t = torch.tensor([[.1, .2, .3], [-.2, .1, .05]], dtype=torch.double)
        scale = torch.tensor([.8, 1.3], dtype=torch.double)
        b = scale[:, None, None] * (a @ r.transpose(1, 2)) + t[:, None]
        got_r, got_t, got_c = MODEL.historical_pose_fit(a, b, return_scale=True)
        torch.testing.assert_close(got_r, r, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(got_t, t, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(got_c, scale, atol=1e-12, rtol=1e-12)
        unit_t = b.mean(1) - (got_r @ a.mean(1)[..., None]).squeeze(-1)
        self.assertGreater(float((unit_t - got_t).abs().max()), .005)

    def test_numpy_donor_reflection_case(self):
        a = np.random.default_rng(7).normal(size=(64, 3)) * [.3, .2, .1] + [.02, .03, -.04]
        b = a * [-1., 1., 1.] + [.1, .3, -.2]
        ac, bc = a-a.mean(0), b-b.mean(0)
        u, s, vh = np.linalg.svd(ac.T @ bc / len(a))
        r = vh.T @ u.T
        if np.linalg.det(r) < 0:
            s[-1] = -s[-1]
            vh[2] = -vh[2]
            r = vh.T @ u.T
        c = s.sum() / a.var(0).sum()
        t = b.mean(0) - c * r @ a.mean(0)
        gr, gt, gc = MODEL.historical_pose_fit(torch.tensor(a), torch.tensor(b), True)
        np.testing.assert_allclose(gr.numpy(), r, atol=1e-12)
        np.testing.assert_allclose(gt.numpy(), t, atol=1e-12)
        self.assertAlmostEqual(float(gc), c, places=12)

    def test_complete_fit_gradient(self):
        a = torch.randn(1, 8, 3, dtype=torch.double) + .4
        b = torch.randn_like(a, requires_grad=True)
        def fit(x):
            r, t = MODEL.historical_pose_fit(a, x)
            return torch.cat((r.flatten(), t.flatten()))
        self.assertTrue(torch.autograd.gradcheck(fit, b, atol=1e-5, rtol=1e-4))

    def test_repeated_singular_values_have_valid_gradient(self):
        matrix = torch.eye(3, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(MODEL._ProperPolar.apply, matrix))

    def test_collinear_ambiguity_is_not_silent(self):
        with self.assertRaisesRegex(FloatingPointError, 'Ambiguous'):
            MODEL._ProperPolar.apply(torch.diag(torch.tensor([1., 0., 0.], dtype=torch.double)))

    def test_zero_rotation_gradient(self):
        omega = torch.zeros(2, 3, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(MODEL.so3_exp, omega))

    def test_legacy_audit_uses_complete_mesh_not_only_pose(self):
        class Assets:
            def mesh(self, oid):
                return torch.tensor([[0.,0.,0.],[3.,0.,0.],[0.,3.,0.]]), torch.tensor([[0,1,2]])
        rotation=torch.eye(3)[None]
        translation=torch.zeros(1,3)
        same=MODEL.compare_legacy_pose_mapping(rotation,translation,rotation,translation,[0],Assets())
        self.assertTrue(same['passed'])
        self.assertEqual(same['samples'][0]['complete_cad_vertices'],3)
        # Entry-wise R difference is below tolerance, but a long CAD amplifies it.
        altered=MODEL.so3_exp(torch.tensor([[0.,0.,1e-6]]))
        changed=MODEL.compare_legacy_pose_mapping(altered,translation,rotation,translation,[0],Assets())
        self.assertLess(changed['rotation_max_absolute'],2e-6)
        self.assertGreater(changed['complete_cad_max_vertex_difference_m'],2e-6)
        self.assertFalse(changed['passed'])
