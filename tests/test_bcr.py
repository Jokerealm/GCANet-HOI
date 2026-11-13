"""BCR geometry, fallback, fixed-human, and nonlinear-descent contracts."""
import unittest
import torch
from gcanet.bcr import BodyGuidedContactRefinement, refine_object, select_contact_indices, _exp


def fixture(offset=.015):
    values = torch.tensor([-.1, 0., .1], dtype=torch.float64)
    y, x = torch.meshgrid(values, values, indexing='ij')
    human = torch.stack((x.flatten(), y.flatten(), torch.zeros(9, dtype=torch.float64)), -1)
    vertices = torch.tensor([[-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0]], dtype=torch.float64)
    return dict(human=human, rotation=torch.eye(3, dtype=torch.float64),
                translation=torch.tensor([0., 0., offset], dtype=torch.float64),
                probabilities=torch.full((9,), .9, dtype=torch.float64),
                coverage=True, presence=.95, vertices=vertices,
                faces=torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long),
                human_indices=torch.arange(9), object_control_points=vertices.clone(),
                camera=torch.tensor([0., 0., 3.], dtype=torch.float64))


class BodyGuidedContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_surface_alignment_preserves_inputs_and_rigidity(self):
        args = fixture()
        before = {k: v.clone() for k, v in args.items() if torch.is_tensor(v)}
        r, t, d = BodyGuidedContactRefinement()(**args)
        self.assertGreater(d['accepted_iterations'], 0)
        self.assertLess(d['final_mean_surface_distance_m'], .5 * d['initial_mean_surface_distance_m'])
        self.assertLess(d['final_objective'], d['initial_objective'])
        torch.testing.assert_close(r.T @ r, torch.eye(3, dtype=r.dtype), rtol=0, atol=1e-10)
        self.assertAlmostEqual(float(torch.det(r)), 1., places=10)
        for k, v in before.items():
            self.assertTrue(torch.equal(args[k], v), k)
        self.assertAlmostEqual(d['initial_mean_surface_distance_m'], .015, places=10)

    def test_aligned_and_rejected_inputs_return_original_tensors(self):
        cases = [fixture(0.), dict(fixture(), coverage=False), dict(fixture(), presence=.49), fixture(.2)]
        for args in cases:
            r, t, d = refine_object(**args)
            self.assertIs(r, args['rotation'])
            self.assertIs(t, args['translation'])
            self.assertEqual(d['accepted_iterations'], 0)

    def test_insufficient_confidence_and_invalid_camera_fallback(self):
        args = fixture()
        args['probabilities'][:4] = .69
        self.assertEqual(refine_object(**args)[2]['reason'], 'insufficient_high_confidence_contacts')
        args = fixture()
        args['camera'][2] = -.2
        self.assertEqual(refine_object(**args)[2]['reason'], 'invalid_initial_projection_depth')

    def test_main_method_has_step_caps_without_total_pose_cap(self):
        args = fixture(.07)
        _, t, d = refine_object(**args)
        # Six 5 mm updates can exceed the historical 20 mm cumulative bound.
        self.assertGreater(float(torch.linalg.vector_norm(t-args['translation'])), .02)
        previous_move = 0.
        for row in d['candidate_trace']:
            if row['accepted']:
                self.assertLessEqual(abs(row['translation_m']-previous_move), .00500000001)
                previous_move = row['translation_m']
        self.assertLessEqual(d['accepted_iterations'], 6)

    def test_acceptance_checks_true_refreshed_energy(self):
        _, _, d = refine_object(**fixture(.04))
        energy, contact = d['initial_objective'], d['initial_contact_objective']
        for row in d['candidate_trace']:
            if row['accepted']:
                self.assertLess(row['objective'], energy-1e-12)
                self.assertLessEqual(row['contact_objective'], contact+1e-12)
                energy, contact = row['objective'], row['contact_objective']
        self.assertLessEqual(d['final_max_surface_distance_m'], .08)

    def test_contact_sampling_is_deterministic_and_disjoint(self):
        torch.manual_seed(5)
        points = torch.randn(6890, 3)
        probabilities = torch.rand(6890)
        ids = torch.arange(431)
        selected, coarse, dense = select_contact_indices(points, probabilities, ids)
        repeated, _, _ = select_contact_indices(points, probabilities, ids)
        self.assertTrue(torch.equal(selected, repeated))
        self.assertEqual(len(torch.unique(selected)), len(selected))
        self.assertLessEqual(coarse, 64)
        self.assertLessEqual(dense, 64)
        self.assertTrue(bool((probabilities[selected] >= .7).all()))

    def test_rotation_jacobian_matches_central_differences(self):
        from gcanet.bcr import _skew
        torch.manual_seed(9)
        point = torch.randn(3, dtype=torch.float64)
        r = _exp(torch.tensor([.2, -.1, .1], dtype=torch.float64))
        expected = -_skew(r @ point)
        columns = []
        for axis in range(3):
            v = torch.zeros(3, dtype=torch.float64)
            v[axis] = 1e-6
            columns.append((_exp(v) @ r @ point - _exp(-v) @ r @ point) / 2e-6)
        torch.testing.assert_close(torch.stack(columns, 1), expected, rtol=0, atol=2e-9)

    def test_unknown_experiment_switches_rejected(self):
        for patch in ({'boundary_bundle': True}, {'surface_mode': 'vertices'}):
            with self.assertRaises(ValueError):
                BodyGuidedContactRefinement(patch)


if __name__ == '__main__':
    unittest.main()
