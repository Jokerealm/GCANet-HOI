"""Portable entry points and training/checkpoint contracts, without model assets."""
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from gcanet.config import load_config
from gcanet.io import external_state_hashes
from gcanet.training.objective import accumulation_weight, optimizer_for
from gcanet.training.runner import compact_payload, restore

ROOT = Path(__file__).resolve().parents[1]
RECIPE = dict(old_lr=1e-5, iacp_if_lr=5e-6, iacp_decoder_lr=1e-5,
              new_lr=1e-4, adam_betas=[.5, .999])


class TinyContact(nn.Module):
    def __init__(self):
        super().__init__()
        self.interactionformer = nn.Linear(2, 2)
        self.decoder = nn.Linear(2, 2)
        self.detector = nn.Linear(2, 2)
        self.detector.register_buffer("fixed_tag", torch.tensor(37))
        self.detector.requires_grad_(False)

    def trainable_groups(self):
        return {"interactionformer": list(self.interactionformer.parameters()),
                "decoder": list(self.decoder.parameters())}


class TinyAdaptation(nn.Module):
    def __init__(self):
        super().__init__()
        self.native = nn.Linear(2, 2)
        self.iacp = TinyContact()
        self.adaptation_refiner = nn.Linear(2, 1)
        self.register_buffer("adaptation_training_calls", torch.tensor(0))

    def forward(self, value):
        value = self.native(value) + self.iacp.detector(value)
        value = self.iacp.decoder(self.iacp.interactionformer(value))
        return self.adaptation_refiner(value)


class PortableEntryTests(unittest.TestCase):
    def process(self, *arguments):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH="")
        with tempfile.TemporaryDirectory() as directory:
            return subprocess.run([sys.executable, "-I", *map(str, arguments)],
                cwd=directory, env=environment, capture_output=True, text=True,
                timeout=60)

    def test_all_documented_entry_points_help_without_assets(self):
        for name in ("train_gsf.py", "train_iacp.py", "evaluate.py", "build_iacp_assets.py"):
            with self.subTest(entry=name):
                result = self.process(ROOT / "scripts" / name, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--output", result.stdout)

    def test_launchers_reject_multiple_gpu_ids_before_loading_assets(self):
        commands = {
            "train_gsf.py": ["--data-root", "missing", "--dino-checkpoint", "missing"],
            "train_iacp.py": ["--gsf-checkpoint", "missing"],
            "evaluate.py": ["--gsf-checkpoint", "missing", "--iacp-checkpoint", "missing"],
        }
        for name, arguments in commands.items():
            with self.subTest(entry=name):
                result = self.process(ROOT / "scripts" / name, *arguments,
                                      "--gpu", "0,1", "--output", "missing")
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("--gpu", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_imports_are_self_contained_from_an_unrelated_directory(self):
        script = r'''
import importlib
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
for name in ('gcanet.config', 'gcanet.io', 'gcanet.surface', 'gcanet.bcr',
             'gcanet.gsf', 'gcanet.iacp', 'gcanet.model', 'gcanet.evaluation',
             'gcanet.training.runner'):
    importlib.import_module(name)
bad = []
for name, module in list(sys.modules.items()):
    filename = getattr(module, '__file__', None)
    if filename:
        path = Path(filename).resolve()
        if root.parent in path.parents and root not in path.parents:
            bad.append((name, str(path)))
assert not bad, bad
for forbidden in ('strong_vendor', 'pihoc_vendor', 'core.config', 'models.templates'):
    assert forbidden not in sys.modules, forbidden
print('isolated imports passed')
'''
        result = self.process("-c", script, ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("isolated imports passed", result.stdout)


class ConfigurationTests(unittest.TestCase):
    def configuration(self):
        return dict(dataset="BEHAVE", data_root="data", dino_checkpoint="assets/dino.pth",
                    iacp_assets=dict(pretrained_checkpoint="assets/pihoc.pt",
                                     canonical_root="assets/canonical"))

    def test_relative_paths_follow_config_root_instead_of_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            path = root / "configs" / "behave.json"
            path.write_text(json.dumps(self.configuration()))
            config = load_config(path)
            self.assertEqual(config["data_root"], str(root / "data"))
            self.assertEqual(config["dino_checkpoint"], str(root / "assets/dino.pth"))
            self.assertEqual(config["iacp_assets"]["canonical_root"], str(root / "assets/canonical"))

    def test_absolute_asset_paths_are_not_rebased(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.configuration()
            config["data_root"] = str(root / "external-data")
            path = root / "custom.json"
            path.write_text(json.dumps(config))
            self.assertEqual(load_config(path)["data_root"], str(root / "external-data"))

    def test_dataset_scope_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.json"
            config = self.configuration()
            config["dataset"] = "InterCap"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "BEHAVE"):
                load_config(path)


class OptimizerAndAccumulationTests(unittest.TestCase):
    def test_every_trainable_parameter_is_updated_in_its_assigned_group(self):
        model = TinyAdaptation()
        optimizer = optimizer_for(model, RECIPE)
        rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
        self.assertEqual(rates, dict(native=1e-5, iacp_interactionformer=5e-6,
                                    iacp_decoder=1e-5, new=1e-4))
        original = {key: value.clone() for key, value in model.state_dict().items()}
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        for key, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertFalse(torch.equal(parameter, original[key]), key)
            else:
                self.assertTrue(torch.equal(parameter, original[key]), key)

    def test_unclassified_or_duplicate_contact_parameters_are_rejected(self):
        model = TinyAdaptation()
        model.iacp.extra = nn.Parameter(torch.ones(1))
        with self.assertRaisesRegex(ValueError, "Unclassified"):
            optimizer_for(model, RECIPE)
        model = TinyAdaptation()
        original_groups = model.iacp.trainable_groups
        def duplicate():
            result = original_groups()
            result["decoder"].append(result["interactionformer"][0])
            return result
        model.iacp.trainable_groups = duplicate
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            optimizer_for(model, RECIPE)

    def test_tail_accumulation_matches_the_actual_per_image_gradient(self):
        for total in (4, 8, 20, 48, 52, 100):
            micro, accumulate = 8, 6
            data = torch.arange(1, total + 1, dtype=torch.float64)
            weight = torch.tensor(.75, dtype=torch.float64, requires_grad=True)
            for step, start in enumerate(range(0, total, micro)):
                batch = data[start:start + micro]
                loss = (weight * batch).square().mean()
                factor = accumulation_weight(step, len(batch), total, micro, accumulate)
                (factor * loss).backward()
                if (step + 1) % accumulate == 0 or start + len(batch) == total:
                    first = (step // accumulate) * micro * accumulate
                    expected = 2 * weight.detach() * data[first:start + len(batch)].square().mean()
                    torch.testing.assert_close(weight.grad, expected, atol=1e-10, rtol=1e-12)
                    weight.grad = None


class AdaptationResumeTests(unittest.TestCase):
    def setUp(self):
        self.cuda_get = mock.patch("torch.cuda.get_rng_state_all", return_value=[])
        self.cuda_set = mock.patch("torch.cuda.set_rng_state_all")
        self.cuda_get.start(); self.cuda_set.start()
        self.addCleanup(self.cuda_get.stop); self.addCleanup(self.cuda_set.stop)
        random.seed(113); np.random.seed(113); torch.manual_seed(113)
        self.model = TinyAdaptation()
        self.optimizer = optimizer_for(self.model, RECIPE)
        self.step(self.model, self.optimizer, torch.randn(3, 2))
        self.external = external_state_hashes(self.model)
        self.config = dict(recipe=RECIPE, input_hash="example", gpu_count=1)

    @staticmethod
    def step(model, optimizer, values):
        optimizer.zero_grad(set_to_none=True)
        model(values).square().mean().backward()
        optimizer.step()

    def payload(self):
        return compact_payload(self.model, 2, self.config, self.external, self.optimizer,
                               image_visits=24, optimizer_updates=4)

    def save(self, directory, value):
        path = Path(directory) / "resume.pt"
        torch.save(value, path)
        return path

    def test_compact_checkpoint_excludes_frozen_parameters_and_buffers(self):
        saved = self.payload()
        self.assertIn("iacp.detector.fixed_tag", self.external)
        self.assertFalse(set(saved["model"]) & set(self.external))
        self.assertEqual(set(saved["model"]) | set(self.external), set(self.model.state_dict()))
        self.assertIn("adaptation_training_calls", saved["model"])

    def test_resume_reproduces_next_random_inputs_and_adam_update(self):
        with tempfile.TemporaryDirectory() as directory:
            resumed = copy.deepcopy(self.model)
            optimizer = optimizer_for(resumed, RECIPE)
            path = self.save(directory, self.payload())
            expected_python, expected_numpy = random.random(), np.random.random()
            expected_input = torch.randn(4, 2)
            self.step(self.model, self.optimizer, expected_input)
            with torch.no_grad():
                for parameter in resumed.parameters():
                    if parameter.requires_grad:
                        parameter.add_(5)
            start, visits, updates = restore(resumed, optimizer, path, self.config, self.external)
            self.assertEqual((start, visits, updates), (3, 24, 4))
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(np.random.random(), expected_numpy)
            actual_input = torch.randn(4, 2)
            self.assertTrue(torch.equal(actual_input, expected_input))
            self.step(resumed, optimizer, actual_input)
            for key, value in self.model.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[key]), key)

    def test_invalid_resume_is_rejected_before_model_mutation(self):
        cases = ("recipe", "frozen_hash", "missing_tensor", "dtype", "shape",
                 "missing_optimizer", "missing_rng", "epoch", "optimizer_groups")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                saved = copy.deepcopy(self.payload())
                key = "native.weight"
                saved["model"][key].add_(7)
                if case == "recipe": saved["config"]["gpu_count"] = 2
                elif case == "frozen_hash": saved["external_frozen_hashes"][next(iter(self.external))] = "0" * 64
                elif case == "missing_tensor": del saved["model"]["native.bias"]
                elif case == "dtype": saved["model"][key] = saved["model"][key].double()
                elif case == "shape": saved["model"][key] = saved["model"][key][:1]
                elif case == "missing_optimizer": del saved["optimizer"]
                elif case == "missing_rng": del saved["rng"]
                elif case == "epoch": saved["epoch"] = 50
                elif case == "optimizer_groups": saved["optimizer"]["param_groups"][0]["name"] = "wrong"
                before = {name: value.clone() for name, value in self.model.state_dict().items()}
                path = self.save(directory, saved)
                with self.assertRaises(ValueError):
                    restore(self.model, self.optimizer, path, self.config, self.external)
                for name, value in self.model.state_dict().items():
                    self.assertTrue(torch.equal(value, before[name]), (case, name))


class InferenceCompositionTests(unittest.TestCase):
    def test_paper_endpoint_rejects_nonfinal_predictors_before_inference(self):
        from gcanet.model import build_model
        from gcanet.training.model import build_adaptation_model
        config = dict(data_root="unused", dino_checkpoint="unused", iacp_assets={})
        with mock.patch("gcanet.reconstruction.runtime.initialize"), \
             mock.patch("gcanet.reconstruction.runtime.build_model", return_value=nn.Linear(1, 1)), \
             mock.patch("gcanet.io.load_torch", return_value={"epoch": 25}):
            with self.assertRaisesRegex(ValueError, "epoch-50"):
                build_model(config, "gsf.pt", "iacp.pt", device="cpu")
            with self.assertRaisesRegex(ValueError, "epoch-50"):
                build_adaptation_model("gsf.pt", None, {}, RECIPE, device="cpu")
        class NonfinalPrior:
            def __init__(self, *args, **kwargs): pass
            def load_adapted_checkpoint(self, path): return {"epoch": 25}
        with mock.patch("gcanet.reconstruction.runtime.initialize"), \
             mock.patch("gcanet.reconstruction.runtime.build_model", return_value=nn.Linear(1, 1)), \
             mock.patch("gcanet.reconstruction.checkpoint.load_model"), \
             mock.patch("gcanet.io.load_torch", return_value={"epoch": 50}), \
             mock.patch("gcanet.iacp.InstanceAwareContactPrior", NonfinalPrior):
            with self.assertRaisesRegex(ValueError, "epoch-50"):
                build_model(config, "gsf.pt", "iacp.pt", device="cpu")

    def test_pair_without_contact_preserves_geometry_and_freezes_predictors(self):
        from gcanet.model import GCANet
        human = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3) / 100
        rotation = torch.eye(3)[None]
        translation = torch.tensor([[.1, .2, .3]])
        class Reconstruction(nn.Module):
            def __init__(self):
                super().__init__(); self.weight = nn.Parameter(torch.ones(1))
            def forward(self, inputs):
                return dict(smpl_verts6000=human, obj_pose=rotation,
                            obj_trans=translation, cam_trans=torch.tensor([[0., 0., 3.]]))
        class Prior(nn.Module):
            def __init__(self):
                super().__init__(); self.weight = nn.Parameter(torch.ones(1))
            def forward(self, rgb, human_mask, object_mask):
                return dict(probabilities=torch.zeros(1, 8), coverage=torch.tensor([False]),
                            presence=torch.zeros(1))
        vertices = np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]], dtype=np.float32)
        mesh = types.SimpleNamespace(vertices=vertices, faces=np.array([[0, 1, 2]]))
        template = types.SimpleNamespace(load_template=lambda: mesh,
                                         load_verts=lambda: np.repeat(vertices[:1], 64, axis=0))
        def select(columns, width):
            indices = torch.stack((torch.arange(len(columns)), torch.tensor(columns)))
            return torch.sparse_coo_tensor(indices, torch.ones(len(columns)), (len(columns), width))
        fake = types.ModuleType("gcanet.reconstruction.models.templates")
        fake.smplh = types.SimpleNamespace(_D=[select([0, 2, 4, 6], 8), select([1, 3], 4)])
        fake.obj_dict = {0: template}
        with mock.patch.dict(sys.modules, {fake.__name__: fake}):
            model = GCANet(Reconstruction(), Prior())
        result = model(dict(img=torch.zeros(1, 5, 16, 16), obj_id=torch.tensor([0])))
        self.assertFalse(model.training)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        self.assertIs(result["smpl_verts6000"], human)
        self.assertTrue(torch.equal(result["obj_pose"], rotation))
        self.assertTrue(torch.equal(result["obj_trans"], translation))
        self.assertEqual(result["bcr_diagnostics"][0]["accepted_iterations"], 0)


if __name__ == "__main__":
    unittest.main()
