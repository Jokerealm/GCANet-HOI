"""Strict portable IACP state loading, without allocating the full network."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
from gcanet.iacp import InstanceAwareContactPrior


def digest(value):
    h = hashlib.sha256()
    h.update(str((tuple(value.shape), str(value.dtype), str(value.layout))).encode())
    h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def tiny_prior():
    model = InstanceAwareContactPrior.__new__(InstanceAwareContactPrior)
    nn.Module.__init__(model)
    model.detector = nn.Linear(2, 2)
    model.query_proj = nn.Linear(2, 3)
    model.detector.requires_grad_(False)
    return model


class IACPCheckpointTests(unittest.TestCase):
    def test_standalone_compact_and_legacy_export_are_strict(self):
        for legacy in [False, True]:
            model = tiny_prior()
            current = model.state_dict()
            prefix = 'pihoc_contact.' if legacy else ''
            trainable = {prefix+k: torch.ones_like(v) for k, v in current.items() if not k.startswith('detector.')}
            fixed = {prefix+k: digest(v) for k, v in current.items() if k.startswith('detector.')}
            payload = dict(format='pihoc_joint_compact_v1' if legacy else 'gcanet_iacp_v1',
                           model=trainable, external_frozen_hashes=fixed, epoch=50)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'iacp.pt'
                torch.save(payload, path)
                audit = model.load_adapted_checkpoint(path)
            self.assertEqual(audit['epoch'], 50)
            self.assertTrue(torch.equal(model.query_proj.weight, torch.ones_like(model.query_proj.weight)))

    def test_wrong_frozen_checkpoint_is_rejected(self):
        model = tiny_prior()
        state = model.state_dict()
        payload = dict(format='gcanet_iacp_v1', model={k:v for k,v in state.items() if not k.startswith('detector.')},
                       external_frozen_hashes={k:'0'*64 for k in state if k.startswith('detector.')})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'iacp.pt'
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'Fixed pretrained'):
                model.load_adapted_checkpoint(path)

    def test_missing_and_extra_learned_state_are_rejected(self):
        for extra in [False, True]:
            model = tiny_prior()
            state = model.state_dict()
            if extra:
                state['unexpected'] = torch.zeros(1)
            else:
                del state['query_proj.bias']
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'iacp.pt'
                torch.save({'model':state}, path)
                with self.assertRaisesRegex(ValueError, 'keys'):
                    model.load_adapted_checkpoint(path)
