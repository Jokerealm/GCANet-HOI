"""CPU tests for GSF geometry, checkpoint integrity, and tail-batch training."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gcanet.gsf import GeometryQueries
from gcanet.reconstruction.checkpoint import load_model, payload
from gcanet.reconstruction.utils.funcs_utils import soft_argmax_2d, soft_argmax_3d


class GSFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_query_geometry_uses_common_translation_frame(self):
        torch.manual_seed(3)
        query = GeometryQueries().eval()
        points = torch.randn(1, 568, 3)
        canonical = torch.randn(1, 568, 3)
        entity = torch.nn.functional.one_hot(torch.tensor([0]*73+[1]*431+[2]*64),3).float()[None]
        mask = query.neighborhood(canonical)
        with torch.no_grad():
            before = query(points, canonical, entity, mask)
            after = query(points + torch.tensor([.1,.2,.3]), canonical, entity, mask)
        torch.testing.assert_close(before, after, rtol=2e-5, atol=2e-6)
        self.assertEqual(tuple(before.shape), (1,568,256))
        # The local masks prevent human/object cross-entity retrieval; global
        # layers still communicate between them.
        self.assertTrue(bool(mask[:, :504, 504:].all()))
        self.assertTrue(bool(mask[:, 504:, :504].all()))

    def test_query_geometry_gradient_is_finite(self):
        torch.manual_seed(5)
        query = GeometryQueries().eval()
        points = torch.randn(1,568,3,requires_grad=True)
        canonical = torch.randn(1,568,3)
        entity = torch.nn.functional.one_hot(torch.tensor([0]*73+[1]*431+[2]*64),3).float()[None]
        result = query(points, canonical, entity, query.neighborhood(canonical))
        result[...,0].square().mean().backward()
        self.assertTrue(bool(torch.isfinite(points.grad).all()))
        self.assertGreater(float(points.grad.abs().sum()),0)
        self.assertGreater(float(query.input[0].weight.grad.abs().sum()),0)

    def test_soft_argmax_cpu_and_gradient(self):
        two = torch.zeros(1,2,4,6,requires_grad=True)
        three = torch.zeros(1,2,3,4,6,requires_grad=True)
        result2, result3 = soft_argmax_2d(two), soft_argmax_3d(three)
        torch.testing.assert_close(result2, torch.tensor([[[2.5,1.5],[2.5,1.5]]]))
        torch.testing.assert_close(result3, torch.tensor([[[2.5,1.5,1.],[2.5,1.5,1.]]]))
        (result2.sum()+result3.sum()).backward()
        self.assertTrue(bool(torch.isfinite(two.grad).all()))
        self.assertTrue(bool(torch.isfinite(three.grad).all()))

    def test_compact_checkpoint_rejects_wrong_external_backbone(self):
        model = nn.Module()
        model.evidence=nn.Module();model.evidence.dino=nn.Module()
        model.evidence.dino.backbone=nn.Linear(2,2).requires_grad_(False)
        model.evidence.dino.provenance={'weight_sha256':'test-backbone'}
        model.head=nn.Linear(2,1)
        saved=payload(model,50,{'mode':'dino_single'})
        self.assertFalse(any(k.startswith('evidence.dino.backbone.') for k in saved['model']))
        load_model(model,saved)
        saved['external_dino_sha256']='wrong-backbone'
        with self.assertRaises(ValueError):load_model(model,saved)

    def test_compact_checkpoint_rejects_missing_trainable_weight(self):
        model=nn.Module();model.head=nn.Linear(2,1)
        saved=payload(model,50,{'mode':'dino_single'})
        saved['model'].pop('head.bias')
        with self.assertRaises(ValueError):load_model(model,saved)

    def test_final_partial_accumulation(self):
        spec=importlib.util.spec_from_file_location('train_gsf',ROOT/'scripts/train_gsf.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        batches=(45380+7)//8
        self.assertEqual(batches,5673)
        self.assertEqual(module.accumulation_divisor(5671,batches,2),2)
        self.assertEqual(module.accumulation_divisor(5672,batches,2),1)
        self.assertEqual((batches+1)//2,2837)


if __name__=='__main__':unittest.main()
