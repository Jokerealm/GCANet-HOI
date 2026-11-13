"""BEHAVE metric coordinates, empty-frame aggregation, and pipeline invariants."""
import types
import sys
import unittest
from unittest import mock
import numpy as np
import torch
from torch import nn
from gcanet.evaluation import contact_counts, surface_cd, paper_summary, evaluate_batch
from gcanet.metric_kernels import rigid_align
from gcanet.model import GCANet
from gcanet.bcr import BodyGuidedContactRefinement


class EvaluationTests(unittest.TestCase):
    def test_contact_threshold_is_strict_and_unknown_labels_are_ignored(self):
        human = np.array([[0.,0,0], [.049,0,0], [.05,0,0], [.1,0,0], [0,0,0]])
        obj = np.zeros((1,3))
        self.assertEqual(contact_counts(human,obj,np.array([1,0,1,0,-1])),dict(tp=1,fp=1,fn=1))

    def test_empty_frame_denominators_follow_paper(self):
        rows = [dict(tp=1,fp=1,fn=0,cd_human_cm=1.,cd_object_cm=2.),
                dict(tp=0,fp=0,fn=2,cd_human_cm=3.,cd_object_cm=4.),
                dict(tp=0,fp=1,fn=0,cd_human_cm=5.,cd_object_cm=6.)]
        answer = paper_summary(rows)
        self.assertEqual(answer['precision'], .25)
        self.assertEqual(answer['recall'], .5)
        self.assertAlmostEqual(answer['f1'], 1/3)
        self.assertEqual(answer['precision_valid_frames'],2)
        self.assertEqual(answer['recall_valid_frames'],2)
        self.assertEqual(answer['cd_human_cm'],3.)

    def test_joint_similarity_alignment_handles_scale_and_reflection(self):
        rng = np.random.default_rng(91)
        a = rng.normal(size=(19,3))
        r = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        b = 1.2*(a@r.T)+[.2,.1,-.3]
        np.testing.assert_allclose(rigid_align(a,b),b,atol=2e-15)

    def test_surface_sampling_is_repeatable_and_restores_rng(self):
        rng = np.random.default_rng(13)
        h = rng.normal(size=(12,3)).astype(np.float32)
        o = rng.normal(size=(12,3)).astype(np.float64)
        faces = np.arange(12).reshape(-1,3)
        before = np.random.get_state()
        first = surface_cd(h,h+.03,o,o+.01,faces,faces,123)
        after = np.random.get_state()
        self.assertEqual(before[0],after[0])
        np.testing.assert_array_equal(before[1],after[1])
        self.assertEqual(before[2:],after[2:])
        self.assertEqual(first,surface_cd(h,h+.03,o,o+.01,faces,faces,123))
        self.assertTrue(all(np.isfinite(value) and value>=0 for value in first.values()))

    def test_batch_uses_centered_human_and_raw_object_for_paper_metrics(self):
        fake = types.ModuleType('gcanet.reconstruction.models.templates')
        regression = np.zeros((1,6890),np.float32);regression[0,0]=1
        faces = np.array([[0,1,2]])
        fake.smplh = types.SimpleNamespace(joint_regressor=regression,root_joint_idx=0,faces=faces)
        mesh = types.SimpleNamespace(vertices=np.array([[0.,0,0],[.1,0,0],[0,.1,0]]),faces=faces)
        fake.obj_dict = {'object':types.SimpleNamespace(load_template=lambda:mesh)}
        human = np.zeros((1,6890,3),np.float32);human[:,:,0]=.25
        output=dict(smpl_verts6000=human,obj_pose=np.eye(3)[None],obj_trans=np.array([[.01,0,0]]))
        targets=dict(smpl_mesh_cam=np.zeros_like(human),obj_pose=np.eye(3)[None],obj_trans=np.zeros((1,3)),h_contacts=np.ones((1,6890)))
        metadata=dict(ann_id=np.array([9]),obj_name=['object'])
        def checked(ph,gh,po,go,hf,of,aid,seed):
            np.testing.assert_array_equal(ph,np.zeros((6890,3)))
            np.testing.assert_array_equal(po,mesh.vertices+np.array([.01,0,0]))
            return dict(cd_human_cm=0.,cd_object_cm=0.)
        with mock.patch.dict(sys.modules,{'gcanet.reconstruction.models.templates':fake}), mock.patch('gcanet.evaluation.surface_cd',side_effect=checked):
            row=evaluate_batch(output,targets,metadata)[0]
        self.assertEqual(row['tp'],6890)
        self.assertEqual(row['ann_id'],9)

    def test_inference_keeps_human_and_only_optimizes_object_pose(self):
        values=torch.tensor([-.1,0.,.1],dtype=torch.float32)
        y,x=torch.meshgrid(values,values,indexing='ij')
        human=torch.stack((x.flatten(),y.flatten(),torch.zeros(9)),dim=-1)[None]
        vertices=np.array([[-.5,-.5,0],[.5,-.5,0],[.5,.5,0],[-.5,.5,0]],np.float32)
        faces=np.array([[0,1,2],[0,2,3]])
        state=dict(smpl_verts6000=human,obj_pose=torch.eye(3)[None],obj_trans=torch.tensor([[0.,0.,.015]]),cam_trans=torch.tensor([[0.,0.,3.]]))
        class Reconstruction(nn.Module):
            def forward(self,inputs):return state
        class Contact(nn.Module):
            def forward(self,rgb,hmask,omask):
                return dict(probabilities=torch.full((1,9),.9),presence=torch.ones(1),coverage=torch.ones(1,dtype=torch.bool))
        model=GCANet.__new__(GCANet);nn.Module.__init__(model)
        model.reconstruction=Reconstruction();model.iacp=Contact();model.bcr=BodyGuidedContactRefinement()
        model.register_buffer('human_indices',torch.arange(9),persistent=False)
        model.objects={0:types.SimpleNamespace(load_template=lambda:types.SimpleNamespace(vertices=vertices,faces=faces),load_verts=lambda:vertices)}
        before={k:v.clone() for k,v in state.items()}
        result=model(dict(img=torch.zeros(1,5,16,16),obj_id=torch.tensor([0])))
        self.assertIs(result['smpl_verts6000'],human)
        self.assertGreater(result['bcr_diagnostics'][0]['accepted_iterations'],0)
        self.assertLess(result['obj_trans'][0,2],state['obj_trans'][0,2])
        for k in before:self.assertTrue(torch.equal(before[k],state[k]))
        self.assertEqual(vertices.shape,(4,3))


if __name__=='__main__':
    unittest.main()
