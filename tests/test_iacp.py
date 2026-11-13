"""Meaningful tests for targets-free association, sparse labels and attention."""
import inspect
from pathlib import Path
import sys
import unittest
import torch
from torch import nn
from gcanet.iacp import InstanceAwareContactPrior, CanonicalProjector, mask_boxes, pair_attention_bias
from gcanet.vendor.pihoc.dinov2.layers.attention import MemEffAttention

def tiny_projector():
    p=CanonicalProjector.__new__(CanonicalProjector);nn.Module.__init__(p)
    p.register_buffer('pixels',torch.tensor([0,2]))
    p.register_buffer('vertices',torch.tensor([[0,1,2],[1,2,3]]))
    p.register_buffer('weights',torch.tensor([[.2,.3,.5],[.1,.6,.3]]))
    d=torch.zeros(6890)
    for k in range(3):d.scatter_add_(0,p.vertices[:,k],p.weights[:,k])
    p.register_buffer('denominator',d);p.register_buffer('support',d>0)
    p._build_reduction()
    return p


class ContactTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(7);torch.set_num_threads(2)
    def test_forward_accepts_no_targets(self):
        self.assertEqual(list(inspect.signature(InstanceAwareContactPrior.forward).parameters),['self','rgb','human_mask','object_mask'])
    def test_mask_box_normalization_and_empty(self):
        x=torch.zeros(2,10,20);x[0,2:7,4:15]=1
        box,valid=mask_boxes(x)
        torch.testing.assert_close(box[0],torch.tensor([.2,.2,.75,.7]))
        self.assertEqual(valid.tolist(),[True,False]);self.assertEqual(box[1].sum(),0)
    def test_pair_attention_does_not_consume_other_pairs(self):
        a=MemEffAttention(16,4,qkv_bias=True).eval()
        x=torch.randn(1,7,16) # two pair tokens, CLS, four patches
        mask=torch.zeros(7,7,dtype=torch.bool);mask[:3,:3]=~torch.eye(3,dtype=torch.bool);mask[2:,:2]=True
        bias=torch.zeros(1,1,7,7).masked_fill(mask,torch.finfo(torch.float32).min)
        full=a(x,bias);single=a(torch.cat((x[:,:1],x[:,2:]),1),pair_attention_bias(4,'cpu'))
        torch.testing.assert_close(full[:,0],single[:,0],rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(full[:,2:],single[:,1:],rtol=1e-5,atol=1e-6)
    def test_sdpa_matches_explicit_masked_attention(self):
        a=MemEffAttention(16,4,qkv_bias=True).eval();x=torch.randn(2,6,16)
        bias=pair_attention_bias(4,'cpu');q,k,v=a.qkv(x).reshape(2,6,3,4,4).unbind(2)
        q,k,v=[t.transpose(1,2) for t in (q,k,v)]
        expected=((q@k.transpose(-1,-2))*.5+bias).softmax(-1)@v
        expected=a.proj(expected.transpose(1,2).reshape(2,6,16))
        torch.testing.assert_close(a(x,bias),expected,rtol=1e-5,atol=1e-6)
    def test_projector_preserves_probabilities_and_gradient(self):
        p=tiny_projector();logits=torch.tensor([[[[0.,10.,2.,-10.]]]],requires_grad=True)
        result=p(logits)
        self.assertEqual(result.shape,(1,6890));self.assertAlmostEqual(float(result[0,0]),.5,6)
        self.assertAlmostEqual(float(result[0,3]),float(torch.sigmoid(torch.tensor(2.))),6)
        result.sum().backward();self.assertEqual(logits.grad[0,0,0,1],0);self.assertGreater(abs(float(logits.grad[0,0,0,0])),0)
    def test_ordered_projector_matches_reference_and_adjoint(self):
        p=tiny_projector();logits=torch.randn(2,1,1,4,requires_grad=True)
        reference=logits.sigmoid().flatten(1)[:,p.pixels];expected=reference.new_zeros(2,6890)
        for corner in range(3):
            expected.scatter_add_(1,p.vertices[:,corner].expand(2,-1),reference*p.weights[:,corner])
        expected=expected/p.denominator.clamp_min(1e-8)
        actual=p(logits);upstream=torch.randn_like(actual)
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
        expected_grad=torch.autograd.grad((expected*upstream).sum(),logits,retain_graph=True)[0]
        actual_grad=torch.autograd.grad((actual*upstream).sum(),logits)[0]
        torch.testing.assert_close(actual_grad,expected_grad,rtol=1e-6,atol=1e-7)
        self.assertTrue(torch.equal(p(logits),p(logits)))
    def test_unknown_vertices_do_not_become_negative_pixels(self):
        p=tiny_projector();labels=torch.zeros(1,6890);labels[0,:3]=1
        known=torch.zeros_like(labels,dtype=torch.bool);known[0,:3]=True
        target,valid=p.known_view_targets(labels,known)
        self.assertEqual(target.tolist(),[[1.,0.]]);self.assertEqual(valid.tolist(),[[True,False]])
    def test_sparse_contact_loss_leaves_unknown_logits_unsupervised(self):
        m=InstanceAwareContactPrior.__new__(InstanceAwareContactPrior);nn.Module.__init__(m);m.projector=tiny_projector()
        logits=torch.zeros(1,6890,requires_grad=True);views=torch.randn(1,1,1,4,requires_grad=True)
        output=dict(raw_probabilities=logits.sigmoid(),view_logits=views,coverage=torch.ones(1,dtype=torch.bool),presence_logits=torch.zeros(1,requires_grad=True))
        labels=torch.zeros_like(logits);known=torch.zeros_like(labels,dtype=torch.bool);known[0,0]=True;labels[0,0]=1
        losses=m.contact_loss(output,labels,known)
        sum(v for v in losses.values() if v.requires_grad).backward()
        self.assertEqual(losses['contact_valid_vertices'],1);self.assertEqual(losses['contact_valid_pixels'],0)
        self.assertNotEqual(logits.grad[0,0],0);self.assertEqual(logits.grad[0,1:].abs().sum(),0)
    def test_absent_detection_retains_zero_loss_and_coverage(self):
        m=InstanceAwareContactPrior.__new__(InstanceAwareContactPrior);nn.Module.__init__(m);m.projector=tiny_projector()
        logits=torch.randn(1,6890,requires_grad=True)
        output=dict(raw_probabilities=logits.sigmoid(),view_logits=torch.randn(1,1,1,4,requires_grad=True),coverage=torch.zeros(1,dtype=torch.bool),presence_logits=torch.zeros(1,requires_grad=True))
        losses=m.contact_loss(output,torch.zeros_like(logits))
        value=sum(v for v in losses.values() if v.requires_grad)
        self.assertEqual(float(value),0);value.backward();self.assertEqual(logits.grad.abs().sum(),0)
    def test_presence_uses_annotation_beyond_canonical_visibility(self):
        m=InstanceAwareContactPrior.__new__(InstanceAwareContactPrior);nn.Module.__init__(m);m.projector=tiny_projector()
        logits=torch.zeros(2,6890,requires_grad=True);presence=torch.zeros(2,requires_grad=True)
        output=dict(raw_probabilities=logits.sigmoid(),view_logits=torch.zeros(2,1,1,4,requires_grad=True),coverage=torch.ones(2,dtype=torch.bool),presence_logits=presence)
        labels=torch.zeros_like(logits);known=torch.zeros_like(labels,dtype=torch.bool)
        labels[0,100]=1;known[0,100]=True  # positive label outside rendered support
        known[1]=True  # complete negative annotation still proves absence
        losses=m.contact_loss(output,labels,known)
        self.assertEqual(losses['contact_valid_presence'],2)
        losses['contact_presence'].backward()
        self.assertLess(presence.grad[0],0);self.assertGreater(presence.grad[1],0)

if __name__=='__main__':unittest.main()
