"""Frozen DINOv2 features and identity-aware 3D queries, read once per image."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from gcanet import gsf_supervision as common

ROOT=Path(__file__).resolve().parent


class FrozenDino(nn.Module):
    def __init__(self):
        super().__init__()
        os.environ['XFORMERS_DISABLED']='1'
        from gcanet.reconstruction.vendor.dinov2.hub.backbones import dinov2_vitl14
        self.backbone=dinov2_vitl14(pretrained=False)
        from gcanet.reconstruction.core.config import cfg
        from gcanet.reconstruction.runtime import sha256, load_checkpoint
        path = Path(cfg.dino_checkpoint or os.environ.get('GCANET_DINO_CHECKPOINT', ''))
        if not path.is_file():
            raise FileNotFoundError('Provide DINOv2 ViT-L/14 weights with initialize(..., dino_checkpoint=...)')
        provenance = {'weight_path': str(path.resolve()), 'weight_sha256': sha256(path)}
        self.backbone.load_state_dict(load_checkpoint(provenance['weight_path']),strict=True)
        self.backbone.requires_grad_(False);self.backbone.eval()
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).reshape(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).reshape(1,3,1,1))
        self.provenance=provenance

    def train(self,mode=True):
        super().train(mode);self.backbone.eval();return self

    def forward(self,image):
        # Same RGB crop/augmentation as H4W, no feature-cache/crop mismatch.
        # 448/14=32 patches on each axis; full 24-layer ViT-L is retained.
        with torch.no_grad(),torch.autocast(device_type=image.device.type, dtype=torch.float16, enabled=image.is_cuda):
            rgb=F.interpolate(image[:,:3],size=(448,448),mode='bilinear',align_corners=False,antialias=True)
            tokens=self.backbone.forward_features((rgb-self.mean)/self.std)['x_norm_patchtokens']
        tokens=tokens.float()
        if not torch.isfinite(tokens).all():raise FloatingPointError('Nonfinite frozen DINO features')
        return tokens


class GeometryQueries(nn.Module):
    """Stable canonical identities, two local and two scene-wide attention layers."""
    def __init__(self):
        super().__init__()
        self.input=nn.Sequential(nn.Linear(9,256),nn.LayerNorm(256),nn.GELU())
        self.identity=nn.Parameter(torch.randn(568,256)*.02)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(256,8,1024,.1,batch_first=True,norm_first=True)
                                   for _ in range(4)])
        self.norm=nn.LayerNorm(256)

    def neighborhood(self,canonical):
        with torch.no_grad():
            entity=torch.arange(568,device=canonical.device)>=504
            distance=torch.cdist(canonical,canonical).masked_fill(entity[None,:,None]!=entity[None,None,:],float('inf'))
            index=distance.topk(16,largest=False).indices
            mask=torch.ones_like(distance,dtype=torch.bool).scatter_(-1,index,False)
            return mask[:,None].expand(-1,8,-1,-1).reshape(-1,568,568)

    def forward(self,points,canonical,entity,mask):
        # Use the common human root, preserving human-object relative placement.
        xyz=points-points[:,:1]
        query=self.input(torch.cat((xyz,canonical,entity.expand(len(points),-1,-1)),-1))+self.identity[None]
        for i,layer in enumerate(self.layers):query=layer(query,src_mask=mask if i<2 else None)
        return self.norm(query)


class GeometryConditionedSemanticFusion(nn.Module):
    def __init__(self,mode="dino_single"):
        super().__init__();self.mode=mode;self.trace=[];self.last_diagnostics={}
        if mode != 'dino_single': raise ValueError('GCANet uses one GSF pass')
        from gcanet.reconstruction.models.templates import smplh
        self.dino=FrozenDino();self.geometry=GeometryQueries()
        self.image=nn.Linear(1024,256);self.context=nn.Linear(2048,256)
        self.query=nn.Linear(256,256);self.key=nn.Linear(256,256)
        self.null=nn.Linear(256,8);self.visibility=nn.Linear(256,1)
        self.value=nn.Linear(256,2048);self.geometry_value=nn.Linear(256,2048)
        self.mix=nn.Parameter(torch.tensor(-2.))
        self.register_buffer('entity',F.one_hot(torch.tensor([0]*73+[1]*431+[2]*64),3).float()[None])
        self.register_buffer('human_faces',torch.tensor(smplh.faces.astype(np.int64)))
        y,x=torch.meshgrid(torch.arange(32),torch.arange(32),indexing='ij')
        self.register_buffer('uv',torch.stack(((x+.5)/32-.5,(y+.5)/32-.5),-1).reshape(-1,2))

    def prepare(self,model,canonical,object_ids,full_human,image=None):
        assert image is not None
        self.trace=[];self.canonical=canonical;self.object_ids=object_ids;self.current_human=full_human
        self.memory=self.image(self.dino(image));self.keys=self.key(self.memory).reshape(len(image),1024,8,32)
        self.local_mask=self.geometry.neighborhood(canonical)
        self.last_diagnostics=dict(dino_tokens=1024,dino_width=1024,dino_precision='frozen autocast fp16' if image.is_cuda else 'frozen fp32 CPU',
                                   dino_feature_std=float(self.memory.detach().std()))

    def forward(self,model,points,canonical,deep,camera,round_index):
        b=len(points)
        query=self.geometry(points,canonical,self.entity,self.local_mask)+self.context(deep.mean((-1,-2)))[:,None]
        q=self.query(query).reshape(b,568,8,32)
        score=torch.einsum('bnhd,bphd->bhnp',q,self.keys)/(32.**.5)
        null=self.null(query).permute(0,2,1)[...,None]
        probabilities=torch.cat((score,null),-1).softmax(-1)
        weights=probabilities[...,:-1].mean(1);null_mass=probabilities[...,-1:].mean(1)
        visible=self.visibility(query).sigmoid()
        visible=torch.cat((torch.ones_like(visible[:,:73]),visible[:,73:]),1)
        assert round_index==0
        projected=None;observed=weights @ self.memory
        token=self.geometry_value(query)+self.mix.sigmoid()*visible*self.value(observed)
        self.trace.append(dict(score=weights.clamp_min(1e-8).log(),null=null_mass.clamp_min(1e-8).log(),
                               weights=weights,visible=visible,points=points,projected=projected,
                               ownership=None,geometry_points=points))
        self.last_diagnostics.update(round=round_index,projected_evidence_sampling=False,
             visible_support_mean=float(visible.detach().mean()),no_evidence_mass=float(null_mass.detach().mean()),
             assignment_mass=float(weights.detach().sum(-1).mean()),mix=float(self.mix.detach().sigmoid()),
             attention_entropy=float(-(probabilities.detach()*probabilities.detach().clamp_min(1e-8).log()).sum(-1).mean()),
             geometry_feature_norm=float(query.detach().norm(dim=-1).mean()))
        return torch.cat((token,points),-1)

    def auxiliary(self,target,meta):
        return common.evidence_auxiliary(self,target,meta)
