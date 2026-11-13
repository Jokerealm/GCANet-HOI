"""Single-pass DINO-query evidence and unchanged training correspondence targets."""
import types
import numpy as np
import torch
import torch.nn.functional as F
from gcanet.reconstruction.utils.funcs_utils import rot6d_to_aa,rot6d_to_rotmat

NF=1000./256.

def projection(xyz, camera):
    camera_xyz = xyz + camera[:, None]
    return camera_xyz[..., :2] / camera_xyz[..., 2:].clamp_min(.05) * NF


def raster_depth(vertices, faces, camera, size=32, chunk=512):
    """Perspective-correct full triangle z-buffer; no hard mask in feature retrieval."""
    with torch.no_grad():
        xyz = vertices.detach() + camera.detach()[:,None]
        uv = xyz[...,:2] / xyz[...,2:].clamp_min(.05) * NF
        y,x=torch.meshgrid(torch.arange(size,device=xyz.device),torch.arange(size,device=xyz.device),indexing='ij')
        grid=torch.stack(((x+.5)/size-.5,(y+.5)/size-.5),-1).reshape(1,1,-1,2)
        depth=torch.full((len(xyz),size*size),float('inf'),device=xyz.device)
        for offset in range(0,len(faces),chunk):
            idx=faces[offset:offset+chunk]
            tri=uv[:,idx];z=xyz[:,idx,2]
            a,b,c=tri[:,:,0,None],tri[:,:,1,None],tri[:,:,2,None]
            den=(b[...,1]-c[...,1])*(a[...,0]-c[...,0])+(c[...,0]-b[...,0])*(a[...,1]-c[...,1])
            valid=den.abs()>1e-12;den=torch.where(valid,den,torch.ones_like(den))
            wa=((b[...,1]-c[...,1])*(grid[...,0]-c[...,0])+(c[...,0]-b[...,0])*(grid[...,1]-c[...,1]))/den
            wb=((c[...,1]-a[...,1])*(grid[...,0]-c[...,0])+(a[...,0]-c[...,0])*(grid[...,1]-c[...,1]))/den
            wc=1-wa-wb
            inside=(wa>=0)&(wb>=0)&(wc>=0)&valid&(z.min(-1).values[:,:,None]>.05)
            inverse=wa/z[:,:,0,None].clamp_min(.05)+wb/z[:,:,1,None].clamp_min(.05)+wc/z[:,:,2,None].clamp_min(.05)
            candidate=(1/inverse.clamp_min(1e-8)).masked_fill(~inside,float('inf'))
            depth=torch.minimum(depth,candidate.min(1).values)
        return depth.reshape(len(xyz),1,size,size)


def evidence_auxiliary(self, target, meta):
    from gcanet.reconstruction.models.templates import obj_dict
    with torch.no_grad():
        xyz = target['h3d_keypoints'][..., :3]
        uv = target['h2d_keypoints'][..., :2]
        zero = torch.zeros_like(uv[..., 0])
        f = torch.full_like(zero, NF)
        ax = torch.stack((f, zero, -uv[..., 0]), -1)
        ay = torch.stack((zero, f, -uv[..., 1]), -1)
        rhsx = uv[..., 0] * xyz[..., 2] - NF * xyz[..., 0]
        rhsy = uv[..., 1] * xyz[..., 2] - NF * xyz[..., 1]
        matrix = torch.cat((ax, ay), 1)
        rhs = torch.cat((rhsx, rhsy), 1)
        camera = torch.linalg.lstsq(matrix, rhs[..., None]).solution[..., 0]
        fit_error = (projection(xyz, camera) - uv).norm(dim=-1).mean(-1)
        gt = torch.cat((xyz, target['smpl_verts431'], target['obj_verts'][..., :3]), 1)
        target_uv = projection(gt, camera)
        valid = (target_uv.abs() < 0.49).all(-1) & (gt[..., 2] + camera[:, None, 2] > 0.05) & (fit_error[:, None] < 0.03)
        index = ((target_uv + 0.5) * 32).long().clamp(0, 31)
        label = index[..., 1] * 32 + index[..., 0]
        visibility = torch.ones_like(valid)
        depths = raster_depth(target['smpl_verts'], self.human_faces, camera)
        if not hasattr(self, '_surface_tensor_cache'):
            self._surface_tensor_cache = {}
        for (i, oid) in enumerate(self.object_ids):
            cache_key = (oid, gt.device, gt.dtype)
            if cache_key not in self._surface_tensor_cache:
                mesh = obj_dict[oid]
                self._surface_tensor_cache[cache_key] = (
                    torch.tensor(np.asarray(mesh.vertices), device=gt.device, dtype=gt.dtype),
                    torch.tensor(np.asarray(mesh.faces), device=gt.device, dtype=torch.long))
            vertices, faces = self._surface_tensor_cache[cache_key]
            vertices = vertices @ target['obj_pose'][i].T + target['obj_trans'][i]
            depths[i:i + 1] = torch.minimum(depths[i:i + 1], raster_depth(vertices[None], faces, camera[i:i + 1]))
        bi = torch.arange(len(gt), device=gt.device)[:, None]
        nearest = depths[:, 0][bi, index[..., 1], index[..., 0]]
        visibility = (gt[..., 2] + camera[:, None, 2] <= nearest + 0.05) & torch.isfinite(nearest)
        visibility[:, :73] = True
    total = gt.new_zeros(())
    parts = {}
    for trace in self.trace:
        logprob = torch.cat((trace['score'], trace['null']), -1).log_softmax(-1)
        good = valid & visibility
        ce = -(logprob.gather(-1, label[..., None]).squeeze(-1) * good).sum() / good.sum().clamp_min(1)
        total = total + 0.02 * ce / len(self.trace)
        hidden = valid & ~visibility
        total = total + 0.02 * (-logprob[..., -1] * hidden).sum() / hidden.sum().clamp_min(1) / len(self.trace)
        bce = F.binary_cross_entropy(trace['visible'][:, 73:, 0], visibility[:, 73:].float(), reduction='none')
        total = total + 0.02 * (bce * valid[:, 73:]).sum() / valid[:, 73:].sum().clamp_min(1) / len(self.trace)
    self.last_diagnostics.update(target_camera_fit_error=float(fit_error.mean()), target_valid_fraction=float(valid.float().mean()), target_visible_surface_fraction=float(visibility[:, 73:].float().mean()))
    with torch.no_grad():
        last = self.trace[-1]
        top = last['weights'].topk(8, -1).indices
        self.last_diagnostics['visible_target_in_top8'] = float(((top == label[..., None]).any(-1) & valid & visibility).sum() / (valid & visibility).sum().clamp_min(1))
    parts['evidence_supervision'] = total
    return (total, parts)

def gsf_forward(model,inputs):
    from gcanet.reconstruction.models.templates import smplh, obj_dict
    b=len(inputs['img'])
    deep,boxes,joint_img,pose,shape,camera=model.hand4whole(inputs['img'],inputs['obj_id'])
    object_pose,object_translation=model.objregressor(deep)
    aa=rot6d_to_aa(pose.reshape(-1,6)).reshape(b,-1)
    body=model.smplh_layer(betas=shape,global_orient=aa[:,:3],body_pose=aa[:,3:66],left_hand_pose=aa[:,66:111],right_hand_pose=aa[:,111:])
    initial_human=body.vertices-body.joints[:,[0]];joints=body.joints-body.joints[:,[0]]
    human=smplh.downsample(initial_human)
    ids=inputs['obj_id'].reshape(-1).tolist()
    canonical_object=torch.stack([torch.tensor(obj_dict[i].load_verts(),device=pose.device,dtype=pose.dtype) for i in ids])
    rotation=rot6d_to_rotmat(object_pose).reshape(b,3,3)
    initial_object=canonical_object @ rotation.transpose(1,2)+object_translation[:,None]
    points=torch.cat((joints.detach(),human.detach(),initial_object.detach()),1)
    template=smplh.layer['neutral'].v_template.to(pose.device)[None].expand(b,-1,-1)
    canonical_human=smplh.downsample(template)
    canonical_joints=torch.tensor(smplh.joint_regressor,device=pose.device,dtype=pose.dtype)[None] @ template
    canonical=torch.cat((canonical_joints,canonical_human,canonical_object),1)
    model.evidence.prepare(model,canonical,ids,initial_human,inputs['img'])
    tokens=model.evidence(model,points,canonical,deep,camera,0)
    refined_joints,refined_human,refined_object=model.tgformer(tokens,inputs['objgraph'].to(points.device))
    v1730,full_human=smplh.getupsample(refined_human)
    model.evidence.current_human=full_human
    model.evidence.first_scene=(full_human,refined_object)
    assert len(model.evidence.trace)==1
    model.evidence.last_diagnostics.update(tg_passes=1,evidence_reads=1,
        evidence_feedback=False,geometry_updated_before_requery=False)
    obj_r,obj_t=model.kps_to_Rt(refined_object,ids)
    return dict(initial_smpl_pose=pose,initial_smpl_shape=shape,initial_obj_pose=object_pose,initial_obj_trans=object_translation,
                initial_smpl_mesh=initial_human,initial_obj_verts=initial_object,
                smpl_verts6000=full_human,smpl_verts1730=v1730,smpl_verts431=refined_human,
                obj_verts=refined_object,obj_pose=obj_r,obj_trans=obj_t,
                h3d_keypoints=joints,h2d_keypoints=model.projection(joints,camera),hp2d_keypoints=joint_img,
                refh3d_keypoints=refined_joints,refh2d_keypoints=model.projection(refined_joints,camera),hand_bbox=boxes,cam_trans=camera)


def attach_gsf(model):
    from gcanet.gsf import GeometryConditionedSemanticFusion
    model.hand4whole.collect_pyramid = True
    # Preserve the checkpoint key namespace used to obtain the paper results.
    model.evidence = GeometryConditionedSemanticFusion().to(next(model.parameters()).device)
    model.evidence_auxiliary_loss = model.evidence.auxiliary
    model.evidence_diagnostics = lambda: model.evidence.last_diagnostics
