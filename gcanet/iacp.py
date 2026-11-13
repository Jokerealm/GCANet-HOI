"""Instance-aware Contact Prior (IACP), GCANet Sec. 3.3.

Pair encoding and canonical-view decoding derive from Pi-HOC (see vendor licenses).

The predictor never accepts contact labels. Instances are associated only with
input masks and frozen DETR boxes. Canonical maps carry explicit provenance.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

VIEWS = ('topfront', 'bottomfront', 'topback', 'bottomback')
CAMERA = ((2,45,315,0,0),(2,315,315,0,.3),(2,45,135,0,0),(2,315,135,0,.3))


def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''): h.update(block)
    return h.hexdigest()


def mask_boxes(masks):
    """Axis-aligned boxes in normalized image coordinates; empty mask is invalid."""
    if masks.ndim == 4: masks=masks[:,0]
    if masks.ndim != 3: raise ValueError('Masks must have shape B,H,W')
    boxes=[]; valid=[]
    for mask in masks:
        y,x=torch.where(mask>.5)
        ok=x.numel()>0; valid.append(ok)
        if ok:
            h,w=mask.shape
            boxes.append(torch.stack((x.min()/w,y.min()/h,(x.max()+1)/w,(y.max()+1)/h)))
        else: boxes.append(mask.new_zeros(4))
    return torch.stack(boxes),torch.tensor(valid,device=masks.device,dtype=torch.bool)


def pair_attention_bias(patch_count, device, dtype=torch.float32):
    """One pair + CLS + patches; exact released no-cross-pair attention semantics."""
    n=patch_count+2
    mask=torch.zeros((n,n),device=device,dtype=torch.bool)
    mask[:2,:2]=~torch.eye(2,device=device,dtype=torch.bool)
    mask[1:,:1]=True
    return torch.zeros((1,1,n,n),device=device,dtype=dtype).masked_fill(mask,torch.finfo(dtype).min)


class QueryMLP(nn.Module):
    def __init__(self):
        super().__init__(); self.layers=nn.ModuleList([nn.Linear(512,256),nn.Linear(256,1024)])
    def forward(self,x): return self.layers[1](F.relu(self.layers[0](x)))


class PresenceClassifier(nn.Module):
    def __init__(self):
        super().__init__(); self.mlp=nn.Sequential(nn.Linear(1024,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,1))
    def forward(self,x): return self.mlp(x)


class _CanonicalWeightedAverage(torch.autograd.Function):
    """Fixed sparse linear projection without CUDA floating-point atomics.

    Each output vertex has one fixed ordered reduction. Its adjoint visits the
    three corners of each pixel exactly once, rather than scattering repeated
    contributions. The canonical maps and all weights are constant assets.
    """
    @staticmethod
    def forward(ctx,prob,vertices,weights,denominator,reduction_pixels,
                reduction_weights,reduction_vertices,buckets):
        result=prob.new_zeros((len(prob),len(denominator)))
        for offset,count,width,vertex_offset in buckets:
            size=count*width
            index=reduction_pixels[offset:offset+size].view(count,width)
            weight=reduction_weights[offset:offset+size].view(count,width)
            ids=reduction_vertices[vertex_offset:vertex_offset+count]
            value=(prob[:,index]*weight).sum(-1)/denominator[ids].clamp_min(1e-8)
            result.index_copy_(1,ids,value)
        ctx.save_for_backward(vertices,weights,denominator)
        return result

    @staticmethod
    def backward(ctx,gradient):
        vertices,weights,denominator=ctx.saved_tensors
        normalized=gradient/denominator.clamp_min(1e-8)
        pixel_gradient=(normalized[:,vertices]*weights).sum(-1)
        return pixel_gradient,None,None,None,None,None,None,None


class CanonicalProjector(nn.Module):
    """Original barycentric scatter of sigmoid mask logits to all 6890 vertices."""
    def __init__(self,root):
        super().__init__(); root=Path(root)
        with np.load(root/'pixel_to_vertex_map_1024.npz') as p:
            indices=torch.from_numpy(np.stack([p[v] for v in VIEWS])).long()
        with np.load(root/'bary_coords_map_1024.npz') as p:
            bary=torch.from_numpy(np.stack([p[v] for v in VIEWS])).float()
        if indices.shape!=(4,1024,1024,3) or bary.shape!=indices.shape:
            raise ValueError('Canonical maps must be four complete 1024 views')
        valid=((indices>=0)&(indices<6890)).all(-1)
        if not torch.isfinite(bary[valid]).all() or (bary[valid].sum(-1)-1).abs().max()>2e-5:
            raise ValueError('Invalid canonical barycentric maps')
        # Keep only fixed valid pixels; indices remain exact canonical triangle identities.
        pix=torch.where(valid.flatten())[0]
        self.register_buffer('pixels',pix,persistent=False)
        self.register_buffer('vertices',indices.reshape(-1,3)[pix],persistent=False)
        self.register_buffer('weights',bary.reshape(-1,3)[pix],persistent=False)
        denom=torch.zeros(6890)
        for c in range(3): denom.scatter_add_(0,self.vertices[:,c],self.weights[:,c])
        self.register_buffer('denominator',denom,persistent=False)
        self.register_buffer('support',denom>0,persistent=False)
        self._build_reduction()

    def _build_reduction(self):
        # Construction happens on CPU before module/device audits. Sorting once
        # avoids non-deterministic atomic scatter_add during every GPU forward.
        vertices=self.vertices.detach().cpu().numpy().reshape(-1)
        weights=self.weights.detach().cpu().numpy().reshape(-1)
        pixels=np.repeat(np.arange(len(self.vertices),dtype=np.int64),3)
        order=np.argsort(vertices,kind='stable')
        vertices=vertices[order];weights=weights[order];pixels=pixels[order]
        count=np.bincount(vertices,minlength=len(self.denominator))
        start=np.cumsum(count)-count
        all_pixels=[];all_weights=[];all_vertices=[];buckets=[]
        offset=vertex_offset=0;lower=0;width=1
        while lower<count.max():
            ids=np.flatnonzero((count>lower)&(count<=width))
            if len(ids):
                bp=np.zeros((len(ids),width),dtype=np.int64)
                bw=np.zeros((len(ids),width),dtype=np.float32)
                for row,v in enumerate(ids):
                    sl=slice(start[v],start[v]+count[v])
                    bp[row,:count[v]]=pixels[sl];bw[row,:count[v]]=weights[sl]
                all_pixels.append(bp.reshape(-1));all_weights.append(bw.reshape(-1));all_vertices.append(ids)
                buckets.append((offset,len(ids),width,vertex_offset))
                offset+=bp.size;vertex_offset+=len(ids)
            lower=width;width*=2
        self.register_buffer('reduction_pixels',torch.from_numpy(np.concatenate(all_pixels)),persistent=False)
        self.register_buffer('reduction_weights',torch.from_numpy(np.concatenate(all_weights)),persistent=False)
        self.register_buffer('reduction_vertices',torch.from_numpy(np.concatenate(all_vertices)),persistent=False)
        self.reduction_buckets=tuple(buckets)

    def forward(self,logits):
        prob=logits.float().sigmoid().flatten(1)[:,self.pixels]
        return _CanonicalWeightedAverage.apply(prob,self.vertices,self.weights,self.denominator,
            self.reduction_pixels,self.reduction_weights,self.reduction_vertices,self.reduction_buckets)

    def known_view_targets(self,labels,known):
        """Published contact rendering uses all three contact vertices per face.

        A triangle is labelled only if all three vertex labels are known. Unknown
        labels never become negatives; no upsampled sparse-label pseudo truth.
        """
        corners=labels[:,self.vertices]
        valid=known[:,self.vertices].all(-1)
        target=(corners>.5).all(-1).float()
        return target,valid


class InstanceAwareContactPrior(nn.Module):
    frozen_module_prefixes=('detector.',
        'contact_decoder.visual_model.image_encoder.',
        'contact_decoder.visual_model.prompt_encoder.')

    def __init__(self,assets,activation_checkpointing=True,detector_min_iou=.1,
                 speed_equivalent_optimizations=True,checkpoint_policy='all',annotation_vertex_indices=None):
        super().__init__()
        from gcanet.vendor.pihoc.detr.models.backbone import build_backbone
        from gcanet.vendor.pihoc.detr.models.transformer import build_transformer
        from gcanet.vendor.pihoc.detr.models.detr import DETR
        from gcanet.vendor.pihoc.dinov2.models.vision_transformer import vit_large
        from gcanet.vendor.pihoc.models.contact_decoders.ivlm_contact import iVLMDecoder
        if isinstance(assets, (str, Path)):
            manifest_path = Path(assets).expanduser().resolve()
            self.asset_manifest = json.loads(manifest_path.read_text())
            asset_base = manifest_path.parent
        elif isinstance(assets, dict):
            manifest_path = None
            self.asset_manifest = dict(assets)
            asset_base = Path.cwd()
        else:
            raise TypeError('assets must be a dictionary or JSON path')
        def asset_path(value):
            path = Path(value).expanduser()
            return path.resolve() if path.is_absolute() else (asset_base / path).resolve()
        checkpoint_path = asset_path(self.asset_manifest['pretrained_checkpoint'])
        expected_hash = self.asset_manifest.get('pretrained_sha256')
        checkpoint_hash = file_sha256(checkpoint_path)
        if expected_hash and checkpoint_hash != expected_hash:
            raise ValueError('Pi-HOC initialization checkpoint hash mismatch')
        canonical_root = asset_path(self.asset_manifest['canonical_root'])
        canonical_manifest = canonical_root / 'canonical_manifest.json'
        expected_hash = self.asset_manifest.get('canonical_manifest_sha256')
        if expected_hash and file_sha256(canonical_manifest) != expected_hash:
            raise ValueError('Canonical manifest hash mismatch')
        cm = json.loads(canonical_manifest.read_text())
        for rec in cm['files']:
            # Manifest paths are portable; older manifests may contain an absolute prefix.
            asset = canonical_root / Path(rec['path']).name
            if file_sha256(asset) != rec['sha256']:
                raise ValueError('Canonical asset hash mismatch: ' + str(asset))
        self.initialization = 'contact'
        self.activation_checkpointing=bool(activation_checkpointing)
        if checkpoint_policy not in ('all','none','alternating'):
            raise ValueError('checkpoint_policy must be all, none or alternating')
        self.checkpoint_policy=checkpoint_policy
        self.speed_equivalent_optimizations=bool(speed_equivalent_optimizations)
        self.detector_min_iou=float(detector_min_iou)
        args=SimpleNamespace(backbone='resnet50',lr_backbone=0.,masks=False,dilation=False,
            hidden_dim=256,position_embedding='sine',dropout=.1,nheads=8,dim_feedforward=2048,
            enc_layers=6,dec_layers=6,pre_norm=False,num_queries=100,aux_loss=True,
            input_image_resolution=336,sam_pretrained=None,train_sam_mask_decoder=True,
            dino2sam_proj_layer_out_dim=256,num_sam_input_views=4)
        self.detector=DETR(build_backbone(args),build_transformer(args),80,100,aux_loss=True)
        args,self.dino_model=vit_large(patch_size=14,img_size=518,init_values=1.,block_chunks=0,adapter=False,args=args)
        self.query_proj=QueryMLP()
        self.binary_contact_classifier=PresenceClassifier()
        self.contact_decoder=iVLMDecoder(args)
        raw=torch.load(checkpoint_path,map_location='cpu',weights_only=False,mmap=True)
        state=raw['model_state_dict']
        required=self.state_dict()
        selected={k:v for k,v in state.items() if k in required}
        missing=sorted(set(required)-set(selected))
        extra=sorted(set(state)-set(required))
        allowed=('semseg_decoder.','partseg_decoder.')
        if missing or any(not k.startswith(allowed) for k in extra):
            raise ValueError('Pi-HOC incompatible state: '+repr((missing,extra[:20])))
        self.load_state_dict(selected,strict=True)
        self.pretrained_audit=dict(checkpoint=str(checkpoint_path),sha256=checkpoint_hash,
            original_epoch=raw.get('epoch'),original_iteration=raw.get('iteration'),
            loaded_tensor_count=len(selected),omitted_training_only_segmentation_tensors=extra,
            canonical_origin=cm['origin'],canonical_byte_exact=cm['byte_exact_to_unreleased_original_assets'])
        del raw,state,selected,required
        self.pretrained_audit['initialization']='contact_pretrained_PiHOC'
        self.projector=CanonicalProjector(canonical_root)
        self.register_buffer('annotation_topology',torch.zeros(0,dtype=torch.bool),persistent=False)
        self.annotation_has_no_complete_faces=False
        if annotation_vertex_indices is not None:
            self.configure_annotation_topology(annotation_vertex_indices)
        from PIL import Image
        pixels=torch.stack([torch.from_numpy(np.asarray(Image.open(canonical_root/f'body_render_norm_{v}.png').convert('RGB')).copy()).permute(2,0,1) for v in VIEWS]).float()
        mean=torch.tensor([123.675,116.28,103.53]).view(1,3,1,1)
        std=torch.tensor([58.395,57.12,57.375]).view(1,3,1,1)
        self.register_buffer('canonical_images',(pixels-mean)/std,persistent=False)
        self.register_buffer('canonical_embeddings',torch.zeros(4,256,64,64),persistent=False)
        cameras=torch.tensor([[d/10,e/360,a/360,(x+1)/2,(y+1)/2] for d,e,a,x,y in CAMERA])
        self.register_buffer('canonical_cameras',cameras,persistent=False)
        self.register_buffer('image_mean',torch.tensor([.485,.456,.406]).view(1,3,1,1),persistent=False)
        self.register_buffer('image_std',torch.tensor([.229,.224,.225]).view(1,3,1,1),persistent=False)
        self.fixed_assets_ready=False
        self.provenance=dict(self.pretrained_audit, asset_manifest_path=str(manifest_path) if manifest_path else None,
            asset_manifest_sha256=file_sha256(manifest_path) if manifest_path else None,
            input_protocol='HOI-TG RGB crop in [0,1] to DETR; separate bicubic antialiased 336x336 ImageNet-normalized IF input',
            inference_precision='Pi_IF_and_SAM_decoder_autocast_BF16_on_CUDA; DETR_FP32; contact_projector_FP32')
        self.provenance.update(speed_equivalent_optimizations=self.speed_equivalent_optimizations,
                               interactionformer_checkpoint_policy=self.checkpoint_policy,
                               annotation_topology_has_no_complete_faces=self.annotation_has_no_complete_faces)
        for name,p in self.named_parameters(): p.requires_grad_(not name.startswith(self.frozen_module_prefixes))
        self.train(True)

    @torch.no_grad()
    def prepare_fixed_assets(self,offload_frozen_image_encoder=False):
        if self.fixed_assets_ready: return
        encoder=self.contact_decoder.visual_model.image_encoder
        encoder.eval()
        device=next(encoder.parameters()).device
        with torch.autocast(device.type,enabled=False):
            for index in range(4):
                self.canonical_embeddings[index].copy_(encoder(self.canonical_images[index:index+1].float())[0])
        self.fixed_assets_ready=True
        if offload_frozen_image_encoder:
            encoder.to('cpu')
            if device.type=='cuda': torch.cuda.empty_cache()

    def train(self,mode=True):
        super().train(mode)
        self.detector.eval()
        self.contact_decoder.visual_model.image_encoder.eval()
        self.contact_decoder.visual_model.prompt_encoder.eval()
        return self

    def trainable_groups(self):
        groups={'interactionformer':[],'decoder':[]}
        for name,p in self.named_parameters():
            if p.requires_grad: groups['interactionformer' if name.startswith('dino_model.') else 'decoder'].append(p)
        return groups

    def configure_annotation_topology(self,indices):
        """Bind a known sparse-label topology once, without inventing labels.

        The per-batch check below still requires its annotation mask to be a
        subset. Dense or other label topologies retain the original 2D loss.
        These nonpersistent tensors add no learned parameters or checkpoint keys.
        """
        vertices=self.projector.vertices.detach().cpu()
        indices=torch.as_tensor(indices,dtype=torch.long).detach().cpu().reshape(-1)
        count=len(self.projector.denominator)
        if not len(indices) or bool((indices<0).any()) or bool((indices>=count).any()):
            raise ValueError('Annotation topology contains invalid vertex identities')
        topology=torch.zeros(count,dtype=torch.bool)
        topology[indices]=True
        self.annotation_has_no_complete_faces=not bool(topology[vertices].all(-1).any())
        topology=topology.to(self.projector.vertices.device)
        if 'annotation_topology' in self._buffers:
            self.annotation_topology=topology
        else:
            self.register_buffer('annotation_topology',topology,persistent=False)

    @torch.no_grad()
    def _instance_features(self,rgb,human_mask,object_mask):
        from gcanet.vendor.pihoc.detr.util.misc import nested_tensor_from_tensor_list
        from gcanet.vendor.pihoc.detr.util.box_ops import box_cxcywh_to_xyxy
        from torchvision.ops import box_iou,batched_nms
        human_box,hvalid=mask_boxes(human_mask); object_box,ovalid=mask_boxes(object_mask)
        normalized=(rgb-self.image_mean)/self.image_std
        samples=nested_tensor_from_tensor_list(list(normalized.float()))
        features,pos=self.detector.backbone(samples)
        src,mask=features[-1].decompose()
        hs,_=self.detector.transformer(self.detector.input_proj(src),mask,self.detector.query_embed.weight,pos[-1])
        feats=hs[-1]; probs=self.detector.class_embed(hs)[-1].softmax(-1)
        scores,labels=probs[...,:-1].max(-1)
        boxes=box_cxcywh_to_xyxy(self.detector.bbox_embed(hs)[-1].sigmoid())
        pair=[]; coverage=[]; overlaps=[]
        for b in range(len(rgb)):
            keep=batched_nms(boxes[b],scores[b],labels[b],.5)
            # Released preparation keeps at least two of each type, even below .2.
            inds=[]
            for human in (True,False):
                cand=keep[(labels[b,keep]==0) if human else (labels[b,keep]!=0)]
                above=cand[scores[b,cand]>=.2]
                cand=cand[scores[b,cand].argsort(descending=True)]
                selected=cand[:2] if len(above)<2 else (cand[:15] if len(above)>15 else above)
                inds.append(selected)
            hi,oi=inds
            ok=bool(hvalid[b] and ovalid[b] and len(hi)>0 and len(oi)>0)
            if ok:
                hu=box_iou(human_box[b:b+1],boxes[b,hi])[0]
                ou=box_iou(object_box[b:b+1],boxes[b,oi])[0]
                ih,io=hu.argmax(),ou.argmax(); overlaps.append(torch.stack((hu[ih],ou[io])))
                ok=bool(hu[ih]>=self.detector_min_iou and ou[io]>=self.detector_min_iou)
                pair.append(torch.cat((feats[b,hi[ih]],feats[b,oi[io]])))
            else:
                pair.append(feats[b].new_zeros(512)); overlaps.append(feats[b].new_zeros(2))
            coverage.append(ok)
        return torch.stack(pair),torch.tensor(coverage,device=rgb.device,dtype=torch.bool),torch.stack(overlaps)

    def _encode_pair(self,image,tokens):
        # All tokens share image memory; the upstream attention mask makes each
        # requested pair independent of other pair tokens. Single-pair decoding
        # therefore retains the released interaction computation.
        x=self.dino_model.prepare_tokens_with_masks(image)
        x=torch.cat((tokens[:,None],x),dim=1)
        bias=pair_attention_bias(x.shape[1]-2,x.device,x.dtype)
        for index,block in enumerate(self.dino_model.blocks):
            policy=getattr(self,'checkpoint_policy','all')
            selected=policy=='all' or (policy=='alternating' and index%2==0)
            if self.activation_checkpointing and selected and self.training and torch.is_grad_enabled():
                x=checkpoint(block,x,bias,use_reentrant=False)
            else: x=block(x,bias)
        return self.dino_model.norm(x)[:,0]

    def _decode(self,feature):
        dec=self.contact_decoder
        token=dec.text_hidden_fcs[0](feature[None,None])
        token=token.repeat(1,4,1)
        token=dec.process_embeddings(token,self.canonical_cameras)
        sparse,dense=dec.visual_model.prompt_encoder(points=None,boxes=None,masks=None,text_embeds=token)
        sparse=sparse.to(token.dtype)
        low,_=dec.visual_model(self.canonical_embeddings.to(sparse.dtype),sparse,dense)
        full=dec.visual_model.postprocess_masks(low,input_size=(1024,1024),original_size=(1024,1024))
        return full[:,0]

    def _decode_batch(self,features):
        """Batch the released sample/view broadcast without changing prompts.

        Every canonical view receives the SAME complete four-camera-token set
        for its sample, exactly as the released B=1 broadcast. Flattening B*4
        here only changes execution batching; no sample shares another's tokens.
        All SAM output tokens and heads are retained, including unused heads.
        """
        dec=self.contact_decoder
        batch=len(features)
        token=dec.text_hidden_fcs[0](features[:,None]).expand(-1,4,-1)
        token=dec.process_embeddings(token,self.canonical_cameras)
        sparse,dense=dec.visual_model.prompt_encoder(points=None,boxes=None,masks=None,text_embeds=token)
        sparse=sparse.to(token.dtype)
        decoder=dec.visual_model.mask_decoder
        output_tokens=torch.cat((decoder.iou_token.weight,decoder.mask_tokens.weight),0)
        output_tokens=output_tokens[None].expand(batch,-1,-1)
        tokens=torch.cat((output_tokens,sparse),1).repeat_interleave(4,dim=0)
        images=self.canonical_embeddings.to(sparse.dtype)
        src=(images[None]+dense[:,None]).reshape(batch*4,*images.shape[1:])
        positional=dec.visual_model.prompt_encoder.get_dense_pe().expand(batch*4,-1,-1,-1)
        _,channels,height,width=src.shape
        hs,src=decoder.transformer(src,positional,tokens)
        mask_tokens=hs[:,1:1+decoder.num_mask_tokens]
        upscaled=decoder.output_upscaling(src.transpose(1,2).reshape(batch*4,channels,height,width))
        hyper=torch.stack([head(mask_tokens[:,i])
                           for i,head in enumerate(decoder.output_hypernetworks_mlps)],1)
        masks=(hyper@upscaled.flatten(2)).reshape(batch*4,decoder.num_mask_tokens,*upscaled.shape[-2:])
        # Keep the same computation/parameter exposure as the released decoder.
        # The IoU output is intentionally not a contact objective in either path.
        decoder.iou_prediction_head(hs[:,0])
        full=dec.visual_model.postprocess_masks(masks[:,:1],input_size=(1024,1024),original_size=(1024,1024))
        return full[:,0].reshape(batch,4,*full.shape[-2:])

    def forward(self,rgb,human_mask,object_mask):
        if not self.fixed_assets_ready:
            raise RuntimeError('Call prepare_fixed_assets() after moving model to its device and before auditing or forward')
        if rgb.ndim!=4 or rgb.shape[1]!=3: raise ValueError('Expected RGB B,3,H,W')
        if not torch.isfinite(rgb).all() or rgb.min() < -1e-5 or rgb.max()>1+1e-5:
            raise ValueError('Pi-HOC RGB input must be [0,1]')
        # Detector is frozen and runs FP32 independently of mixed-precision IF.
        with torch.autocast(rgb.device.type,enabled=False):
            raw,coverage,iou=self._instance_features(rgb.float(),human_mask,object_mask)
        resized=F.interpolate(rgb,size=(336,336),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
        image=(resized-self.image_mean)/self.image_std
        with torch.autocast(rgb.device.type,dtype=torch.bfloat16,enabled=rgb.is_cuda):
            feature=self._encode_pair(image,self.query_proj(raw))
            presence_logits=self.binary_contact_classifier(feature).squeeze(-1)
            view_logits=(self._decode_batch(feature) if self.speed_equivalent_optimizations
                         else torch.stack([self._decode(f) for f in feature]))
        feature=feature.float();presence_logits=presence_logits.float();view_logits=view_logits.float()
        probability=self.projector(view_logits)
        return dict(probabilities=probability*coverage[:,None],raw_probabilities=probability,
            pair_features=feature*coverage[:,None],presence=presence_logits.sigmoid()*coverage,
            presence_logits=presence_logits,coverage=coverage,instance_iou=iou,
            view_logits=view_logits,canonical_vertex_support=self.projector.support)

    def contact_loss(self,output,labels,valid_mask=None):
        labels=labels.float()
        if labels.shape!=output['raw_probabilities'].shape: raise ValueError('Expected 6890 labels per pair')
        annotated=torch.ones_like(labels,dtype=torch.bool) if valid_mask is None else valid_mask.bool()
        known=annotated & output['coverage'][:,None] & self.projector.support[None]
        prob=output['raw_probabilities'].float().clamp(1e-6,1-1e-6)
        zero=prob.sum()*0
        bce=F.binary_cross_entropy(prob,labels,reduction='none')
        focal=.5*(1-(-bce).exp()).pow(3)*bce
        den=known.sum().clamp_min(1)
        # Sparsity only on known labels; unknown sparse labels are not negatives.
        loss3d=4*((focal*known).sum()/den+.01*(prob*known).sum()/den)
        skip_empty_2d=(getattr(self,'speed_equivalent_optimizations',False)
                       and getattr(self,'annotation_has_no_complete_faces',False)
                       and valid_mask is not None
                       and self.annotation_topology.numel()==annotated.shape[-1]
                       and not bool((annotated & ~self.annotation_topology[None]).any()))
        if skip_empty_2d:
            # Structural proof: no face has three potentially known vertices.
            # Original focal/Dice are exactly zero, with zero mask gradients.
            loss2d=dice=zero
            valid_pixels=known.new_zeros((),dtype=torch.long)
        else:
            target2d,valid2d=self.projector.known_view_targets(labels,known)
            pred2d=output['view_logits'].float().flatten(1)[:,self.projector.pixels]
            bce2d=F.binary_cross_entropy_with_logits(pred2d,target2d,reduction='none')
            focal2d=.5*(1-(-bce2d).exp()).pow(3)*bce2d
            loss2d=4*(focal2d*valid2d).sum()/valid2d.sum().clamp_min(1)
            pv=pred2d.sigmoid()*valid2d; tv=target2d*valid2d
            has_positive=tv.sum(-1)>0
            dice=1-(2*(pv*tv).sum(-1)+1e-5)/(pv.sum(-1)+tv.sum(-1)+1e-5)
            dice=(dice*has_positive).sum()/has_positive.sum().clamp_min(1)
            valid_pixels=valid2d.sum().detach()
        # A positive observed vertex proves presence; absence requires all labels.
        positive=((labels>.5)&annotated).any(-1)
        full_known=annotated.all(-1)
        presence_valid=(positive|full_known)&output['coverage']
        cp=F.binary_cross_entropy_with_logits(output['presence_logits'].float(),positive.float(),reduction='none')
        presence_loss=(cp*presence_valid).sum()/presence_valid.sum().clamp_min(1)
        return dict(contact_3d=loss3d,contact_2d=loss2d+zero,contact_dice=dice+zero,
            contact_presence=presence_loss,contact_valid_vertices=known.sum().detach(),
            contact_valid_pixels=valid_pixels,contact_valid_presence=presence_valid.sum().detach())


    def load_adapted_checkpoint(self, path):
        """Load a full IACP state or export from the recorded joint-adaptation format.

        Compact legacy checkpoints omit fixed pretrained tensors. They are accepted
        only after hashes establish that the restored fixed tensors are identical.
        """
        saved = torch.load(Path(path), map_location='cpu', weights_only=False, mmap=True)
        state = saved.get('iacp', saved.get('model', saved.get('state_dict', saved)))
        if not isinstance(state, dict):
            raise ValueError('Expected an IACP parameter dictionary')
        prefixes = ('pihoc_contact.', 'iacp.')
        prefix = next((p for p in prefixes if any(k.startswith(p) for k in state)), None)
        if prefix:
            state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        current = self.state_dict()
        if saved.get('format') in ('pihoc_joint_compact_v1', 'gcanet_iacp_v1'):
            if saved.get('format') == 'pihoc_joint_compact_v1' and prefix != 'pihoc_contact.':
                raise ValueError('Joint checkpoint has no IACP branch')
            fixed = {k[len(prefix):]: v for k, v in saved['external_frozen_hashes'].items()
                     if k.startswith(prefix)} if prefix else dict(saved['external_frozen_hashes'])
            expected_fixed = {k for k in current if k.startswith(self.frozen_module_prefixes)}
            if set(fixed) != expected_fixed or set(state) != set(current) - expected_fixed:
                raise ValueError('Compact IACP state does not exactly cover fixed and adapted tensors')
            for key, expected in fixed.items():
                value = current[key].detach().cpu().contiguous()
                # Same dtype/shape/layout checksum as the recorded training code.
                digest = hashlib.sha256()
                digest.update(str((tuple(value.shape), str(value.dtype), str(value.layout))).encode())
                if value.layout != torch.strided:
                    raise ValueError('Unexpected non-dense IACP state: ' + key)
                digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
                if digest.hexdigest() != expected:
                    raise ValueError('Fixed pretrained IACP tensor mismatch: ' + key)
            state = {**current, **state}
        if set(state) != set(current):
            raise ValueError('IACP checkpoint keys do not match the complete model')
        for key, value in state.items():
            if value.shape != current[key].shape or value.dtype != current[key].dtype:
                raise ValueError('IACP tensor shape or dtype mismatch: ' + key)
        self.load_state_dict(state, strict=True)
        return {'checkpoint': str(Path(path).resolve()), 'epoch': saved.get('epoch'),
                'sha256': file_sha256(path), 'loaded_tensors': len(state)}
