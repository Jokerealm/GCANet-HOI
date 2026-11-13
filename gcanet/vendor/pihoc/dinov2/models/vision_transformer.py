# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from gcanet.vendor.pihoc.dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block


import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

logger = logging.getLogger("dinov2")

def attn_mask_to_attn_bias(attn_mask: torch.Tensor, dtype=None):
    """
    attn_mask: bool tensor, True = masked/disallowed
    returns: additive bias tensor (float) with -inf at masked positions, 0 elsewhere
    """
    if dtype is None:
        dtype = torch.float32 if not attn_mask.is_floating_point() else attn_mask.dtype

    neg = torch.finfo(dtype).min  # safer than -inf for fp16/bf16 kernels
    attn_bias = torch.zeros_like(attn_mask, dtype=dtype)
    attn_bias = attn_bias.masked_fill(attn_mask, neg)
    return attn_bias

def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        channel_adaptive=False,
        adapter=False,
        args=None,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth # depth of transformer
        self.num_heads = num_heads # heads in MHSA
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.bag_of_channels = channel_adaptive

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = np.linspace(0, drop_path_rate, depth).tolist()  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)
        
        self.adapter = adapter
        self.args = args
        if adapter:
            adapters_list = [
                LAINadapter(args, args.adapt_dim,
                            init_option='lora',
                            adapter_scalar='learnable_scalar',
                            adapter_num_layers=args.adapter_num_layers,
                            ) for i in range(depth)
                ]
            self.lain_adapters = nn.ModuleList(adapters_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, la_priors, ho_tokens, attn_mask, la_masks, masks=None): # mask here is for dino training to mask image patches. attn_mask is different
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)
        x = torch.cat([ho_tokens.unsqueeze(0), x], dim=1)

        attn_bias = attn_mask_to_attn_bias(attn_mask)
        attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)
        if self.adapter:
            raise RuntimeError("Released Pi-HOC uses no_interactionadapter; adapter variants are not supported")
            for blk, lain_adapter in zip(self.blocks, self.lain_adapters):
                x = x.permute(1,0,2)
                adapt_x = lain_adapter(x, prior=la_priors,la_masks=la_masks)
                x = x + adapt_x
                x = x.permute(1,0,2)
                
                x = blk(x, attn_bias)
        else:
            for blk_num, blk in enumerate(self.blocks):
                x = blk(x, attn_bias)
                # viz = True
                # if viz and blk_num==23:
                #     inp_image = la_masks[3]
                #     num_pairs = ho_tokens.shape[0]
                #     for pair_num in range(0, num_pairs):
                #         h_id, o_id = int(la_masks[1][pair_num]), int(la_masks[2][pair_num])
                #         h_bbox = la_masks[0][h_id].int().detach().cpu().numpy()
                #         o_bbox = la_masks[0][o_id].int().detach().cpu().numpy()
                        
                #         last_attn = blk.attn.last_attn[0].mean(dim=0)[pair_num][-576:].reshape(24,24)
                #         img = np.asarray(inp_image, dtype=np.float32)
                #         img01 = (img - img.min()) / (img.max() - img.min() + 1e-8)
                #         img01 = np.clip(img01, 0.0, 1.0)
                #         H, W, _ = img01.shape
                #         attn = last_attn
                #         attn = attn.detach().cpu().float()
                #         attn_up = F.interpolate(
                #             attn[None, None, ...], size=(H, W), mode="bilinear", align_corners=False
                #         )[0, 0]

                #         # normalize to [0,1]
                #         attn_up = attn_up - attn_up.min()
                #         attn_up = attn_up / (attn_up.max() + 1e-8)
                #         attn_up = attn_up.numpy()
                #         fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
                #         ax.imshow(img01)
                #         ax.imshow(attn_up, cmap="jet", alpha=0.45)

                #         # draw bboxes: [x1,y1,x2,y2]
                #         hx1, hy1, hx2, hy2 = map(int, h_bbox)
                #         ox1, oy1, ox2, oy2 = map(int, o_bbox)

                #         ax.add_patch(Rectangle((hx1, hy1), hx2 - hx1, hy2 - hy1,
                #                             fill=False, linewidth=2, edgecolor="lime"))
                #         ax.add_patch(Rectangle((ox1, oy1), ox2 - ox1, oy2 - oy1,
                #                             fill=False, linewidth=2, edgecolor="red"))

                #         ax.axis("off")
                #         fig.tight_layout(pad=0)
                #         fig.savefig(f"out_{pair_num}.png", bbox_inches="tight", pad_inches=0)
                #         plt.close(fig)
                    

        x_norm = self.norm(x)
        return {
            "x_norm_global_feat": x_norm[:, : len(ho_tokens) + 1],
            "x_norm_local_feat": x_norm[:, len(ho_tokens) + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:

        if self.bag_of_channels:
            B, C, H, W = x.shape
            x = x.reshape(B * C, 1, H, W)  # passing channels to batch dimension to get encodings for each channel

        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if self.bag_of_channels:
            output = tuple(zip(outputs, class_tokens))
            output = list(
                zip(*output)
            )  # unzip the tuple: (list of patch_tokens per block, list of class tokens per block)
            patch_tokens_per_block = output[0]  # [BLOCK1, BLOCK2, ...] where BLOCK1.shape: B*C, N, D
            cls_tokens_per_block = output[1]  # [BLOCK1, BLOCK2, ...] where BLOCK1.shape: B*C, D
            patch_tokens_per_block = [
                patch_tokens.reshape(B, C, patch_tokens.shape[-2], patch_tokens.shape[-1])
                for patch_tokens in patch_tokens_per_block
            ]  # [BLOCK1, BLOCK2, ...] where BLOCK1.shape: B, C, N, D
            cls_tokens_per_block = [cls_tokens.reshape(B, -1) for cls_tokens in cls_tokens_per_block]
            output = tuple(zip(patch_tokens_per_block, cls_tokens_per_block))
            return output

        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, image, la_priors, ho_tokens, attn_mask, la_masks, is_training=False, **kwargs):
        ret = self.forward_features(image, la_priors, ho_tokens, attn_mask, la_masks, **kwargs)
        # print(ret.keys())
        return ret['x_norm_global_feat'], ret['x_norm_local_feat']
        # if is_training:
        #     return ret
        # else:
        #     return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, in_chans=3, channel_adaptive=False, adapter=False, args=None, **kwargs):
    args.dino_visual_width_vit = 384
    args.dino_visual_input_resolution_vit = 518
    args.dino_visual_patch_size_vit = patch_size
    args.dino_visual_num_patches = args.dino_visual_input_resolution_vit // args.dino_visual_patch_size_vit
    args.dino_visual_num_image_tokens = args.dino_visual_num_patches**2
    
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        in_chans=in_chans,
        channel_adaptive=channel_adaptive,
        adapter=adapter,
        args=args,
        **kwargs,
    )
    return args, model


def vit_base(patch_size=16, num_register_tokens=0, in_chans=3, channel_adaptive=False, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        in_chans=in_chans,
        channel_adaptive=channel_adaptive,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, in_chans=3, channel_adaptive=False, adapter=False, args=None, **kwargs):
    args.dino_visual_width_vit = 1024
    args.dino_visual_input_resolution_vit = 518
    args.dino_visual_patch_size_vit = patch_size
    args.dino_visual_num_patches = args.dino_visual_input_resolution_vit // args.dino_visual_patch_size_vit
    args.dino_visual_num_image_tokens = args.dino_visual_num_patches**2
    
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        in_chans=in_chans,
        channel_adaptive=channel_adaptive,
        adapter=adapter,
        args=args,
        **kwargs,
    )
    return args, model


def vit_giant2(patch_size=16, num_register_tokens=0, in_chans=3, channel_adaptive=False, adapter=False, args=None, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    args.dino_visual_width_vit = 1536
    args.dino_visual_input_resolution_vit = 518
    args.dino_visual_patch_size_vit = patch_size
    args.dino_visual_num_patches = args.dino_visual_input_resolution_vit // args.dino_visual_patch_size_vit
    args.dino_visual_num_image_tokens = args.dino_visual_num_patches**2
    
    model = DinoVisionTransformer( 
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        ffn_layer='swiglufused',
        num_register_tokens=num_register_tokens,
        in_chans=in_chans,
        channel_adaptive=channel_adaptive,
        adapter=adapter,
        args=args,
        **kwargs,
    )
    return args, model

def vit_custom(patch_size=16, num_register_tokens=0, in_chans=3, channel_adaptive=False, adapter=False, args=None, depth=6, **kwargs):
    args.dino_visual_width_vit = 1024
    args.dino_visual_input_resolution_vit = 518
    args.dino_visual_patch_size_vit = patch_size
    args.dino_visual_num_patches = args.dino_visual_input_resolution_vit // args.dino_visual_patch_size_vit
    args.dino_visual_num_image_tokens = args.dino_visual_num_patches**2
    
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=depth,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        in_chans=in_chans,
        channel_adaptive=channel_adaptive,
        adapter=adapter,
        args=args,
        **kwargs,
    )
    return args, model