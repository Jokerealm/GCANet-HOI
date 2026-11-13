import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from gcanet.vendor.pihoc.models.segment_anything import build_sam_vit_h

from gcanet.vendor.pihoc.canonical_constants import HUMAN_VIEW_DICT

class ContactProjector(torch.nn.Module):
    def __init__(self, multiview_channels=4, eps=1e-8):
        super().__init__()
        self.multiview_channels = multiview_channels
        
        metadata_root = './data'
        hC_sam_view_type = '4MV-Z_Vitru'
        metadata_folder = HUMAN_VIEW_DICT[hC_sam_view_type]['folder']
        pixel_to_vertex_dict = np.load(os.path.join(metadata_root, metadata_folder, HUMAN_VIEW_DICT[hC_sam_view_type]['pixel_to_vertex']))
        bary_coord_dict = np.load(os.path.join(metadata_root, metadata_folder, HUMAN_VIEW_DICT[hC_sam_view_type]['bary_coords']))

        self.views = HUMAN_VIEW_DICT[hC_sam_view_type]['names'].flatten()
        self.num_vertices = HUMAN_VIEW_DICT[hC_sam_view_type]['num_vertices']
        
        pixel_to_vertex_list, bary_coord_list = [], []
        for view in self.views:
            pixel_to_vertex_list.append(pixel_to_vertex_dict[view])
            bary_coord_list.append(bary_coord_dict[view])
        
        self.pixel_to_vertex_map = torch.from_numpy(np.array(pixel_to_vertex_list))
        self.bary_coord_map = torch.from_numpy(np.array(bary_coord_list))
        
    @torch.no_grad()
    def _valid_rows(self, vertices_flat):
        """rows where all three vertices are valid indices"""
        valid_mask = (vertices_flat >= 0) & (vertices_flat < self.num_vertices)
        return valid_mask.all(dim=1)

    def forward(self, seg_maps, threshold_for_inference=None):
        """
        seg_maps: list of length B; each item is a tensor [V, H, W] where V=self.multiview_channels
                  values are logits (default) or probabilities in [0,1] if input_is_logits=False
        """
        device = seg_maps[0].device
        dtype  = seg_maps[0].dtype
        B = len(seg_maps)

        pred_3d_contacts = torch.zeros((B, self.num_vertices), device=device, dtype=dtype)
        view_count = torch.zeros((B, self.num_vertices), device=device, dtype=dtype)

        for b, seg_map in enumerate(seg_maps):
            
            if self.multiview_channels == 1:
                raise NotImplementedError("Grid format deprecated, use multiview_channels > 1")

            # Loop over views; seg_map[v] is [H, W]
            for v in range(self.multiview_channels):
                view_seg_map = seg_map[v]  # [H, W]
                p = torch.sigmoid(view_seg_map)  # [H, W] in [0,1]

                # Flatten per-view tensors
                p_flat = p.reshape(-1)  # [N], N = H*W

                p2v_map   = self.pixel_to_vertex_map[v].to(device)        # [H, W, 3] (int)
                bary_map  = self.bary_coord_map[v].to(device, dtype=dtype) # [H, W, 3] (float)

                vertices  = p2v_map.reshape(-1, 3)   # [N, 3]
                weights   = bary_map.reshape(-1, 3)  # [N, 3]

                # Keep only rows with valid vertex triplets (this does NOT depend on seg_map => keeps grad)
                valid_rows = self._valid_rows(vertices)
                if not torch.any(valid_rows):
                    continue

                vertices  = vertices[valid_rows]             # [M, 3]
                weights   = weights[valid_rows]              # [M, 3]
                p_sel     = p_flat[valid_rows]               # [M]

                # Scatter-add contributions to vertices: for each triangle corner i in {0,1,2}
                for i in range(3):
                    idx = vertices[:, i].long()              # [M]
                    src = p_sel * weights[:, i]              # [M]

                    pred_3d_contacts[b].scatter_add_(0, idx, src)
                    view_count[b].scatter_add_(0, idx, weights[:, i])

        # Normalize by accumulated weights; keep zeros where no support
        # the vertices where view_count = 0 the pred_3d_contact is also 0. So division by 1e-8 doesnot make any difference.
        denom = view_count.clamp_min(1e-8)
        pred_3d_contacts = pred_3d_contacts / denom  # [B, N_verts], in [0,1] when inputs are probs/sigmoid(logits)
        
        if threshold_for_inference is not None:
            return (pred_3d_contacts > threshold_for_inference).to(dtype)
        
        return pred_3d_contacts

class VIv1CamPoseEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=512, output_dim=1024, num_views=4):
        super().__init__()

        # Spatial understanding
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # View-specific transformations
        self.view_transforms = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for _ in range(num_views)
        ])
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, cam_params, view_idx):
        base_encoding = self.spatial_encoder(cam_params)
        view_specific = self.view_transforms[view_idx](base_encoding)
        view_specific = self.sigmoid(view_specific)
        return view_specific

class ModifiedSAM(nn.Module):
    def __init__(self, original_sam):
        super().__init__()
        self.image_encoder = original_sam.image_encoder
        self.prompt_encoder = original_sam.prompt_encoder
        self.mask_decoder = original_sam.mask_decoder
        self.postprocess_masks = original_sam.postprocess_masks
    
    def forward(self, image_embeddings, sparse_prompt_embeddings, dense_prompt_embeddings):
        
        fused_embeddings = image_embeddings
        mask_decoder = self.mask_decoder

        masks, iou_pred = mask_decoder(
            image_embeddings=fused_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=False,
        )
        return masks, iou_pred
    
class iVLMDecoder(nn.Module):
    def __init__(self, args): # proj_layer_in_dim=1024, proj_layer_out_dim=256):
        super().__init__()
        
        self.args = args
        self.multiview_channels = 4
        
        ############# Build SAM #############
        
        original_sam = build_sam_vit_h(self.args.sam_pretrained)
        self.visual_model = ModifiedSAM(original_sam)
        
        self.image_embeddings = None
        
        # Freeze parameters of the visual model
        for param in self.visual_model.parameters():
            param.requires_grad = False

        # If training the mask decoder, unfreeze its parameters
        if self.args.train_sam_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True
                
        ############# `DINOV2` to SAM Projection Layers #############
        # Projection layer
        in_dim = args.dino_visual_width_vit
        out_dim = args.dino2sam_proj_layer_out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim//2, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True
        
        ############# Camera Encoder #############
        self.cam_pose_encoder = VIv1CamPoseEncoder(input_dim=5, hidden_dim=self.args.dino2sam_proj_layer_out_dim//2, output_dim=self.args.dino2sam_proj_layer_out_dim, num_views=self.args.num_sam_input_views)
        self.cam_pose_encoder.train()
        for param in self.cam_pose_encoder.parameters():
            param.requires_grad = True
            
        print("iVLM decoder initialised")
        
        
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.visual_model.image_encoder(pixel_values[i])
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.stack(image_embeddings_list, dim=0)

        return image_embeddings
    
    def process_embeddings(self, embedding, cam_params):

        # Make the llava embeding view aware
        view_encodings = []
        for view_idx in range(self.args.num_sam_input_views):
            view_enc = self.cam_pose_encoder(cam_params[[view_idx]], view_idx)
            view_encodings.append(view_enc)
        view_encodings = torch.stack(view_encodings, dim=1)  # [B, V, D]
        
        # Combine interaction feature embeddings with view-specific information
        embedding = embedding * view_encodings

        return embedding
    
    def forward(self, interaction_tokens, targets):
        # just compute this once as sam images are the same throughout all examples
        if self.image_embeddings is None:
            # sam_images: [B, MultiView, 3, H, W]
            sam_images = torch.stack([target['sam_images'] for target in targets], dim=0)
            self.image_embeddings = self.get_visual_embs(sam_images[0:1])
        
        image_embeddings = self.image_embeddings.repeat((interaction_tokens.shape[0], 1, 1, 1, 1)) # [num_ho_tokens, 4, 256, 64, 64]
        
        cam_params = torch.stack([target['sam_cam_params'] for target in targets], dim=0)
        cam_params = cam_params[0:1].repeat((interaction_tokens.shape[0], 1, 1)) # [num_ho_tokens, 4, 5]
        
        pred_embeddings = self.text_hidden_fcs[0](interaction_tokens.unsqueeze(1)) # [num_ho_tokens, 1, 512] -> [num_ho_tokens, 1, 256]
        
        pred_masks, gt_masks = [], []
        for i in range(len(pred_embeddings)):
            pred_emb = pred_embeddings[i].unsqueeze(1) # [1, 1, 256]
            pred_emb = pred_emb.repeat(1, self.multiview_channels, 1) # [1, 4, 256]
            processed_embedding = self.process_embeddings(pred_emb, cam_params[i]) # [1, 4, 256]
            
            sparse_embeddings, dense_embeddings = self.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=processed_embedding,
            )
            sparse_embeddings = sparse_embeddings.to(processed_embedding.dtype) # [1, 4, 256]
            
            low_res_masks, iou_predictions = self.visual_model(
                image_embeddings=image_embeddings[i],
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
            )
            
            pred_mask = self.visual_model.postprocess_masks(
                low_res_masks,
                input_size=(1024, 1024),
                original_size=(1024, 1024),
            )
            
            pred_masks.append(pred_mask[:, 0])
        
        return pred_masks