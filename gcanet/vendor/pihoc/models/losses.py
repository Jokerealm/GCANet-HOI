import torch
import torch.distributed as dist
from torchvision.ops.boxes import box_iou

from gcanet.vendor.pihoc.utils.ops import binary_focal_loss_with_logits

from gcanet.vendor.pihoc.detr.util import box_ops

from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

IGNORE_LABEL = -1

class seg_loss_function(nn.Module):
    def __init__(self):
        super(seg_loss_function, self).__init__()
        self.ce = nn.BCELoss()

    def forward(self, y_true, y_pred):
        loss = self.ce(y_pred, y_true)
        return loss
    
class interacton_loss_function(nn.Module):
    def __init__(self):
        super(interacton_loss_function, self).__init__()
        self.ce_with_logits = nn.BCEWithLogitsLoss()

    def forward(self, y_true, y_pred):
        loss = self.ce_with_logits(y_pred, y_true)
        return loss

# class InteractionLoss:
#     def __init__(self,
#                  num_classes: int,
#                  for_bp: bool,
#                  alpha: float = 0.5,
#                  gamma: float = 2.0,
#                  fg_iou_thresh: float = 0.5,
#                  eps: float = 1e-8):
        
#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.gamma = gamma
#         self.fg_iou_thresh = fg_iou_thresh
#         self.eps = eps
        
#         self.for_bp = for_bp

#     def recover_boxes(self, boxes, size):
#         boxes = box_ops.box_cxcywh_to_xyxy(boxes)
#         h, w = size
#         scale_fct = torch.stack([w, h, w, h])
#         boxes = boxes * scale_fct
#         return boxes

#     def associate_interaction_with_ground_truth(self, boxes_h, boxes_o, targets):  # for training
#         n = boxes_h.shape[0]
#         # NOTE: keep behavior but ensure shape matches later advanced indexing
#         # labels = num_ho_pairs x num_classes
#         labels = torch.zeros((n, self.num_classes), device=boxes_h.device)

#         # if meta_info['dataset'] == 'damon_hico':
#         gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
#         gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])

#         iou_h = box_iou(boxes_h, gt_bx_h)
#         iou_o = box_iou(boxes_o, gt_bx_o)
#         min_iou = torch.min(iou_h, iou_o)
#         idx = torch.nonzero(min_iou >= self.fg_iou_thresh, as_tuple=False)
        
#         filtered_x, filtered_y = [], []
#         if idx.numel() != 0:  
#             x, y = idx.unbind(1) 
#             sorted_idxs = torch.argsort(min_iou[x,y], descending=True)
#             x = x[sorted_idxs]
#             y = y[sorted_idxs]
            
#             for x_, y_ in zip(x, y): 
#                 if x_ not in filtered_x:
#                     filtered_x.append(int(x_))
#                     filtered_y.append(int(y_))
                    
#             filtered_x = torch.tensor(filtered_x, device=labels.device)
#             filtered_y = torch.tensor(filtered_y, device=labels.device)
            
#             labels[filtered_x] = targets['partwise_contact_gt'].float()
    
#         # x, y = torch.nonzero(torch.min(
#         #     box_iou(boxes_h, gt_bx_h),
#         #     box_iou(boxes_o, gt_bx_o)
#         # ) >= self.fg_iou_thresh).unbind(1)

#         # if self.num_classes in (117, 24, 407):
#         #     labels[x, targets['labels'][y]] = 1  # targets['labels']: verb/action
#         # elif self.num_classes == 15:
#         #     labels[x] = targets['partwise_contact_gt'].float()
#         # else:
#         #     labels[x, targets['hoi'][y]] = 1
                
#         return labels

#     def compute_interaction_loss(self, boxes, bh, bo, logits, prior, targets):
#         # bx, bo: indices of boxes
#         labels = []
#         for bx, h, o, target in zip(boxes, bh, bo, targets):
#             label = self.associate_interaction_with_ground_truth(bx[h], bx[o], target)
#             labels.append(label)
#         labels = torch.cat(labels)

#         # we filter vcoco interactions in prior calculations. when training is for damon_hico
#         prior = torch.cat(prior, dim=1).prod(0)
#         x, y = torch.nonzero(prior).unbind(1)  # x is pair number, y is verb
        
#         logits = torch.cat(logits)
#         logits = logits[x, y]
#         prior = prior[x, y]
#         labels = labels[x, y]

#         n_p = len(torch.nonzero(labels))
#         if dist.is_initialized():
#             world_size = dist.get_world_size()
#             n_p = torch.as_tensor([n_p], device=logits.device if logits.is_cuda else 'cpu')
#             dist.barrier()
#             dist.all_reduce(n_p)
#             n_p = (n_p / world_size).item()

#         # remember this prior formula as scaling logits. if prior is high then logits will increase in value and 
#         # if prior is low then logits will go to a smaller value.
#         loss = binary_focal_loss_with_logits(
#         torch.log(
#             prior / (1 + torch.exp(-logits) - prior) + 1e-8
#         ), labels, reduction='sum',
#         alpha=self.alpha, gamma=self.gamma
#         )

#         return loss / max(n_p, 1e-6)  # guard just in case n_p == 0
    

    
class ContactLoss:
    def __init__(self, 
                 contact2d_bce_loss_weight, contact2d_bce_loss_alpha, contact2d_focal_gamma,
                 contact2d_dice_loss_weight, contact2d_dice_loss_scale, 
                 contact3d_bce_loss_weight, contact3d_focal_alpha, contact3d_focal_gamma, contact3d_sparsity_weight,
                 body_part_contact_bce_loss_weight,
                 ):
        
        self.contact2d_bce_loss_weight = contact2d_bce_loss_weight
        self.contact2d_bce_loss_alpha = contact2d_bce_loss_alpha
        self.contact2d_focal_gamma = contact2d_focal_gamma
        
        self.contact2d_dice_loss_weight = contact2d_dice_loss_weight
        self.contact2d_dice_loss_scale = contact2d_dice_loss_scale
        
        self.contact3d_bce_loss_weight = contact3d_bce_loss_weight
        self.contact3d_focal_alpha = contact3d_focal_alpha
        self.contact3d_focal_gamma = contact3d_focal_gamma
        self.contact3d_sparsity_weight = contact3d_sparsity_weight
        
        self.body_part_contact_bce_loss_weight = body_part_contact_bce_loss_weight
        
    def recover_boxes(self, boxes, size):
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h])
        boxes = boxes * scale_fct
        return boxes
        
    ##################################### FOCAL LOSS #############################################
    def compute_contact2d_focal_loss(self, inputs, targets):

        def focal_loss_fn(input_view, target_view):
            valid_mask = (target_view != IGNORE_LABEL)
            input_view = input_view[valid_mask]
            target_view = target_view[valid_mask]

            # return 0 loss if there are no valid mask regions
            if input_view.numel() == 0:
                return torch.tensor(0.0, device=input_view.device, dtype=input_view.dtype)
            
            pos_alpha = self.contact2d_bce_loss_alpha 
            neg_alpha = 1 - pos_alpha
        
            bce_loss = F.binary_cross_entropy_with_logits(input_view, target_view, reduction='none')
            # If the model predicts correctly then bce_loss is small, hence pt ≈ 1
            # If the model predicts wrongly then bce_loss is large, hence pt ≈ 0
            pt = torch.exp(-bce_loss)
            
            alpha_t = torch.where(target_view == 1, pos_alpha, neg_alpha)
            # (1-pt) is large weight on wrong predictions. gamma = 2 means more weight on wrong predictions 
            focal_loss = alpha_t * (1 - pt) ** self.contact2d_focal_gamma * bce_loss
            return focal_loss.mean()

        # Process each view separately for multi-view data
        losses = []
        for v in range(inputs.shape[0]):
            input_view = inputs[v].view(-1)
            target_view = targets[v].view(-1)
            view_loss = focal_loss_fn(input_view, target_view)
            losses.append(view_loss)
        
        return torch.stack(losses).mean()
    
    ##################################### DICE LOSS #############################################
    
    def compute_contact2d_dice_loss(self, inputs, targets, eps=1e-5):
        inputs = inputs.sigmoid()
        scale = self.contact2d_dice_loss_scale
        
        losses = []
        num_valid_views = 0
        for v in range(inputs.shape[0]):
            input_view = inputs[v].view(-1)
            target_view = targets[v].view(-1)
            
            valid_mask = (target_view != IGNORE_LABEL)
            input_view = input_view[valid_mask]
            target_view = target_view[valid_mask]

            # if target_view.sum() == 0 the numerator in dice loss is zero so 0 is returned directly
            if target_view.numel() == 0 or target_view.sum() == 0:
                view_loss = torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)
            else:
                num_valid_views += 1
                numerator = 2 * (input_view / scale * target_view).sum()
                denominator = (input_view / scale).sum() + (target_view / scale).sum()
                view_loss = 1 - ((numerator + eps) / (denominator + eps))
                
                if torch.isnan(view_loss):
                    num_valid_views -= 1
                    print(f"\nNaN detected in dice loss:")
                    print(f"Input range: [{input_view.min():.6f}, {input_view.max():.6f}]")
                    print(f"Target range: [{target_view.min():.6f}, {target_view.max():.6f}]")
                    print(f"Numerator: {numerator:.6f}, Denominator: {denominator:.6f}")
                    # Return zero loss for this iteration to continue training
                    view_loss = torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)

            losses.append(view_loss)
        
        return torch.stack(losses).sum() / max(num_valid_views, 1)
    
    def get_contact2d_loss(self, pred_2d, gt_2d):
        gt_2d = gt_2d.to(pred_2d.dtype)
        
        loss_list_bce, loss_list_dice = [], []
        for pred_mask, gt_mask in zip(pred_2d, gt_2d):
            loss_bce = self.compute_contact2d_focal_loss(pred_mask, gt_mask).mean()
            if loss_bce != 0.0:
                loss_list_bce.append(loss_bce)
            
            loss_dice = self.compute_contact2d_dice_loss(pred_mask, gt_mask).mean()
            if loss_dice != 0.0:
                loss_list_dice.append(loss_dice)
        
        # Apply weights and average over batch
        if len(loss_list_bce) > 0:
            contact2d_bce_loss = self.contact2d_bce_loss_weight * sum(loss_list_bce)/max(len(loss_list_bce), 1)
        else:
            contact2d_bce_loss = torch.tensor(0.0, device=pred_2d.device, dtype=pred_2d.dtype)
        
        if len(loss_list_dice) > 0:
            contact2d_dice_loss = self.contact2d_dice_loss_weight * sum(loss_list_dice)/max(len(loss_list_dice), 1)
        else:
            contact2d_dice_loss = torch.tensor(0.0, device=pred_2d.device, dtype=pred_2d.dtype)
        
        return contact2d_bce_loss, contact2d_dice_loss
    
    def get_contact3d_loss(self, pred_3d, gt_3d):
        
        pos_alpha = self.contact3d_focal_alpha 
        neg_alpha = 1 - pos_alpha

        ############ BCE / FOCAL Loss ############
        gt_3d = gt_3d.to(pred_3d.dtype)
        bce_loss = F.binary_cross_entropy(pred_3d, gt_3d, reduction='none')
        
        # pt = ~1 for good prediction and ~0 for wrong prediction
        pt = torch.exp(-bce_loss)
        
        alpha_t = torch.where(gt_3d == 1, pos_alpha, neg_alpha)
        # alpha_t = torch.where(gt_3d == 1, pos_alpha, pos_alpha)
        focal_loss = alpha_t * (1-pt)**self.contact3d_focal_gamma * bce_loss
        
        ############ SPARSITY LOSS ###############
        sparsity_loss = pred_3d.mean()
        
        total_loss = focal_loss.mean() + self.contact3d_sparsity_weight * sparsity_loss
        
        return total_loss
    
    def compute_contact_loss(self, contact_results, contact_results_3d, gt, gt_3d):
        
        ############ 3D Loss ###############
        pred_3d = contact_results_3d
        gt_3d = gt_3d
        contact3d_loss = self.contact3d_bce_loss_weight * self.get_contact3d_loss(pred_3d, gt_3d)
        
        ############ 2D Loss ###############
        pred_2d = torch.stack(contact_results)
        gt_2d = gt
        contact2d_bce_loss, contact2d_dice_loss = self.get_contact2d_loss(pred_2d, gt_2d)
        # mask_bce_loss = self.get_contact_mask_loss_weighted_bce(pred_masks, gt_masks)
        
        ###################################
        
        loss_dict = {}
        loss_dict['contact_3d_loss'] = contact3d_loss
        loss_dict['contact2d_bce_loss'] = contact2d_bce_loss
        loss_dict['mask_dice_loss'] = contact2d_dice_loss
        
        return loss_dict

    def compute_mlp_contact_loss(self, contact_results_3d, gt_3d):
        
        ############ 3D Loss ###############
        pred_3d = contact_results_3d
        gt_3d = gt_3d
        contact3d_loss = self.contact3d_bce_loss_weight * self.get_contact3d_loss(pred_3d, gt_3d)
        
        ###################################
        
        loss_dict = {}
        loss_dict['contact_3d_loss'] = contact3d_loss
        
        return loss_dict
    
    def compute_body_partwise_contact_loss(self, body_part_logits, partwise_gt, partwise_gt_validity):
        
        body_part_logits.squeeze(-1)
        loss_per_part = F.binary_cross_entropy_with_logits(body_part_logits.squeeze(-1), partwise_gt.float(), reduction="none")  # shape (5, 15)
        
        valid_mask = partwise_gt_validity.unsqueeze(1)
        loss_masked = loss_per_part * valid_mask
        
        body_part_bce_loss = loss_masked.sum() / valid_mask.sum().clamp(min=1)
        
        loss_dict = {}
        loss_dict['body_part_contact_loss'] = self.body_part_contact_bce_loss_weight * body_part_bce_loss
        return loss_dict
    
    
    ############ IF you want to use it later: WEIGHTED BCE LOSS #################
    
    def compute_bce_loss_weighted(self, inputs, targets, pos_weight=10.0):
        
        def bce_loss_fn(input_view, target_view, pos_weight):
            valid_mask = (target_view != IGNORE_LABEL)
            input_view = input_view[valid_mask]
            target_view = target_view[valid_mask]

            if input_view.numel() == 0:
                return torch.tensor(0.0, device=input_view.device, dtype=input_view.dtype)
            
            weights = torch.where(target_view == 1, pos_weight, 1.0)
            
            bce_loss = F.binary_cross_entropy_with_logits(input_view, target_view, reduction='none')
            
            weighted_loss = weights * bce_loss
            
            return weighted_loss.mean()

        # Process each view separately for multi-view data
        losses = []
        for v in range(inputs.shape[0]):
            input_view = inputs[v].view(-1)
            target_view = targets[v].view(-1)
            view_loss = bce_loss_fn(input_view, target_view, pos_weight)
            losses.append(view_loss)
        
        return torch.stack(losses).mean()
        
    
    def get_contact_mask_loss_weighted_bce(self, pred_contact_masks, gt_contact_masks):
        mask_bce_loss = torch.tensor(0.0, device=pred_contact_masks.device, dtype=pred_contact_masks.dtype)
        gt_contact_masks = gt_contact_masks.to(pred_contact_masks.dtype)
        
        loss_list_bce = []
        for pred_mask, gt_mask in zip(pred_contact_masks, gt_contact_masks):
            loss_bce = self.compute_bce_loss_weighted(pred_mask, gt_mask).mean()
            if loss_bce != 0.0:
                loss_list_bce.append(loss_bce)
                
        mask_bce_loss = self.bce_loss_weight * sum(loss_list_bce)/max(len(loss_list_bce), 1)
        return mask_bce_loss
    
    ################################################
        