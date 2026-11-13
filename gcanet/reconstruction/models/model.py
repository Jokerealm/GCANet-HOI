import torch
import torch.nn as nn
import copy
import numpy as np
from gcanet.reconstruction.core.config import cfg
from gcanet.reconstruction.models.module import Hand4Whole, ObjectRegressor
from gcanet.reconstruction.models.transformer import TGformer
from gcanet.reconstruction.models.templates import smplh, obj_dict
from gcanet.reconstruction.utils.funcs_utils import rot6d_to_aa, rot6d_to_rotmat, sample_joint_features, rigid_transform_3D
import torch.nn.functional as F

class GCANetReconstruction(nn.Module):
    def __init__(self, initialize_backbone=True):
        super().__init__()
        self.hand4whole = Hand4Whole()
        self.objregressor = ObjectRegressor()
        self.tgformer = TGformer()
        self.smplh_layer = copy.deepcopy(smplh.layer['neutral'])
        self.init_weights(initialize_backbone)
        from gcanet.gsf_supervision import attach_gsf
        attach_gsf(self)
        self.trainable_modules = [self.hand4whole, self.objregressor,self.tgformer]

    def init_weights(self, initialize_backbone=True):
        from gcanet.reconstruction.pretrained import load_complete_initializer
        self.pretrain_report = load_complete_initializer(self.hand4whole) if initialize_backbone else {'source': 'full reconstruction checkpoint'}
        self.objregressor.init_weights()
        self.tgformer.init_weights()

    def forward(self, inputs):
        from gcanet.gsf_supervision import gsf_forward
        return gsf_forward(self, inputs)

    def projection(self, coords, cam_trans):
        x = (coords[:,:,0] + cam_trans[:,None,0]) / (coords[:,:,2] + cam_trans[:,None,2] + 1e-4) * cfg.CAMERA.focal[0] / cfg.MODEL.input_body_shape[0]
        y = (coords[:,:,1] + cam_trans[:,None,1]) / (coords[:,:,2] + cam_trans[:,None,2] + 1e-4) * cfg.CAMERA.focal[1] / cfg.MODEL.input_body_shape[1]
        return torch.stack((x,y),2)

    def kps_to_Rt(self, pred_kps, obj_ids):
        template_kps = []
        for i in obj_ids:
            template_kps.append(torch.tensor(obj_dict[i].load_verts()))
        template_kps = torch.stack(template_kps).to(pred_kps.device)

        rot_list, trans_list = [], []
        for i in range(len(obj_ids)):
            scale, rot ,trans = rigid_transform_3D(template_kps[i].detach().cpu().numpy(), pred_kps[i].detach().cpu().numpy())
            rot_list.append(rot); trans_list.append(trans)
        rot_list = torch.tensor(rot_list, device=pred_kps.device)
        trans_list = torch.tensor(trans_list, device=pred_kps.device)

        return rot_list, trans_list


def get_model():
    return GCANetReconstruction()
