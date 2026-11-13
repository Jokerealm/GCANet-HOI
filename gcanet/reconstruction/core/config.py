import os
import torch
import shutil
import datetime
import numpy as np
import os.path as osp
from easydict import EasyDict as edict



cfg = edict()


""" Directory """
cfg.cur_dir = osp.dirname(osp.abspath(__file__))
cfg.root_dir = osp.abspath(osp.join(cfg.cur_dir, '../../..'))
cfg.data_dir = os.environ.get('GCANET_DATA_ROOT', osp.join(cfg.root_dir, 'data'))
cfg.dino_checkpoint = os.environ.get('GCANET_DINO_CHECKPOINT', '')
cfg.hand4whole_checkpoint = os.environ.get('GCANET_HAND4WHOLE_CHECKPOINT', '')

KST = datetime.timezone(datetime.timedelta(hours=9)) # CHANGE TIMEZONE FROM HERE


""" Dataset """
cfg.DATASET = edict()
cfg.DATASET.name = 'BEHAVE'
cfg.DATASET.workers = 4
cfg.DATASET.random_seed = 123
cfg.DATASET.bbox_expand_ratio = 1.3
cfg.DATASET.obj_set = 'behave'


""" Model - HMR """
cfg.MODEL = edict()
cfg.MODEL.input_img_shape = (512, 512)
cfg.MODEL.input_body_shape = (256, 256)
cfg.MODEL.input_hand_shape = (256, 256)
cfg.MODEL.img_feat_shape = (8, 8, 8)
cfg.MODEL.weight_path = ''


""" Train Detail """
cfg.TRAIN = edict()
cfg.TRAIN.batch_size = 16
cfg.TRAIN.shuffle = True
cfg.TRAIN.begin_epoch = 1
cfg.TRAIN.end_epoch = 50
cfg.TRAIN.warmup_epoch = 3
cfg.TRAIN.scheduler = 'step'
cfg.TRAIN.lr = 1.0e-4
cfg.TRAIN.min_lr = 1e-6
cfg.TRAIN.lr_step = [30]
cfg.TRAIN.lr_factor = 0.1
cfg.TRAIN.optimizer = 'adam'
cfg.TRAIN.momentum = 0
cfg.TRAIN.weight_decay = 0
cfg.TRAIN.beta1 = 0.5
cfg.TRAIN.beta2 = 0.999
cfg.TRAIN.print_freq = 10

cfg.TRAIN.loss_names = ['contact', 'vert', 'edge', 'param', 'coord', 'hand_bbox']
cfg.TRAIN.contact_loss_weight = 1.0
cfg.TRAIN.smpl_vert_loss_weight = 1.0
cfg.TRAIN.obj_vert_loss_weight = 1.0
cfg.TRAIN.smpl_edge_loss_weight = 1.0
cfg.TRAIN.smpl_pose_loss_weight = 1.0
cfg.TRAIN.smpl_shape_loss_weight = 1.0
cfg.TRAIN.obj_pose_loss_weight = 1.0
cfg.TRAIN.obj_trans_loss_weight = 1.0
cfg.TRAIN.smpl_3dkp_loss_weight = 1.0
cfg.TRAIN.smpl_2dkp_loss_weight = 1.0
cfg.TRAIN.pos_2dkp_loss_weight = 1.0
cfg.TRAIN.hand_bbox_loss_weight = 1.0


""" Augmentation """
cfg.AUG = edict()
cfg.AUG.scale_factor = 0.2
cfg.AUG.rot_factor = 30
cfg.AUG.shift_factor = 0
cfg.AUG.color_factor = 0.2
cfg.AUG.blur_factor = 0
cfg.AUG.flip = False


""" Test Detail """
cfg.TEST = edict()
cfg.TEST.batch_size = 32
cfg.TEST.shuffle = False
cfg.TEST.do_eval = True
cfg.TEST.eval_metrics = ['contact_rec_p', 'contact_rec_r','cd_human', 'cd_object']
cfg.TEST.print_freq = 10
cfg.TEST.contact_thres = 0.05


""" CAMERA """
cfg.CAMERA = edict()
cfg.CAMERA.focal = (1000, 1000)
cfg.CAMERA.princpt = (cfg.MODEL.input_img_shape[1]/2, cfg.MODEL.input_img_shape[0]/2)
cfg.CAMERA.depth_factor = 4.4
cfg.CAMERA.obj_depth_factor = 2.2*2

