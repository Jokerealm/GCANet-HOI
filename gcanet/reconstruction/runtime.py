"""Portable BEHAVE data, initialization, and native reconstruction loss API.

Call initialize before importing template-dependent modules. All assets are
resolved from data_root; importing this module never changes the working directory.
"""
import hashlib
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch

_INITIALIZED_ROOT = None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path):
    # These are user-supplied, trusted training checkpoints, including RNG state.
    import inspect
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(path, **kwargs)


def initialize(data_root=None, dino_checkpoint=None, hand4whole_checkpoint=None):
    global _INITIALIZED_ROOT
    from .core.config import cfg
    value = data_root or os.environ.get('GCANET_DATA_ROOT')
    if not value:
        raise ValueError('Specify data_root or GCANET_DATA_ROOT')
    root = Path(value).expanduser().resolve()
    if not (root / 'base_data').is_dir():
        raise FileNotFoundError(f'Expected base_data under data_root: {root}')
    if _INITIALIZED_ROOT is not None and root != _INITIALIZED_ROOT:
        raise RuntimeError('Use a new process to change loaded template assets')
    if 'gcanet.reconstruction.models.templates' in sys.modules and Path(cfg.data_dir).resolve() != root:
        raise RuntimeError('initialize must precede imports of template-dependent modules')
    cfg.data_dir = str(root)
    cfg.DATASET.name, cfg.DATASET.obj_set = 'BEHAVE', 'behave'
    if dino_checkpoint is not None:
        cfg.dino_checkpoint = str(Path(dino_checkpoint).expanduser().resolve())
    if hand4whole_checkpoint is not None:
        cfg.hand4whole_checkpoint = str(Path(hand4whole_checkpoint).expanduser().resolve())
    _INITIALIZED_ROOT = root
    return cfg


def build_model(device='cuda', initialize_backbone=True):
    if _INITIALIZED_ROOT is None:
        initialize()
    from .models.model import GCANetReconstruction
    return GCANetReconstruction(initialize_backbone=initialize_backbone).to(device)


def dataset(split, training=False):
    if _INITIALIZED_ROOT is None:
        initialize()
    if split not in ('train', 'test'):
        raise ValueError('BEHAVE split must be train or test')
    if training and split != 'train':
        raise ValueError('Training augmentation may only use the training split')
    from torchvision.transforms import ToTensor
    from .dataset.behave import BEHAVE
    data = BEHAVE(ToTensor(), split)
    expected = 45380 if split == 'train' else 4129
    if len(data) != expected:
        raise ValueError(f'Full BEHAVE {split} requires {expected} examples, found {len(data)}')
    data.data_split = 'train' if training else 'evaluation'
    data.return_training_targets = bool(training)
    return data


def worker_seed(worker_id):
    import cv2
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    np.random.seed(torch.initial_seed() % 2**32)
    random.seed(torch.initial_seed())


def loader(data, batch, workers, training=False, generator=None):
    return torch.utils.data.DataLoader(data, batch_size=batch, shuffle=training,
        num_workers=workers, pin_memory=True, drop_last=False, persistent_workers=False,
        worker_init_fn=worker_seed, generator=generator)


def move(values, device='cuda'):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k,v in values.items()}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class OriginalLoss:
    def __init__(self):
        from gcanet.reconstruction.core.loss import get_loss
        self.loss = get_loss()

    def __call__(self, output, target, meta):
        from gcanet.reconstruction.utils.funcs_utils import rotmat_to_6d
        loss = self.loss
        parts = {}
        parts['vert'] = loss['vert'](output['smpl_verts6000'], target['smpl_verts'])
        parts['vert'] += loss['vert'](output['obj_verts'], target['obj_verts'])
        parts['vert'] += loss['vert'](output['smpl_verts1730'], target['smpl_verts1730'])
        parts['vert'] += loss['vert'](output['smpl_verts431'], target['smpl_verts431'])
        parts['edge'] = loss['edge'](output['smpl_verts6000'], target['smpl_verts'])
        parts['param'] = loss['param'](output['initial_smpl_pose'], rotmat_to_6d(target['smpl_pose']), meta['has_smpl_param'])
        parts['param'] += loss['param'](output['initial_smpl_shape'], target['smpl_shape'], meta['has_smpl_param'])
        parts['param'] += loss['param'](output['initial_obj_pose'], rotmat_to_6d(target['obj_pose']).reshape(len(target['obj_pose']), 6), meta['has_obj_param'])
        parts['param'] += loss['param'](output['initial_obj_trans'], target['obj_trans'], meta['has_obj_param'])
        parts['coord'] = loss['coord'](output['h3d_keypoints'], target['h3d_keypoints'])
        parts['coord'] += loss['coord'](output['h2d_keypoints'], target['h2d_keypoints'][..., :2])
        parts['coord'] += loss['coord'](output['hp2d_keypoints'], target['hp2d_keypoints'][..., :3])
        parts['coord'] += loss['coord'](output['refh3d_keypoints'], target['h3d_keypoints'])
        parts['coord'] += loss['coord'](output['refh2d_keypoints'], target['h2d_keypoints'][..., :2])
        parts['hand_bbox'] = loss['hand_bbox'](output['hand_bbox'], target['hand_bbox'])
        return sum(parts.values()), parts

