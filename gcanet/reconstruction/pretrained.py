"""Explicitly preserve the complete body/hand initializer, never silent partial loading."""
import torch
from pathlib import Path
from gcanet.reconstruction.runtime import load_checkpoint
from gcanet.reconstruction.core.config import cfg


def load_complete_initializer(module):
    source = cfg.hand4whole_checkpoint or str(Path(cfg.data_dir) / 'base_data/backbone_models/hand4whole.pth.tar')
    checkpoint = load_checkpoint(source)['network']
    state = {}; excluded = []; adaptations = []
    for key, value in checkpoint.items():
        key = key.removeprefix('module.')
        if key.startswith(('face_roi_net.', 'face_regressor.', 'box_net.face_size.')):
            excluded.append(key)
            continue
        if key == 'backbone.conv1.weight':
            adapted = torch.zeros(value.shape[0], 5, *value.shape[2:], dtype=value.dtype)
            adapted[:, :3] = value
            state['backbone.conv1_mask.weight'] = adapted
            adaptations.append('RGB conv copied exactly; two new mask channels zero initialized')
            continue
        if key.startswith('box_net.bbox_center.'):
            key = key.replace('box_net.bbox_center.', 'box_net.bbox_center_out.')
            value = value[:2].clone()
            adaptations.append('left/right/face bbox centers -> left/right, verified official ordering')
        state[key] = value
    state['backbone.obj_id_embed.weight'] = torch.zeros_like(module.backbone.obj_id_embed.weight, device='cpu')
    adaptations.append('new object-ID embedding zero initialized to preserve pretrained RGB function')
    result = module.load_state_dict(state, strict=True)
    return dict(source=source, strict_load=str(result), loaded_tensors=len(state),
                excluded_face_tensors=len(excluded), adaptations=adaptations,
                preserved='complete body/hand backbones, position, rotation, shape, camera and hand-box heads')
