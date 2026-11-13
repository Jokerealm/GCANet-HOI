"""Exact resume checkpoints with frozen, hash-pinned DINO weights stored externally."""
import torch

PREFIX = 'evidence.dino.backbone.'


def compact_state(model):
    state = model.state_dict()
    excluded = [key for key in state if key.startswith(PREFIX)]
    reference = None
    if excluded:
        assert all(not p.requires_grad for p in model.evidence.dino.backbone.parameters())
        reference = model.evidence.dino.provenance['weight_sha256']
    return {k: v for k, v in state.items() if k not in excluded}, excluded, reference


def load_model(model, checkpoint):
    current = model.state_dict()
    for key, value in checkpoint['model'].items():
        if key not in current or not torch.is_tensor(value) or value.shape != current[key].shape or value.dtype != current[key].dtype:
            raise ValueError('Reconstruction checkpoint tensor mismatch: ' + key)
    if checkpoint.get('format') not in ('gcanet_gsf_v1', 'official_full_compact_v1'):
        if set(checkpoint['model']) != set(current):
            raise ValueError('Incomplete reconstruction checkpoint')
        return model.load_state_dict(checkpoint['model'], strict=True)
    excluded = set(checkpoint['external_frozen_keys'])
    expected = {key for key in model.state_dict() if key.startswith(PREFIX)}
    if excluded != expected:
        raise ValueError('External frozen key set differs from the constructed model')
    if excluded and checkpoint['external_dino_sha256'] != model.evidence.dino.provenance['weight_sha256']:
        raise ValueError('External frozen DINO weight hash differs')
    if set(checkpoint['model']) != set(current) - excluded:
        raise ValueError('Compact reconstruction state must cover all nonexternal tensors')
    result = model.load_state_dict(checkpoint['model'], strict=False)
    if set(result.missing_keys) != excluded or result.unexpected_keys:
        raise ValueError('Unexpected missing/unexpected checkpoint keys: ' + str(result))
    return result


def payload(model, epoch, config, **extra):
    state, excluded, reference = compact_state(model)
    return dict(format='gcanet_gsf_v1', model=state, epoch=epoch, config=config,
                external_frozen_keys=excluded, external_dino_sha256=reference, **extra)
