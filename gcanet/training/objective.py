"""Fixed joint adaptation objective and optimizer groups."""
from contextlib import nullcontext
import torch

def optimizer_for(model, recipe):
    """Partition ALL trainable parameters exactly once, with distinct IACP rates."""
    group_ids = {}
    if hasattr(model, 'iacp'):
        contact = model.iacp
        if not hasattr(contact, 'trainable_groups'):
            raise ValueError('Complete IACP must expose trainable_groups()')
        declared = contact.trainable_groups()
        if set(declared) != {'interactionformer', 'decoder'}:
            raise ValueError('IACP group contract differs: ' + str(declared.keys()))
        for name, parameters in declared.items():
            for parameter in parameters:
                if not parameter.requires_grad:
                    raise ValueError('A declared trainable IACP parameter is frozen')
                if id(parameter) in group_ids:
                    raise ValueError('Duplicate IACP training group parameter')
                group_ids[id(parameter)] = 'iacp_' + name
    groups = {name: [] for name in ('native', 'iacp_interactionformer', 'iacp_decoder', 'new')}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('iacp.'):
            if id(parameter) not in group_ids:
                raise ValueError('Unclassified trainable IACP parameter: ' + name)
            category = group_ids[id(parameter)]
        elif name.startswith('adaptation_refiner.'):
            category = 'new'
        else:
            category = 'native'
        groups[category].append(parameter)
    rates = dict(native=recipe.get('old_lr', 1e-5),
                 iacp_interactionformer=recipe.get('iacp_if_lr', 5e-6),
                 iacp_decoder=recipe.get('iacp_decoder_lr', 1e-5),
                 new=recipe.get('new_lr', 1e-4))
    packed = [dict(name=name, params=parameters, lr=rates[name], initial_lr=rates[name])
              for name, parameters in groups.items() if parameters]
    if not packed or not groups['native']:
        raise ValueError('Original reconstruction must remain trainable')
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for group in packed for p in group['params']]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Optimizer groups do not exactly partition trainable model parameters')
    return torch.optim.Adam(packed, betas=tuple(recipe.get('adam_betas', [.5, .999])))

def accumulation_weight(step, batch_images, total_images, micro_batch, accumulate):
    """Weight a mean microbatch loss by its actual share of the update window."""
    first = (step // accumulate) * accumulate
    window_images = min(micro_batch * accumulate, total_images - first * micro_batch)
    if window_images <= 0 or not 0 < batch_images <= micro_batch:
        raise ValueError('Invalid gradient accumulation window')
    return batch_images / window_images

def total_loss(model, prediction, target, meta, donor_loss, supervision, ramp):
    native = prediction.get('native_output', prediction.get('donor_output'))
    if native is None:
        raise ValueError('The complete native reconstruction output must be retained')
    recipe = getattr(model, 'adaptation_recipe', {})
    profile = bool(recipe.get('speed_benchmark'))
    def stage(name):
        return torch.profiler.record_function(name) if profile else nullcontext()
    with stage('loss/native_reconstruction'):
        total, parts = donor_loss(native, target, meta)
    parts = {'native_' + key: value for key, value in parts.items()}
    has_refinement = bool(prediction.get('joint_rounds', prediction.get('adaptation_rounds')))
    prepared = None
    if recipe.get('speed_equivalent_optimizations', False) and has_refinement:
        with stage('targets/shared_geometry_and_visibility'):
            prepared = supervision.build_targets(target, prediction['object_ids'], meta['ann_id'])
        with stage('loss/native_image_reusing_visibility'):
            image_loss = native_image_loss_from_prepared(model.evidence, prepared)
    else:
        with stage('targets_and_loss/native_image_original'):
            image_loss, _ = model.evidence_auxiliary_loss(target, meta)
    total = total + image_loss
    parts['native_image'] = image_loss
    if has_refinement:
        with stage('targets_and_loss/refinement'):
            additions = supervision.losses(prediction, target, meta, prepared=prepared, ramp=ramp)
        if not isinstance(additions, dict) or not additions:
            raise ValueError('IACP supervision must return nonempty weighted scalar losses')
        for name, value in additions.items():
            if not torch.is_tensor(value) or value.ndim != 0:
                raise ValueError('Invalid loss value: ' + name)
        if set(parts).intersection(additions):
            raise ValueError('Duplicate native/IACP loss names')
        parts.update(additions)
        total = total + sum(additions.values())
    return total, parts

def native_image_loss_from_prepared(evidence, prepared):
    """Use the original loss arithmetic with one already computed z-buffer.

    The joint target builder includes the same original 73+431+64 queries.
    Camera fitting, triangle rasterization and target construction therefore
    run once, while every original supervision term and diagnostic is retained.
    """
    target = prepared['native_image_target']
    label, valid, visibility = target['label'], target['valid'], target['visible']
    total = prepared['human'].new_zeros(())
    for trace in evidence.trace:
        logprob = torch.cat((trace['score'], trace['null']), -1).log_softmax(-1)
        good = valid & visibility
        ce = -(logprob.gather(-1, label[..., None]).squeeze(-1) * good).sum() / good.sum().clamp_min(1)
        total = total + .02 * ce / len(evidence.trace)
        hidden = valid & ~visibility
        total = total + .02 * (-logprob[..., -1] * hidden).sum() / hidden.sum().clamp_min(1) / len(evidence.trace)
        bce = torch.nn.functional.binary_cross_entropy(trace['visible'][:, 73:, 0], visibility[:, 73:].float(), reduction='none')
        total = total + .02 * (bce * valid[:, 73:]).sum() / valid[:, 73:].sum().clamp_min(1) / len(evidence.trace)
    with torch.no_grad():
        top = evidence.trace[-1]['weights'].topk(8, -1).indices
        top_hit = ((top == label[..., None]).any(-1) & valid & visibility).sum() / (valid & visibility).sum().clamp_min(1)
        values = torch.stack((prepared['target_camera_fit_error'].mean(), valid.float().mean(),
                              visibility[:, 73:].float().mean(), top_hit)).cpu().tolist()
    names = ('target_camera_fit_error', 'target_valid_fraction',
             'target_visible_surface_fraction', 'visible_target_in_top8')
    evidence.last_diagnostics.update(zip(names, values))
    return total
