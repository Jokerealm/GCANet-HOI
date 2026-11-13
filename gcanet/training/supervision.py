"""Terminal and response supervision for IACP joint adaptation.

Native contact annotations are supervised only at their actual 431 vertex
identities. The separate relation heads use explicitly named GT geometric
five-centimetre proximity labels. There is no invented dense contact GT.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .geometry_supervision import JointSupervision, balanced_contact_bce, image_auxiliary_loss


def selected_mean(value, selected):
    selected = selected.to(value)
    while selected.ndim < value.ndim:
        selected = selected[..., None]
    expanded = selected.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1)


class ContactAdaptationSupervision(JointSupervision):
    """Three-round supervision; natural and intervention trajectories equal."""
    def __init__(self, assets, point_chunk=64, raster_function=None):
        super().__init__(assets, point_chunk=point_chunk, raster_function=raster_function)

    def losses(self, output, targets, meta, prepared=None, ramp=1.):
        if not 0 <= float(ramp) <= 1:
            raise ValueError('Loss ramp must lie in [0,1]')
        natural = output.get('adaptation_rounds', output.get('joint_rounds', []))
        if not natural:
            return {}
        auxiliary = output.get('auxiliary_rounds', [])
        if len(natural) != 3 or (auxiliary and len(auxiliary) != 3):
            raise ValueError('IACP adaptation recipe requires three correction rounds')
        if prepared is None:
            prepared = self.build_targets(targets, output['object_ids'], meta['ann_id'])
        truth_object = prepared['loss_points'] @ prepared['rotation'].transpose(1, 2) + prepared['translation'][:, None]
        native = output.get('native_output', output['donor_output'])
        result = {}
        trajectories = [('natural', natural, native['smpl_verts6000'])]
        if auxiliary:
            trajectories.append(('response', auxiliary, output['auxiliary_initial']['human']))
        trajectory_weight = 1. / len(trajectories)
        for role, rounds, origin in trajectories:
            for index, (state, round_weight) in enumerate(zip(rounds, (1 / 6., 2 / 6., 3 / 6.))):
                predicted_object = prepared['loss_points'] @ state['rotation'].float().transpose(1, 2) + state['translation'].float()[:, None]
                # Unweighted terminal errors: predicted confidence/stiffness
                # cannot hide a hard example or erase its supervision.
                final = F.l1_loss(state['human'].float(), prepared['human']) + F.l1_loss(predicted_object, truth_object)
                relation = self.relation_loss(state, prepared)
                geometric_contact = .5 * (balanced_contact_bce(state['contact_logits_h'], prepared['contact_target_h']) +
                                           balanced_contact_bce(state['contact_logits_o'], prepared['contact_target_o']))
                image = image_auxiliary_loss(state['image_trace'], prepared['image_target'])
                # Apply a weak human structure regularizer against the original
                # native surface for natural states, and the clean GT surface
                # for interventions, never against a perturbed target.
                reference = native['smpl_verts6000'] if role == 'natural' else prepared['human']
                structure = self.surface_loss(state['human'], reference)
                factor = float(ramp) * trajectory_weight * round_weight
                values = [('final', final, 1.), ('support', relation, .1),
                          ('geometric_contact_5cm', geometric_contact, .1),
                          ('image', image, 1.), ('structure', structure, .01)]
                if role == 'response':
                    masks = output['auxiliary_masks']
                    preserve_h = selected_mean((state['human'].float() - prepared['human']).abs(), masks['protect_human'])
                    preserve_o = selected_mean((predicted_object - truth_object).abs(), masks['protect_object'])
                    values.append(('protect_unperturbed', preserve_h + preserve_o, 1.))
                for name, value, coefficient in values:
                    result[f'{role}_round{index + 1}_{name}'] = factor * coefficient * value
        # The contact module implements the original projected-contact losses
        # with a sparse validity mask. Valid-count diagnostics are not losses.
        contact_parts = output.get('iacp_losses', {})
        for name, value in contact_parts.items():
            if torch.is_tensor(value) and value.ndim == 0 and value.requires_grad:
                result['iacp_' + name] = .1 * value
        if not contact_parts and torch.is_grad_enabled():
            raise ValueError('Training output is missing sparse native IACP contact supervision')
        return result

    forward = losses
