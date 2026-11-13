"""The coupled training graph used to adapt IACP; it is not deployed BCR."""
import types

import torch

from .geometry import AdaptationGeometry, historical_pose_fit, _device_asset
from .refiner import ContactAdaptationRefiner, perturb_training_state


def build_adaptation_model(gsf_checkpoint, assets, contact_assets, recipe, device="cuda"):
    from gcanet.reconstruction import runtime, checkpoint
    from gcanet.iacp import InstanceAwareContactPrior
    from gcanet.io import load_torch

    model = runtime.build_model(device=device, initialize_backbone=False)
    saved = load_torch(gsf_checkpoint)
    if saved.get("epoch") != 50:
        raise ValueError("IACP adaptation starts from the completed epoch-50 GSF checkpoint")
    checkpoint.load_model(model, saved)
    del saved
    model.requires_grad_(True)
    model.evidence.dino.backbone.requires_grad_(False)
    model.joint_geometry = AdaptationGeometry(model, device)
    model.joint_geometry.enable_speed_equivalent_optimizations()
    model.joint_assets = assets
    model.adaptation_recipe = dict(recipe)
    model.register_buffer("adaptation_training_calls", torch.zeros((), dtype=torch.long, device=device))
    original_forward = model.forward
    original_readout = model.kps_to_Rt

    def save_tokens(module, arguments):
        model._adaptation_tokens = arguments[0]
    model._adaptation_hook = model.tgformer.trans_linear4.register_forward_pre_hook(save_tokens)
    model._readout_checked = False

    def differentiable_readout(self, predicted, object_ids):
        ids = torch.as_tensor(object_ids, dtype=torch.long, device=predicted.device)
        current = historical_pose_fit(self.joint_geometry.canonical64[ids], predicted)
        if not self._readout_checked:
            from .geometry import compare_legacy_pose_mapping
            with torch.no_grad():
                legacy = original_readout(predicted, object_ids)
                report = compare_legacy_pose_mapping(*current, *legacy, object_ids, assets)
                if not report["passed"]:
                    raise RuntimeError("Differentiable pose readout changed GSF geometry: " + str(report))
            self._readout_checked = True
        return current
    model.kps_to_Rt = types.MethodType(differentiable_readout, model)
    model.iacp = InstanceAwareContactPrior(
        contact_assets, activation_checkpointing=True,
        speed_equivalent_optimizations=True,
        checkpoint_policy=recipe.get("contact_checkpoint_policy", "all"),
        annotation_vertex_indices=model.joint_geometry.human_indices).to(device)
    model.iacp.prepare_fixed_assets(offload_frozen_image_encoder=True)
    devices = [] if torch.device(device).type == "cpu" else [torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(recipe["seed"]) + 314159)
        model.adaptation_refiner = ContactAdaptationRefiner(
            edge_chunk=recipe.get("edge_chunk", 32), speed_equivalent_optimizations=True).to(device)

    def forward(self, inputs, meta_info=None, targets=None, auxiliary=False):
        native = original_forward(inputs)
        image = inputs["img"]
        if image.shape[1] != 5:
            raise ValueError("Expected RGB, human mask and object mask")
        contact = self.iacp(image[:, :3], image[:, 3], image[:, 4])
        output = dict(native, native_output=native, donor_output=native, iacp=contact)
        if targets is not None and self.training:
            annotation = targets["h_contacts"].float().reshape(len(image), -1)
            indices = self.joint_geometry.human_indices
            if annotation.shape[1] != len(indices):
                raise ValueError("BEHAVE training requires the original 431 contact labels")
            labels = torch.zeros_like(contact["probabilities"])
            valid = torch.zeros_like(labels, dtype=torch.bool)
            labels[:, indices] = annotation
            valid[:, indices] = True
            output["iacp_losses"] = self.iacp.contact_loss(contact, labels, valid_mask=valid)
        ids = inputs["obj_id"].reshape(-1)
        asset = _device_asset(self.joint_assets.batch(ids), native["obj_trans"].device)
        tokens, memory = self._adaptation_tokens, self.evidence.memory
        context = self.adaptation_refiner.prepare_context(
            native, tokens, memory, asset, self.joint_geometry, contact)
        rounds = self.adaptation_refiner(native, tokens, memory, asset,
                                        self.joint_geometry, contact, context=context)
        final = rounds[-1]
        canonical = self.joint_geometry.canonical64[ids.long()]
        output.update(smpl_verts6000=final["human"],
                      smpl_verts1730=self.joint_geometry.mid(final["human"]),
                      smpl_verts431=final["human_coarse"], obj_pose=final["rotation"],
                      obj_trans=final["translation"],
                      obj_verts=canonical @ final["rotation"].transpose(1, 2) + final["translation"][:, None],
                      refh3d_keypoints=final["joints"],
                      refh2d_keypoints=self.projection(final["joints"], native["cam_trans"]),
                      joint_rounds=rounds, adaptation_rounds=rounds,
                      object_ids=ids, auxiliary_rounds=[])
        if auxiliary:
            if not self.training or targets is None or meta_info is None:
                raise ValueError("Auxiliary target interventions are training-only")
            initial, masks = perturb_training_state(
                native, targets, self.joint_geometry, meta_info["ann_id"],
                int(self.adaptation_training_calls), seed=recipe["seed"], native_state=context["native_state"])
            output["auxiliary_rounds"] = self.adaptation_refiner(
                native, tokens, memory, asset, self.joint_geometry, contact, initial=initial, context=context)
            output.update(auxiliary_initial=initial, auxiliary_masks=masks)
            self.adaptation_training_calls.add_(1)
        return output
    model.forward = types.MethodType(forward, model)
    return model

