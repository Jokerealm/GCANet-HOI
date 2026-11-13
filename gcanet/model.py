"""GCANet inference: GSF reconstruction, IACP anchors, and rigid BCR."""
import torch
from torch import nn

from .bcr import BodyGuidedContactRefinement


class GCANet(nn.Module):
    def __init__(self, reconstruction, iacp):
        super().__init__()
        from .reconstruction.models.templates import smplh, obj_dict
        self.reconstruction, self.iacp = reconstruction, iacp
        self.bcr = BodyGuidedContactRefinement()
        down0, down1 = [matrix.coalesce() for matrix in smplh._D]
        def selection(matrix):
            indices, values = matrix.indices(), matrix.values()
            if matrix._nnz() != matrix.shape[0] or not bool((values == 1).all()) or not torch.equal(
                    indices[0], torch.arange(matrix.shape[0], device=indices.device)):
                raise ValueError("Expected the original fixed body vertex selections")
            return indices[1]
        self.register_buffer("human_indices", selection(down0)[selection(down1)], persistent=False)
        self.objects = obj_dict
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, inputs):
        """Only RGB, instance masks, object identity and template graph are inputs."""
        initial = self.reconstruction(inputs)
        image = inputs["img"]
        contact = self.iacp(image[:, :3], image[:, 3], image[:, 4])
        rotations, translations, reports = [], [], []
        for i, object_id in enumerate(inputs["obj_id"].reshape(-1).tolist()):
            human = initial["smpl_verts6000"][i]
            template = self.objects[int(object_id)]
            mesh = template.load_template()
            rotation, translation, report = self.bcr(
                human=human, rotation=initial["obj_pose"][i], translation=initial["obj_trans"][i],
                probabilities=contact["probabilities"][i], coverage=contact["coverage"][i],
                presence=contact["presence"][i],
                vertices=torch.as_tensor(mesh.vertices, dtype=torch.float32, device=human.device),
                faces=torch.as_tensor(mesh.faces, dtype=torch.long, device=human.device),
                human_indices=self.human_indices.to(human.device),
                object_control_points=torch.as_tensor(template.load_verts(), device=human.device),
                camera=initial["cam_trans"][i])
            rotations.append(rotation); translations.append(translation); reports.append(report)
        # The human array is passed through exactly. Object shape/scale are fixed.
        return {"smpl_verts6000": initial["smpl_verts6000"],
                "obj_pose": torch.stack(rotations), "obj_trans": torch.stack(translations),
                "cam_trans": initial["cam_trans"], "contact_probabilities": contact["probabilities"],
                "contact_presence": contact["presence"], "contact_coverage": contact["coverage"],
                "bcr_diagnostics": reports}


def build_model(config, gsf_checkpoint, iacp_checkpoint, device="cuda"):
    from .reconstruction import runtime, checkpoint
    from .iacp import InstanceAwareContactPrior
    from .io import load_torch
    runtime.initialize(config["data_root"], config["dino_checkpoint"])
    reconstruction = runtime.build_model(device=device, initialize_backbone=False)
    saved = load_torch(gsf_checkpoint)
    if saved.get("epoch") != 50:
        raise ValueError("The main-method endpoint requires the epoch-50 GSF checkpoint")
    checkpoint.load_model(reconstruction, saved)
    del saved
    contact = InstanceAwareContactPrior(config["iacp_assets"], activation_checkpointing=False,
        speed_equivalent_optimizations=True, checkpoint_policy="none")
    contact_report = contact.load_adapted_checkpoint(iacp_checkpoint)
    if contact_report["epoch"] != 50:
        raise ValueError("The main-method endpoint requires the epoch-50 IACP checkpoint")
    contact = contact.to(device)
    model = GCANet(reconstruction, contact)
    contact.prepare_fixed_assets(offload_frozen_image_encoder=True)
    return model

