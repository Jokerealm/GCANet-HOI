# Third-party code and assets

GCANet includes adaptations of upstream reconstruction and contact-estimation code. Attribution is retained even where a module has been renamed to match the GCANet paper. The applicable license texts are included alongside the corresponding source or in this directory.

| Component | Upstream source | License / role |
|---|---|---|
| HOI-TG reconstruction framework | [ZhenrongWang/HOI-TG](https://github.com/ZhenrongWang/HOI-TG), reference commit `1dfbf5fe45e6525867b91b1654b2072a2921355a` | CC BY-NC 4.0; reconstruction, graph modules, data/evaluation infrastructure |
| CONTHO framework | [dqj5182/CONTHO_RELEASE](https://github.com/dqj5182/CONTHO_RELEASE) | Upstream framework inherited through HOI-TG; retain the supplied attribution and license notices |
| Hand4Whole | [mks0601/Hand4Whole_RELEASE](https://github.com/mks0601/Hand4Whole_RELEASE) | Human/body-hand initialization; external pretrained asset |
| Pi-HOC | [SravanChittupalli/Pi-HOC](https://github.com/SravanChittupalli/Pi-HOC), reference commit `f6e42b7bff6e703a4dd5b85e417ffbe889919861` | MIT; adapted contact predictor |
| DINOv2 | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Apache 2.0 for the included code; semantic encoder and InteractionFormer building blocks |
| DETR | [facebookresearch/detr](https://github.com/facebookresearch/detr) | Apache 2.0; frozen instance detector |
| Segment Anything | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) | Apache 2.0; canonical-view contact decoder components |
| TorchGeometry rotation conversion | [arraiyopensource/torchgeometry](https://github.com/arraiyopensource/torchgeometry) | Apache 2.0; a local conversion adapter preserves the original equations while supporting Boolean masks in newer PyTorch versions |
| PyTorch3D | [facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d) | Separate dependency; canonical-view rendering |
| InteractVLM canonical definitions | [saidwivedi/InteractVLM](https://github.com/saidwivedi/InteractVLM) | Source of the published canonical rendering definitions used by the asset generator |

SMPL-H body models, BEHAVE data, object meshes, pretrained weights, and derived canonical binary assets are acquired or generated separately under their respective terms. Their absence from this code release does not change the licenses of the included source files. No third-party license is replaced by the GCANet module naming.
