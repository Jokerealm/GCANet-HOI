# BEHAVE data and external assets

GCANet uses one RGB image, a person mask, an object mask, and a known canonical object mesh. Object identity selects the template, its 64 control points, and its graph. Mesh coordinates are in **meters**. The loader expects the processed BEHAVE format used by HOI-TG and CONTHO.

## Dataset

1. Download the Date01–Date07 BEHAVE sequences from the [official project page](https://virtualhumans.mpi-inf.mpg.de/behave/) into `data/BEHAVE/sequences/`, following the [CONTHO data instructions](https://github.com/dqj5182/CONTHO_RELEASE#data). The RGB frames and both mask files below must be present.
2. Download CONTHO's `base_data` from the links in its [data instructions](https://github.com/dqj5182/CONTHO_RELEASE#data) and unpack it at `data/base_data/`. Obtain the SMPL-H models separately under their license and place them as shown below.
3. Add the graph and object metadata from the HOI-TG release. The following uses the reference commit, and deliberately replaces CONTHO's `_info.json` with HOI-TG's graph-augmented version:

```bash
HOITG_BASE=https://raw.githubusercontent.com/ZhenrongWang/HOI-TG/1dfbf5fe45e6525867b91b1654b2072a2921355a/data/base_data
mkdir -p data/base_data/graph data/base_data/object_models/behave
for name in smpl_431_adjmat_indices.pt smpl_431_adjmat_values.pt smpl_431_adjmat_size.pt; do
  curl -fL "$HOITG_BASE/graph/$name" -o "data/base_data/graph/$name"
done
curl -fL "$HOITG_BASE/object_models/behave/_info.json" \
  -o data/base_data/object_models/behave/_info.json
```

The metadata refers to the object meshes inside `data/base_data/object_models/behave/`; it does not contain the meshes themselves. Keep the meshes from the prepared data. If adapting an existing `_info.json` with absolute paths, change its `path` fields to paths relative to the data root.

The two annotation files are `base_data/annotations/behave_train.json` and `base_data/annotations/behave_test.json`, relative to your configured data root. They contain 45,380 and 4,129 frames, respectively. The split contains 231 training sequences and 90 test sequences, with no shared images or sequences. All test frames are retained.

For an annotation image such as:

```text
BEHAVE/sequences/Date03_Sub03_backpack_back/t0003.000/k1.color.jpg
```

the loader reads the corresponding masks:

```text
BEHAVE/sequences/Date03_Sub03_backpack_back/t0003.000/k1.person_mask.jpg
BEHAVE/sequences/Date03_Sub03_backpack_back/t0003.000/k1.obj_rend_mask.jpg
```

The standard BEHAVE protocol supplies the person mask and obtains the object mask by projecting the fitted object mesh. Evaluation therefore assumes these masks and the annotated object identity are available. Arbitrary photographs require their own instance masks and compatible object template.

Training annotations include body parameters, 2D/3D joints, object pose, and sparse contact labels. Test annotations provide targets to the evaluator. Predicted geometry and the IACP contact field drive BCR; evaluation targets must not be used to choose its anchors or optimize its pose.

## Body and object assets

Prepare this layout below the data root:

```text
base_data/
├── annotations/
│   ├── behave_train.json
│   └── behave_test.json
├── backbone_models/
│   └── hand4whole.pth.tar
├── human_models/
│   ├── smplh/
│   │   ├── SMPLH_NEUTRAL.pkl
│   │   ├── SMPLH_MALE.pkl
│   │   └── SMPLH_FEMALE.pkl
│   ├── smpl_downsampling.npz
│   ├── smpl_upsampling.npz
│   └── smpl_mean_params.npz
├── object_models/behave/
│   ├── _info.json
│   ├── backpack.obj
│   └── ...
└── graph/
    ├── smpl_431_adjmat_indices.pt
    ├── smpl_431_adjmat_values.pt
    └── smpl_431_adjmat_size.pt
```

Register for and obtain SMPL-H from its [official website](https://mano.is.tue.mpg.de/). Use the compatible 6,890-vertex models and the supplied fixed mesh-sampling operators. The 431 supervised body indices must match these operators. Body models and derived binary assets are not redistributed in this repository.

The object `_info.json` contains each template path, 64 canonical control points (`kps`), and its 64 × 64 adjacency matrix (`graph`). Preserve the template axes, metric scale, control-point ordering, and graph ordering used during training. A mesh with a different orientation or a reordered point set is not an interchangeable asset.

## Initialization and trained checkpoints

| Asset | Purpose | Source |
|---|---|---|
| `hand4whole.pth.tar` | Initialize the trainable human reconstruction backbone | [Hand4Whole](https://github.com/mks0601/Hand4Whole_RELEASE), using the compatible initialization distributed with the HOI-TG/CONTHO preparation |
| `dinov2_vitl14_pretrain.pth` | Frozen DINOv2-L/14 encoder for GSF | [DINOv2 pretrained models](https://github.com/facebookresearch/dinov2#pretrained-models), ViT-L/14 without distillation |
| `pihoc.pt` | Initialize IACP's detector, InteractionFormer, and contact decoder | [Pi-HOC model preparation](https://github.com/SravanChittupalli/Pi-HOC#2-download-pi-hoc-files), file `checkpoints/pihoc/pihoc.pt` |
| Canonical contact assets | Fixed four-view rendering and barycentric lifting | Generate below from your SMPL-H model |
| Trained GSF checkpoint | Retained reconstruction network for deployment | Train using `scripts/train_gsf.py` |
| Adapted IACP checkpoint | BEHAVE contact branch for deployment | Export from the contact-adaptation training |

The released Pi-HOC initialization and the BEHAVE-adapted contact checkpoint have different roles. The generic initialization alone does not reproduce the adapted GCANet result. No GCANet checkpoint download is supplied in this code-only release.

Place the DINOv2 and Pi-HOC files at the default paths, or edit the two paths in `configs/behave.json`. The expected external initialization files used by the reference pipeline have these SHA-256 hashes:

```text
d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428  dinov2_vitl14_pretrain.pth
d93f3eb6f3aa001621288290eebc7940c73de59d185bac11c771aaea6b63463a  pihoc.pt
```

Compact GCANet checkpoints omit frozen tensors; evaluation verifies that the external initialization matches the checkpoint. Use the same initialization and canonical assets for training and testing.

## Canonical contact assets

With PyTorch3D installed, generate a new canonical directory:

```bash
python scripts/build_iacp_assets.py \
  --body-model data/base_data/human_models/smplh/SMPLH_NEUTRAL.pkl \
  --output assets/iacp_canonical
```

The generator uses the published canonical-view definitions, creates 1,024 × 1,024 normal-rendered body views, and records its inputs and output hashes. The result includes:

```text
assets/iacp_canonical/
├── body_render_norm_topfront.png
├── body_render_norm_bottomfront.png
├── body_render_norm_topback.png
├── body_render_norm_bottomback.png
├── pixel_to_vertex_map_1024.npz
├── bary_coords_map_1024.npz
├── canonical_mesh.npz
└── canonical_manifest.json
```

The model consumes views in the order **topfront, bottomfront, topback, bottomback**. Each valid pixel maps to a triangle's three vertex indices and barycentric weights. Pixels outside the body have no valid vertex association. The generated assets reproduce the local preparation procedure; they are not asserted to match unpublished upstream binaries byte for byte.

Finally, edit `configs/behave.json` to reference your data root, DINOv2 weights, Pi-HOC initialization, and generated canonical directory. Keep downloaded and generated assets outside source control.
