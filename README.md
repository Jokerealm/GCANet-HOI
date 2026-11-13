<div align="center">

# GCANet: Geometric and Contact-Anchored Monocular 3D Human–Object Reconstruction

[Code](https://github.com/Jokerealm/GCANet-HOI) · [Data preparation](docs/data.md) · [Reproduction guide](docs/reproduction.md)

</div>

**GCANet** reconstructs a human and an interacting object from a single RGB image, instance masks, and a canonical object mesh. It connects global reconstruction with local contact through three modules:

- **Geometry-conditioned Semantic Fusion (GSF)** retrieves DINOv2 image evidence using geometric queries and regulates its contribution through visibility-aware gating.
- **Instance-aware Contact Prior (IACP)** predicts contact for the selected human–object pair and transfers it to confidence-weighted anchors on the reconstructed human surface.
- **Body-guided Contact Refinement (BCR)** balances contact agreement with an initial-pose prior to refine object rotation and translation while preserving the human mesh and object shape.

This release contains the **main method on BEHAVE**, including reconstruction training, contact adaptation, inference, and evaluation. Datasets, body models, pretrained weights, and trained GCANet checkpoints are prepared separately.

| Paper module | Implementation |
|---|---|
| GSF — Geometry-conditioned Semantic Fusion | [`gcanet/gsf.py`](gcanet/gsf.py) |
| IACP — Instance-aware Contact Prior | [`gcanet/iacp.py`](gcanet/iacp.py) |
| BCR — Body-guided Contact Refinement | [`gcanet/bcr.py`](gcanet/bcr.py) |

## Installation

Create a dedicated environment. The complete contact pipeline uses Python 3.10 and PyTorch 2.6; the older HOI-TG environment alone does not supply all its dependencies.

```bash
git clone https://github.com/Jokerealm/GCANet-HOI.git
cd GCANet-HOI

conda create -n gcanet python=3.10
conda activate gcanet
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy==1.26.4 scipy==1.15.3
python -m pip install --no-build-isolation chumpy==0.70
python -m pip install -r requirements.txt
```

Install [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) against the same PyTorch environment to generate the canonical contact assets. Its compiled extensions must match the installed PyTorch and, for a CUDA build, the CUDA toolkit. See the [environment notes](docs/reproduction.md#environment) for the verified historical versions and the scope of installation validation.

## Data

Obtain the BEHAVE sequences and CONTHO `base_data`, add HOI-TG's graph and object metadata, and prepare the licensed SMPL-H models and external initialization assets. [docs/data.md](docs/data.md) gives the sources, exact paths, and commands for the HOI-TG additions and canonical asset generation. The repository does not require an HOI-TG checkout at runtime.

```text
GCANet-HOI/
├── data/
│   ├── base_data/
│   │   ├── annotations/          # behave_train.json, behave_test.json
│   │   ├── backbone_models/      # Hand4Whole initialization
│   │   ├── human_models/         # SMPL-H and mesh sampling operators
│   │   ├── object_models/behave/ # Canonical meshes and _info.json
│   │   └── graph/               # Human mesh adjacency
│   └── BEHAVE/sequences/        # RGB images and instance masks
├── assets/                      # DINOv2, Pi-HOC, canonical contact assets
├── outputs/                     # Locally trained checkpoints and evaluation results
└── configs/behave.json
```

The benchmark uses **45,380 training frames and 4,129 test frames**. Update the paths in `configs/behave.json` to match your local assets.

## Running GCANet

Run commands from the repository root. Each command takes one physical GPU ID and a new output directory.

### Train

Train the reconstruction network with GSF:

```bash
python scripts/train_gsf.py \
  --data-root data \
  --dino-checkpoint assets/dinov2_vitl14_pretrain.pth \
  --output outputs/gsf \
  --gpu 0
```

Adapt the contact predictor using the BEHAVE training procedure:

```bash
python scripts/train_iacp.py \
  --config configs/behave.json \
  --gsf-checkpoint outputs/gsf/final.pt \
  --output outputs/iacp \
  --gpu 0
```

Contact adaptation uses a training-only geometry branch and refiner; the final pipeline combines the separately trained GSF reconstruction with the exported IACP branch and BCR. The [reproduction guide](docs/reproduction.md) describes initialization, checkpoint selection, and the distinction between the provided single-GPU entry point and the historical distributed adaptation.

### Test

Training saves the retained reconstruction to `outputs/gsf/final.pt` and the exported contact branch to `outputs/iacp/iacp_final.pt`. Evaluate the complete GSF → IACP → BCR pipeline:

```bash
python scripts/evaluate.py \
  --config configs/behave.json \
  --gsf-checkpoint outputs/gsf/final.pt \
  --iacp-checkpoint outputs/iacp/iacp_final.pt \
  --output outputs/behave \
  --gpu 0
```

Checkpoint paths above refer to files you have trained or obtained separately; this code release does not bundle trained weights.

### Results

The main-paper BEHAVE result is:

| Method | Human CD ↓ (cm) | Object CD ↓ (cm) | Contact P ↑ | Contact R ↑ | Contact F1 ↑ |
|---|---:|---:|---:|---:|---:|
| GCANet | 3.87 | 6.89 | 0.677 | 0.669 | 0.673 |

These are the recorded paper results on all 4,129 test frames, using the retained epoch-50 predictors. They are not a new training result from this release. Human/object CD uses joint similarity alignment; contact uses the reconstructed placement before alignment and the benchmark's 5 cm threshold. See [metric definitions and evaluation protocol](docs/reproduction.md#evaluation-and-reported-result).

## Acknowledgements

Our implementation builds on [HOI-TG](https://github.com/ZhenrongWang/HOI-TG), [CONTHO](https://github.com/dqj5182/CONTHO_RELEASE), and [Hand4Whole](https://github.com/mks0601/Hand4Whole_RELEASE) for human–object reconstruction; [DINOv2](https://github.com/facebookresearch/dinov2) for semantic image features; and [Pi-HOC](https://github.com/SravanChittupalli/Pi-HOC), [DETR](https://github.com/facebookresearch/detr), and [Segment Anything](https://github.com/facebookresearch/segment-anything) for the contact predictor. Canonical-view preparation follows the published rendering definitions of [InteractVLM](https://github.com/saidwivedi/InteractVLM).

Upstream code retains its attribution and license notices. See [LICENSES/THIRD_PARTY.md](LICENSES/THIRD_PARTY.md).
