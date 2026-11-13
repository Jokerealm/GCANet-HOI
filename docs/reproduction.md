# Reproducing the BEHAVE main method

The deployed GCANet pipeline uses a reconstruction network with GSF, a separately retained adapted IACP branch, and BCR. Both neural predictors are frozen during BCR. BCR changes only the object's rotation and translation; the human mesh, object shape, scale, and connectivity stay fixed.

## Environment

The full historical BEHAVE inference pipeline ran under Python 3.10.20, PyTorch 2.6.0 + CUDA 11.8, torchvision 0.21.0, NumPy 1.26.4, SciPy 1.15.3, and SMPL-X/SMPL-H runtime package `smplx` 0.1.28. The README uses this Python/PyTorch combination and `requirements.txt` records the main Python dependencies.

PyTorch3D is required for canonical asset generation. The historical installation reports version 0.7.8 and was built from a local source tree; that version string alone does not identify a public Git revision. Follow the [official source-build instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md), build against the installed PyTorch, and retain the source revision and generated asset manifest for your reproduction. Installing an arbitrary PyTorch3D wheel for a different PyTorch build can fail at import or rendering time.

Environment preparation and CPU source tests are separate from a full training reproduction. A new machine, a rebuilt rendering dependency, and regenerated canonical assets require validation with the complete data and externally prepared model assets. This release does not include new 50-epoch training results.

Run the numerical, checkpoint, and entry-point tests without data or weights:

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
```

The tests cover contact lifting and gradients, pose derivatives, BCR acceptance and fallback, metric definitions, optimizer groups, and checkpoint/RNG restoration. New GSF checkpoints use `gcanet_gsf_v1`; the loader also accepts the original compact checkpoint format while retaining its tensor names.

## Training and checkpoint flow

1. Prepare the BEHAVE data, body models, graph operators, and external initialization assets following [data.md](data.md).
2. Train reconstruction with GSF and retain its epoch-50 checkpoint.
3. Initialize contact adaptation from the released Pi-HOC model and the trained reconstruction network. Train the joint adaptation procedure and retain its epoch-50 IACP branch.
4. For deployment, load the original reconstruction checkpoint from step 2 and the adapted contact branch from step 3. The adaptation-stage reconstruction and learned refiner are training components; deployment uses BCR.
5. Run inference and evaluate all 4,129 BEHAVE test frames, including frames where BCR returns the original object pose.

### Reconstruction with GSF

| Setting | BEHAVE recipe |
|---|---|
| Epochs / training seed | 50 / 20261014 |
| Physical batch / accumulation | 8 / 2 |
| Optimizer | Adam, betas `(0.5, 0.999)`, zero weight decay |
| Learning rate | `1e-4` for epochs 1–30; `1e-5` for epochs 31–50 |
| Trainable initialization | Hand4Whole body/hand initialization; remaining reconstruction and GSF parameters initialized for training |
| Frozen encoder | DINOv2-L/14 |
| Image-evidence losses | Patch retrieval, null retrieval, and visibility gate; weight `0.02` each |

Annotation crops use a 1.3 expansion factor and are warped to 512 × 512. The five-channel RGB/mask initializer input is resized to 256 × 256; GSF extracts DINOv2 features from 448 × 448 RGB. The three query groups contain 73 joints, 431 sampled body vertices, and 64 canonical object controls.

Training uses scale, rotation, and color augmentation. Test preprocessing is deterministic. Projection and visibility targets supervise GSF during training; inference retrieves image evidence using predicted geometry.

The two accumulated microbatches retain their own valid-count normalizations. Changing microbatch size can change these reductions even if the nominal effective batch size is unchanged.

### IACP adaptation

IACP adaptation retains the full contact predictor, including the selected-pair detector path, InteractionFormer, four-view contact decoder, and differentiable lifting to the body vertices. The 431 supplied body-contact labels are supervised at their known indices; other vertices are unknown. The historical topology has no fully annotated triangle, so the inherited dense pixel focal/Dice losses contribute zero. Contact supervision reaches the decoder through vertex lifting.

The training-only geometry branch and three-round refiner pass geometry gradients through the pair feature into InteractionFormer. Contact probability and presence are detached at the geometry interface. The exported branch is therefore the result of joint adaptation.

| Setting | Historical BEHAVE adaptation |
|---|---|
| Epochs / seed | 50 / 123 |
| GPUs / batch per GPU | 6 / 8 |
| Optimizer updates / image visits | 47,300 / 2,269,000 |
| Optimizer | Adam, betas `(0.5, 0.999)`, zero weight decay |
| Initial learning rates | Reconstruction `1e-5`; InteractionFormer `5e-6`; other trainable contact modules `1e-5`; learned refiner `1e-4` |
| Schedule | Per-update cosine decay to one tenth of each initial rate |
| Gradient clipping | Norm 10 |
| Frozen components | DINOv2, DETR, SAM image encoder and prompt encoder |
| Batch normalization | Fixed running statistics; trainable affine parameters |

The provided launcher accepts one physical GPU and defaults to microbatch 8 with six-step accumulation (nominal effective batch 48). This is a portable training configuration; it is not a step-for-step replay of the historical six-GPU schedule. Data ordering, distributed loss reduction, update count, and batch-normalization behavior must all be matched before claiming exact training reproduction. Single-GPU gradient accumulation alone does not establish that equivalence.

The reconstruction command writes `outputs/gsf/final.pt`. Contact adaptation writes the complete resume state to `latest.pt`, model snapshots to `epoch_25.pt` and `epoch_50.pt`, and the deployable branch to `iacp_final.pt` under its output directory. Evaluation loads the reconstruction and exported branch separately. To resume an interrupted training run, pass its `latest.pt` with `--resume` and choose a new output directory. The milestone snapshots and exported contact-only checkpoint do not contain the optimizer and RNG state required for resuming.

## Body-guided Contact Refinement

The main-method settings are fixed:

| Setting | Value |
|---|---:|
| Pair-mask IoU threshold | 0.1 |
| Contact-presence threshold | 0.5 |
| Anchor probability threshold | 0.7 |
| Coarse / dense anchor budgets | 64 / 64 |
| Minimum retained anchors | 6 |
| Maximum anchor-to-surface distance | 0.08 m |
| Pseudo-Huber scale | 0.02 m |
| Initial-pose prior weight | 0.25 |
| Canonical prior controls | 64 |
| Gauss–Newton damping | `1e-6` |
| Per-step rotation / translation limits | 1 degree / 0.005 m |
| Line-search fractions | 1, 1/2, 1/4, 1/8 |
| Maximum accepted updates | 6 |
| Minimum camera-space control depth | 0.05 m, with `1e-4` m numerical tolerance |

The objective combines confidence-weighted robust distance to the complete object triangle surface with a displacement prior on the canonical controls. Every candidate update is checked using recomputed surface distances, total/contact energy, and camera-space depth. The human mesh is fixed throughout.

## Evaluation and reported result

The paper reports **3.87 cm human CD, 6.89 cm object CD, 0.677 precision, 0.669 recall, and 0.673 F1** on all 4,129 test frames. The underlying reconstruction and contact predictors are retained at epoch 50. The final BCR evaluation was a complete cached evaluation with seed 123 and no sample exclusions; it did not train either network.

**Chamfer distance.** Fit a single similarity transform to the concatenated predicted human/object vertices and corresponding targets. Evaluate each aligned component using 6,000 area-sampled points per surface and the sum of the two mean Euclidean nearest-neighbor distances, expressed in centimeters. Because alignment is joint, human CD can change when BCR moves only the object.

**Contact.** Before similarity alignment, mark a body vertex in contact when its nearest decoded object vertex is closer than 0.05 m. Precision is averaged over frames with predicted contacts and recall over frames with annotated contacts. Report their harmonic mean, `2 * P * R / (P + R)`. This differs from averaging per-frame F1 and from evaluating the raw IACP probability head.

The 45,380 training frames and 4,129 test frames have disjoint images and sequences. BEHAVE test results were visible during method development. The reported result uses one training seed per predictor; it is not a multi-seed average.
