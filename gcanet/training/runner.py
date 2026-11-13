"""Single-GPU, full-budget IACP joint adaptation with explicit resume state."""
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from gcanet.io import (atomic_checkpoint, external_state_hashes, load_torch,
                       sha256, source_fingerprint, write_json)
from .assets import SurfaceAssets
from .model import build_adaptation_model
from .objective import accumulation_weight, optimizer_for, total_loss
from .supervision import ContactAdaptationSupervision


def compact_payload(model, epoch, config, external, optimizer=None, **extra):
    state = {k: v for k, v in model.state_dict().items() if k not in external}
    result = dict(format="gcanet_joint_adaptation_v1", model=state, epoch=epoch,
                  config=config, external_frozen_hashes=external, **extra)
    if optimizer is not None:
        result.update(optimizer=optimizer.state_dict(), rng=dict(
            python=random.getstate(), numpy=np.random.get_state(),
            torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
    return result


def restore(model, optimizer, path, config, external):
    saved = load_torch(path)
    if saved.get("format") != "gcanet_joint_adaptation_v1" or saved.get("config") != config:
        raise ValueError("Resume requires an unchanged GCANet joint-adaptation recipe and assets")
    if saved.get("external_frozen_hashes") != external:
        raise ValueError("External frozen tensors differ from the resume checkpoint")
    current = model.state_dict()
    if set(saved["model"]) != set(current) - set(external):
        raise ValueError("Incomplete or incompatible adaptation checkpoint")
    if "optimizer" not in saved or "rng" not in saved or not 0 < saved["epoch"] < 50:
        raise ValueError("Resume needs a full optimizer/RNG checkpoint before epoch 50")
    for key, value in saved["model"].items():
        if not torch.is_tensor(value) or value.shape != current[key].shape or value.dtype != current[key].dtype:
            raise ValueError("Resume tensor shape or dtype mismatch: " + key)
    if set(saved["rng"]) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("Incomplete resume RNG state")
    saved_groups = saved["optimizer"].get("param_groups", [])
    if len(saved_groups) != len(optimizer.param_groups) or any(
        len(a["params"]) != len(b["params"]) or a.get("name") != b.get("name")
        for a, b in zip(saved_groups, optimizer.param_groups)):
        raise ValueError("Resume optimizer parameter groups differ")
    model.load_state_dict({**current, **saved["model"]}, strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    rng = saved["rng"]
    random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"]); torch.cuda.set_rng_state_all(rng["cuda"])
    return saved["epoch"] + 1, int(saved["image_visits"]), int(saved["optimizer_updates"])


def run(args):
    import cv2
    from gcanet.config import load_config, validate_assets
    from gcanet.reconstruction import runtime
    config = load_config(args.config)
    validate_assets(config)
    recipe = config["iacp_training"]
    if recipe["epochs"] != 50 or recipe["loss_ramp_epochs"] != 5:
        raise ValueError("The main adaptation recipe uses 50 epochs and a five-epoch loss ramp")
    for key in ("micro_batch", "accumulate", "workers", "seed"):
        if type(recipe[key]) is not int or recipe[key] < (0 if key in ("workers", "seed") else 1):
            raise ValueError("Invalid adaptation configuration: " + key)
    if not recipe["speed_equivalent_optimizations"]:
        raise ValueError("This release retains the verified shared-computation training path")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one visible CUDA GPU")
    torch.set_num_threads(4); cv2.setNumThreads(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    runtime.seed_all(recipe["seed"])
    runtime.initialize(config["data_root"], config["dino_checkpoint"])
    from gcanet.reconstruction.models.templates import smplh, obj_dict
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    train = runtime.dataset("train", training=True)
    assets = SurfaceAssets(obj_dict, smplh.faces).to("cuda")
    model = build_adaptation_model(args.gsf_checkpoint, assets, config["iacp_assets"], recipe)
    optimizer = optimizer_for(model, recipe)
    supervision = ContactAdaptationSupervision(assets)
    donor_loss = runtime.OriginalLoss()
    external = external_state_hashes(model)
    # Paths can move across machines; resume identity is bound to contents.
    identity = dict(recipe=recipe, source_sha256=source_fingerprint(),
        gsf_checkpoint_sha256=sha256(args.gsf_checkpoint),
        train_annotation_sha256=sha256(Path(config["data_root"]) / "base_data/annotations/behave_train.json"),
        canonical_manifest_sha256=sha256(Path(config["iacp_assets"]["canonical_root"]) / "canonical_manifest.json"),
        surface_assets=assets.manifest(), gpu_count=1)
    start, visits, updates = 1, 0, 0
    if args.resume:
        start, visits, updates = restore(model, optimizer, args.resume, identity, external)
        expected_visits = (start - 1) * len(train)
        expected_updates = (start - 1) * math.ceil(len(train) / (recipe["micro_batch"] * recipe["accumulate"]))
        if visits != expected_visits or updates != expected_updates:
            raise ValueError("Resume counters do not describe complete training epochs")
    write_json(output / "protocol.json", dict(identity, resume=args.resume,
        effective_batch=recipe["micro_batch"] * recipe["accumulate"],
        data_count=len(train), expected_epochs=50, no_sample_exclusions=True,
        checkpoint_selection="fixed epoch 50",
        training_role="coupled adaptation; export only IACP for inference",
        historical_distributed_equivalence=False))
    with (output / "training.jsonl").open("x") as log:
        for epoch in range(start, 51):
            started = time.monotonic()
            generator = torch.Generator().manual_seed(recipe["seed"] + 1000003 * epoch)
            batches = runtime.loader(train, recipe["micro_batch"], recipe["workers"], True, generator)
            model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            optimizer.zero_grad(set_to_none=True)
            seen, ids, totals = 0, [], {}
            for step, (inputs, target, metadata) in enumerate(batches):
                inputs, target, metadata = (runtime.move(v) for v in (inputs, target, metadata))
                n = len(inputs["img"])
                fraction = (step + 1) / len(batches)
                progress = (epoch - 1 + fraction) / 50
                scale = recipe["lr_min_ratio"] + (1 - recipe["lr_min_ratio"]) * .5 * (1 + math.cos(math.pi * progress))
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * scale
                ramp = min(1., (epoch - 1 + fraction) / 5)
                # Match the historical global-update perturbation seed across
                # all microbatches contributing to this single-GPU update.
                model.adaptation_training_calls.fill_(updates)
                prediction = model(inputs, meta_info=metadata, targets=target, auxiliary=True)
                loss, parts = total_loss(model, prediction, target, metadata, donor_loss, supervision, ramp)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {step}")
                weight = accumulation_weight(step, n, len(train), recipe["micro_batch"], recipe["accumulate"])
                (loss * weight).backward()
                if (step + 1) % recipe["accumulate"] == 0 or step + 1 == len(batches):
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_clip_norm"])
                    if not bool(torch.isfinite(norm)):
                        raise FloatingPointError("Nonfinite gradient")
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    updates += 1
                for name, value in dict(total=loss, **parts).items():
                    totals[name] = totals.get(name, 0.) + float(value.detach()) * n
                seen += n; visits += n
                ids.extend(map(int, metadata["ann_id"].tolist()))
                if step % 50 == 0:
                    print(f"IACP epoch {epoch}/50: {seen}/{len(train)} images, update {updates}, loss {float(loss.detach()):.6g}", flush=True)
                del prediction, loss, parts
            if seen != len(train) or len(set(ids)) != len(train):
                raise RuntimeError("Incomplete or duplicated training epoch")
            row = dict(epoch=epoch, image_visits=visits, optimizer_updates=updates,
                       sample_count=seen, losses={k: v / seen for k, v in totals.items()},
                       seconds=time.monotonic() - started)
            log.write(json.dumps(row, allow_nan=False) + "\n"); log.flush()
            extra = dict(image_visits=visits, optimizer_updates=updates)
            atomic_checkpoint(output / "latest.pt", compact_payload(
                model, epoch, identity, external, optimizer, **extra))
            if epoch in (25, 50):
                atomic_checkpoint(output / f"epoch_{epoch:02d}.pt", compact_payload(
                    model, epoch, identity, external, **extra))
            write_json(output / "progress.json", row)
    expected = math.ceil(len(train) / (recipe["micro_batch"] * recipe["accumulate"])) * 50
    if visits != len(train) * 50 or updates != expected:
        raise RuntimeError("Full training budget was not completed")
    if external_state_hashes(model) != external:
        raise RuntimeError("Training changed a frozen parameter or buffer")
    contact_state = model.iacp.state_dict()
    contact_external = {k[len("iacp."):]: v for k, v in external.items() if k.startswith("iacp.")}
    atomic_checkpoint(output / "iacp_final.pt", dict(format="gcanet_iacp_v1", epoch=50,
        model={k: v for k, v in contact_state.items() if k not in contact_external},
        external_frozen_hashes=contact_external,
        canonical_asset_hashes=model.iacp.canonical_asset_hashes, config=identity))
    write_json(output / "completion.json", dict(complete=True, epoch=50,
        image_visits=visits, optimizer_updates=updates, exported_branch="IACP"))

