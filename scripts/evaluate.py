#!/usr/bin/env python3
"""Evaluate the GCANet main method on all 4,129 BEHAVE test frames."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/behave.json")
    parser.add_argument("--gsf-checkpoint", required=True)
    parser.add_argument("--iacp-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", required=True, help="One physical GPU index")
    args = parser.parse_args()
    if not args.gpu.isdigit():
        parser.error("--gpu requires one physical GPU index")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["XFORMERS_DISABLED"] = "1"
    import random
    import numpy as np
    import torch
    import cv2
    from gcanet.config import load_config, validate_assets
    from gcanet.model import build_model
    from gcanet.reconstruction import runtime
    from gcanet.evaluation import evaluate_batch, paper_summary
    from gcanet.io import sha256, write_json
    config = load_config(args.config)
    validate_assets(config)
    recipe = config["evaluation"]
    seed = recipe["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1); cv2.setNumThreads(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Evaluation requires one visible CUDA GPU")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir()
    identity = dict(gsf_checkpoint_sha256=sha256(args.gsf_checkpoint),
                    iacp_checkpoint_sha256=sha256(args.iacp_checkpoint), config=config,
                    gsf_epoch=50, iacp_epoch=50,
                    batch_size=recipe["batch_size"], seed=seed, gpu_count=1,
                    no_sample_exclusions=True, expected_frames=4129)
    write_json(output / "protocol.json", identity)
    model = build_model(config, args.gsf_checkpoint, args.iacp_checkpoint)
    data = runtime.dataset("test", training=False)
    batches = runtime.loader(data, recipe["batch_size"], recipe["workers"], training=False)
    rows, observed = [], set()
    with (output / "frames.jsonl").open("x") as log, (output / "bcr_diagnostics.jsonl").open("x") as diagnostics:
        for index, (inputs, targets, metadata) in enumerate(batches):
            prediction = model(runtime.move(inputs))
            current = evaluate_batch(prediction, targets, metadata, seed)
            for row, report in zip(current, prediction["bcr_diagnostics"]):
                if row["ann_id"] in observed:
                    raise ValueError("Duplicate annotation in test evaluation")
                observed.add(row["ann_id"])
                log.write(json.dumps(row, allow_nan=False) + "\n")
                diagnostics.write(json.dumps(dict(ann_id=row["ann_id"], **report), allow_nan=False) + "\n")
            rows.extend(current)
            arrays = {key: value.detach().cpu().numpy() for key, value in prediction.items() if torch.is_tensor(value)}
            arrays["ann_ids"] = metadata["ann_id"].numpy()
            np.savez_compressed(output / "predictions" / f"batch_{index:05d}.npz", **arrays)
            print(f"Evaluated {len(rows)}/{len(data)} frames", flush=True)
    if observed != {int(row["ann_id"]) for row in data.datalist} or len(rows) != 4129:
        raise RuntimeError("Incomplete full-test coverage")
    result = dict(paper_summary(rows), protocol=identity)
    write_json(output / "metrics.json", result)
    write_json(output / "completion.json", dict(complete=True, sample_count=len(rows)))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
