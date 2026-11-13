#!/usr/bin/env python3
"""Adapt GCANet's IACP on BEHAVE and export its inference checkpoint."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/behave.json")
    parser.add_argument("--gsf-checkpoint", required=True, help="Completed epoch-50 GSF checkpoint")
    parser.add_argument("--output", required=True, help="New output directory, including when resuming")
    parser.add_argument("--gpu", required=True, help="One physical GPU index")
    parser.add_argument("--resume", help="Explicit complete-epoch joint adaptation checkpoint")
    args = parser.parse_args()
    if not args.gpu.isdigit():
        parser.error("--gpu requires one physical GPU index")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["XFORMERS_DISABLED"] = "1"
    from gcanet.training.runner import run
    run(args)


if __name__ == "__main__":
    main()

