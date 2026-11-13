"""Explicit portable paths and the BEHAVE main-method configuration."""
import json
from pathlib import Path


def load_config(path):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())
    if config.get("dataset") != "BEHAVE":
        raise ValueError("This release supports the BEHAVE main method")
    # Paths in the checked-in config are relative to the repository root.
    root = path.parent.parent if path.parent.name == "configs" else path.parent
    def resolve(value):
        value = Path(value).expanduser()
        return str((root / value).resolve()) if not value.is_absolute() else str(value)
    config["data_root"] = resolve(config["data_root"])
    config["dino_checkpoint"] = resolve(config["dino_checkpoint"])
    for key in ("pretrained_checkpoint", "canonical_root"):
        config["iacp_assets"][key] = resolve(config["iacp_assets"][key])
    return config


def validate_assets(config, require_canonical=True):
    data = Path(config["data_root"])
    paths = [data / "base_data/annotations/behave_train.json",
             data / "base_data/annotations/behave_test.json",
             data / "base_data/human_models/smplh/SMPLH_NEUTRAL.pkl",
             data / "base_data/human_models/smplh/SMPLH_MALE.pkl",
             data / "base_data/human_models/smplh/SMPLH_FEMALE.pkl",
             data / "base_data/human_models/smpl_downsampling.npz",
             data / "base_data/human_models/smpl_upsampling.npz",
             data / "base_data/human_models/smpl_mean_params.npz",
             data / "base_data/graph/smpl_431_adjmat_indices.pt",
             data / "base_data/graph/smpl_431_adjmat_values.pt",
             data / "base_data/graph/smpl_431_adjmat_size.pt",
             data / "base_data/object_models/behave/_info.json",
             Path(config["dino_checkpoint"]),
             Path(config["iacp_assets"]["pretrained_checkpoint"])]
    if require_canonical:
        paths.append(Path(config["iacp_assets"]["canonical_root"]) / "canonical_manifest.json")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Prepare the external assets described in docs/data.md:\n" + "\n".join(missing))

