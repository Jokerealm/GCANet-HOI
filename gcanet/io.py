"""Checkpoint integrity and safe output handling."""
import hashlib
import inspect
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    files = sorted([*root.joinpath("gcanet").rglob("*.py"), *root.joinpath("scripts").glob("*.py")])
    records = {str(path.relative_to(root)): sha256(path) for path in files}
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def load_torch(path):
    """Load a user-supplied trusted checkpoint, including legacy metadata."""
    import torch
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    if "mmap" in inspect.signature(torch.load).parameters:
        options["mmap"] = True
    return torch.load(path, **options)


def tensor_sha256(value):
    import torch
    value = value.detach().cpu()
    digest = hashlib.sha256()
    digest.update(str((tuple(value.shape), str(value.dtype), str(value.layout))).encode())
    def add(tensor):
        digest.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    if value.layout == torch.sparse_coo:
        value = value.coalesce()
        add(value.indices()); add(value.values())
    elif value.layout == torch.strided:
        add(value)
    else:
        raise ValueError("Unsupported checkpoint tensor layout")
    return digest.hexdigest()


def frozen_state_names(model):
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    prefixes = []
    for name, module in model.named_modules():
        if not name or any(name == p or name.startswith(p + ".") for p in prefixes):
            continue
        parameters = list(module.parameters())
        if parameters and all(not p.requires_grad for p in parameters):
            prefixes.append(name)
    return sorted(n for n in model.state_dict() if n in frozen or
                  any(n == p or n.startswith(p + ".") for p in prefixes))


def external_state_hashes(model):
    state = model.state_dict()
    return {name: tensor_sha256(state[name]) for name in frozen_state_names(model)}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def atomic_checkpoint(path, value):
    import torch
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
