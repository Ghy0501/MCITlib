# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Use DeepSpeed's trainable-only recovery format with a verified frozen base."""
import hashlib
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

RECEIPT = "video_frozen_parts.json"


def binding(files):
    entries = {}
    for file in files:
        path = Path(file).resolve(strict=True)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 2**20), b""):
                digest.update(block)
        entries[str(path)] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    if not entries:
        raise ValueError("A compact native recovery snapshot requires immutable base and tower assets")
    return entries


def recovery_scope(files):
    return native_recovery(files) if files else nullcontext()


@contextmanager
def native_recovery(files):
    from deepspeed.runtime.engine import DeepSpeedEngine

    frozen = binding(files)
    save, load = DeepSpeedEngine.save_checkpoint, DeepSpeedEngine.load_checkpoint

    def save_boundary(engine, directory, *args, **kwargs):
        kwargs["exclude_frozen_parameters"] = True
        saved = save(engine, directory, *args, **kwargs)
        if not saved:
            raise RuntimeError("DeepSpeed failed to save the compact native boundary")
        if torch.distributed.get_rank() == 0:
            with (Path(directory) / RECEIPT).open("x") as handle:
                json.dump(frozen, handle, indent=2)
        torch.distributed.barrier()
        return saved

    def load_boundary(engine, directory, *args, **kwargs):
        path = Path(directory)
        if json.loads((path / RECEIPT).read_text()) != frozen:
            raise ValueError("Immutable video assets changed since the DGG native save")
        tag = (path / "latest").read_text().strip()
        model_file = path / tag / "mp_rank_00_model_states.pt"
        checkpoint = torch.load(model_file, map_location="cpu")
        names = {name: p for name, p in engine.module.named_parameters() if p.requires_grad}
        state = checkpoint["module"]
        if any(name not in state or state[name].shape != p.shape for name, p in names.items()):
            raise ValueError("DGG native recovery is missing trainable tensor state")
        kwargs["load_module_strict"] = False
        return load(engine, directory, *args, **kwargs)

    DeepSpeedEngine.save_checkpoint = save_boundary
    DeepSpeedEngine.load_checkpoint = load_boundary
    try:
        yield
    finally:
        DeepSpeedEngine.save_checkpoint = save
        DeepSpeedEngine.load_checkpoint = load
