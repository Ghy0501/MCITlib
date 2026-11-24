# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Scoped DGG training integration for the pinned V4 LLaVA runtime."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from transformers import TrainerCallback

from .core import GradientGuidance
from .deepspeed_compat import optimizer_backward_hook


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_adapter(path: Path) -> dict:
    binary, safe = path / "adapter_model.bin", path / "adapter_model.safetensors"
    if binary.is_file():
        state = torch.load(binary, map_location="cpu")
    elif safe.is_file():
        from safetensors.torch import load_file
        state = load_file(str(safe), device="cpu")
    else:
        raise FileNotFoundError(f"DGG reference adapter is missing at {path}")
    return {key.replace(".lora_A.weight", ".lora_A.default.weight")
               .replace(".lora_B.weight", ".lora_B.default.weight"): value
            for key, value in state.items()}


def load_guidance(model, path: str, probability: float) -> GradientGuidance:
    """Load and validate a complete prior-task reference against current parameters."""
    directory = Path(path)
    history = _load_adapter(directory)
    non_lora = directory / "non_lora_trainables.bin"
    if not non_lora.is_file():
        raise FileNotFoundError(f"DGG reference non-LoRA state is missing: {non_lora}")
    history.update(torch.load(non_lora, map_location="cpu"))
    current = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing = current.keys() - history.keys()
    if not current or missing:
        raise ValueError(f"DGG reference does not cover trainable parameters: {sorted(missing)[:5]}")
    filtered = {}
    for name, parameter in current.items():
        tensor = history[name]
        if parameter.shape != tensor.shape:
            raise ValueError(f"DGG reference parameter shape mismatch: {name}")
        filtered[name] = tensor
    return GradientGuidance(filtered, probability)


def guard_reference(output: str, data_path: str, history_path: str | None, probability: float, resume: str | None,
                    expected_data_sha256: str | None = None, replay_manifest: str | None = None) -> None:
    """Tie a run or mid-task continuation to the exact data and anchor bytes."""
    data = Path(data_path)
    data_sha256 = _digest(data)
    if expected_data_sha256 is not None and data_sha256 != expected_data_sha256:
        raise ValueError("DGG primary data differs from the sample manifest")
    if replay_manifest is not None:
        prepared = json.loads(Path(replay_manifest).read_text(encoding="utf-8"))
        if (prepared.get("version") != 1 or prepared.get("output_sha256") != data_sha256 or
                Path(prepared.get("output", "")).resolve() != data.resolve()):
            raise ValueError("DGG replay data differs from its preparation manifest")
    record = {"version": 1, "data_sha256": data_sha256, "probability": probability, "history": None}
    if history_path:
        directory = Path(history_path)
        adapter = directory / "adapter_model.bin"
        if not adapter.is_file():
            adapter = directory / "adapter_model.safetensors"
        non_lora = directory / "non_lora_trainables.bin"
        record["history"] = {"adapter_sha256": _digest(adapter), "non_lora_sha256": _digest(non_lora)}
    destination = Path(output) / "dgg_reference.json"
    if resume:
        if not destination.is_file() or json.loads(destination.read_text()) != record:
            raise ValueError("DGG checkpoint resume requires unchanged data, history, probability and reference manifest")
    elif destination.exists():
        if json.loads(destination.read_text()) != record:
            raise ValueError("DGG output directory contains a different reference manifest")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")


class DGGCallback(TrainerCallback):
    """Install the callback at the released boundary, restoring it on exit."""

    def __init__(self, trainer, guidance: GradientGuidance):
        self.trainer = trainer
        self.guidance = guidance
        self.scope = None
        self.draws = self.applied = self.seen = self.changed = 0

    def on_train_begin(self, args, state, control, **kwargs):
        engine = self.trainer.model_wrapped
        if (not hasattr(engine, "zero_optimization_stage") or engine.zero_optimization_stage() != 2
                or not engine.bfloat16_enabled() or engine.world_size != 1):
            raise RuntimeError("DGG requires the reviewed single-GPU bf16 DeepSpeed ZeRO-2 execution path")

        def apply(current_engine):
            parameters = list(current_engine.module.named_parameters())
            self.draws += 1
            if self.guidance.apply(parameters):
                self.applied += 1
                self.seen += self.guidance.last_seen
                self.changed += self.guidance.last_changed
                if self.guidance.last_seen == 0:
                    raise RuntimeError("DGG callback saw no gradients; inspect the DeepSpeed reduction path")

        self.scope = optimizer_backward_hook(engine, apply)
        self.scope.__enter__()
        return control

    def evidence(self):
        return {"backward_calls": self.draws, "bernoulli_successes": self.applied,
                "gradient_tensors_seen": self.seen, "gradient_tensors_modified": self.changed}

    def close(self):
        if self.scope is not None:
            self.scope.__exit__(None, None, None)
            self.scope = None
