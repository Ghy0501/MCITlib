# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Persist the HF scheduler that the pinned DeepSpeed engine does not own.

DeepSpeed still owns the native model, optimizer, and master-weight checkpoint.
This sidecar records the separate Trainer scheduler without changing its group
ordering or learning rates.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import tempfile
from pathlib import Path

from transformers import TrainerCallback

SIDECAR = "dgg_scheduler_v1.json"
FORMAT = "mcitlib-dgg-hf-scheduler"
VERSION = 1


def _canonical(value):
    """Reject nonportable scheduler values instead of silently dropping them."""
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _function_signature(fn):
    if isinstance(fn, functools.partial):
        return {"function": _function_signature(fn.func), "args": _canonical(fn.args),
                "keywords": _canonical(fn.keywords or {})}
    module, name = getattr(fn, "__module__", None), getattr(fn, "__qualname__", None)
    if not module or not name or "<locals>" in name:
        raise ValueError("DGG scheduler callable has no stable module-qualified identity")
    return {"module": module, "name": name}


def _scheduler_and_groups(trainer):
    engine = trainer.model_wrapped
    zero = getattr(engine, "optimizer", None)
    optimizer = getattr(zero, "optimizer", None)
    scheduler = trainer.lr_scheduler
    if getattr(engine, "lr_scheduler", None) is not None:
        raise ValueError("DGG scheduler sidecar expects the reviewed HF-owned scheduler")
    # The pinned HF Trainer normally holds the raw scheduler. Support an
    # Accelerate wrapper only when it exposes the same underlying scheduler.
    underlying = getattr(scheduler, "scheduler", scheduler)
    if optimizer is None or getattr(underlying, "optimizer", None) is not optimizer:
        raise ValueError("DGG scheduler is not attached to the active ZeRO base optimizer")
    bit16_groups = getattr(zero, "bit16_groups", None)
    if bit16_groups is None or len(bit16_groups) != len(optimizer.param_groups):
        raise ValueError("DGG cannot identify the active ZeRO optimizer groups")
    names = {id(p): name for name, p in engine.module.named_parameters()}
    groups = []
    seen = set()
    for params, group in zip(bit16_groups, optimizer.param_groups):
        identity = []
        for param in params:
            name = names.get(id(param))
            if name is None or name in seen:
                raise ValueError("DGG optimizer group has missing or duplicate named parameters")
            seen.add(name)
            identity.append([name, list(param.shape), str(param.dtype)])
        if not identity:
            raise ValueError("DGG optimizer has an unexpected empty active group")
        encoded = json.dumps(identity, separators=(",", ":"), ensure_ascii=False).encode()
        options = _canonical({key: value for key, value in group.items() if key != "params"})
        groups.append({"parameter_count": len(identity), "parameters_sha256": hashlib.sha256(encoded).hexdigest(),
                       "options": options})
    return engine, scheduler, underlying, groups


def _signature(trainer, state, underlying):
    kind = trainer.args.lr_scheduler_type
    kind = getattr(kind, "value", kind)
    if not isinstance(kind, str):
        raise ValueError("DGG scheduler type is not a string")
    functions = getattr(underlying, "lr_lambdas", None)
    if functions is None:
        raise ValueError("DGG scheduler sidecar requires an HF LambdaLR scheduler")
    return {"scheduler_class": type(underlying).__module__ + "." + type(underlying).__qualname__,
            "type": kind, "max_steps": int(state.max_steps),
            "warmup_steps": int(trainer.args.get_warmup_steps(state.max_steps)),
            "functions": [_function_signature(fn) for fn in functions]}


def capture(trainer, state):
    engine, scheduler, underlying, groups = _scheduler_and_groups(trainer)
    return {"format": FORMAT, "version": VERSION, "trainer_step": int(state.global_step),
            "deepspeed_step": int(engine.global_steps), "signature": _signature(trainer, state, underlying),
            "optimizer_groups": groups, "scheduler_state": _canonical(scheduler.state_dict())}


def restore(trainer, state, payload):
    """Restore only after DeepSpeed loaded the checkpoint and its optimizer."""
    if not isinstance(payload, dict) or payload.get("format") != FORMAT or payload.get("version") != VERSION:
        raise ValueError("DGG scheduler sidecar has an unsupported format or version")
    current = capture(trainer, state)
    for key in ("trainer_step", "deepspeed_step", "signature", "optimizer_groups"):
        if payload.get(key) != current[key]:
            raise ValueError(f"DGG scheduler sidecar does not match restored {key}")
    saved = payload.get("scheduler_state")
    if not isinstance(saved, dict):
        raise ValueError("DGG scheduler sidecar is missing its scheduler state")
    # load_state_dict may mutate its argument; round-trip through JSON first.
    _, scheduler, _, _ = _scheduler_and_groups(trainer)
    scheduler.load_state_dict(_canonical(saved))
    after = capture(trainer, state)
    if after["scheduler_state"] != saved or after["optimizer_groups"] != current["optimizer_groups"]:
        raise ValueError("DGG scheduler restoration changed optimizer groups or did not round-trip")


class DGGSchedulerCheckpoint(TrainerCallback):
    """Save/load a DGG scheduler sidecar alongside each native checkpoint."""

    def __init__(self, trainer, resume_from_checkpoint=None):
        self.trainer = trainer
        self.resume_from_checkpoint = resume_from_checkpoint

    def sidecar_path(self, checkpoint):
        return Path(checkpoint) / SIDECAR

    def on_train_begin(self, args, state, control, **kwargs):
        if self.resume_from_checkpoint:
            path = self.sidecar_path(self.resume_from_checkpoint)
            if not path.is_file():
                raise FileNotFoundError(f"DGG checkpoint is missing its scheduler sidecar: {path}")
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(f"DGG scheduler sidecar cannot be read: {path}") from exc
            restore(self.trainer, state, payload)
        return control

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        output = Path(self.trainer._get_output_dir(getattr(self.trainer, "_trial", None)))
        checkpoint = output / f"checkpoint-{state.global_step}"
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"DGG native checkpoint was not saved: {checkpoint}")
        target = self.sidecar_path(checkpoint)
        body = json.dumps(capture(self.trainer, state), sort_keys=True, indent=2, allow_nan=False) + "\n"
        fd, temporary = tempfile.mkstemp(prefix=".dgg-scheduler-", dir=str(checkpoint))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                file.write(body)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return control
