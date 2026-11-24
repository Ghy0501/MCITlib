# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Scope non-reentrant checkpointing to a single rank's training call."""
from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch.utils.checkpoint as checkpoint_module


@dataclass
class CheckpointObservation:
    calls: int = 0


@contextmanager
def octopus_checkpoint_scope(method: str | None, stage: str | None, args: Any) -> Iterator[CheckpointObservation | None]:
    """Apply the released distributed policy to Octopus stages one and two."""
    if method != "octopus" or stage not in {"1", "2"}:
        yield None
        return
    with non_reentrant_checkpoint_scope(args, label="Octopus") as observation:
        yield observation


@contextmanager
def olora_checkpoint_scope(method: Any, args: Any) -> Iterator[CheckpointObservation | None]:
    """Keep Trainer checkpointing enabled while forcing non-reentrant torch calls.

    Transformers 4.31 calls ``gradient_checkpointing_enable()`` without kwargs
    and LlamaModel calls ``torch.utils.checkpoint.checkpoint`` directly. A
    training-arguments attribute alone therefore cannot select the algorithm.
    Each launcher rank is a separate process; this replacement is limited to
    the OLoRA trainer call and is restored even when training fails.
    """
    if getattr(method, "id", None) not in {"olora", "same"} or not args.gradient_checkpointing:
        yield None
        return

    overrides = method.training_overrides()
    expected = {"use_reentrant": False}
    if overrides != {"gradient_checkpointing_kwargs": expected}:
        raise RuntimeError("OLoRA requires an explicit non-reentrant checkpoint override")
    with non_reentrant_checkpoint_scope(args, label="OLoRA") as observation:
        yield observation


@contextmanager
def non_reentrant_checkpoint_scope(args: Any, *, label: str) -> Iterator[CheckpointObservation | None]:
    """Apply the released distributed checkpoint policy without disabling it.

    The pinned Transformers 4.31 Llama implementation calls torch.checkpoint
    directly. Keep this scope per rank, restore the original even on failure,
    and reject conflicting call-site policies.
    """
    if not args.gradient_checkpointing:
        yield None
        return
    expected = {"use_reentrant": False}
    existing = getattr(args, "gradient_checkpointing_kwargs", None)
    if existing is not None and existing != expected:
        raise RuntimeError(f"conflicting gradient_checkpointing_kwargs for {label}")
    args.gradient_checkpointing_kwargs = expected.copy()

    original = checkpoint_module.checkpoint
    if getattr(original, "_mcitlib_olora_checkpoint", False):
        raise RuntimeError(f"{label} checkpoint scope is already active")
    if "use_reentrant" not in inspect.signature(original).parameters:
        raise RuntimeError("installed torch checkpoint does not support use_reentrant")
    observation = CheckpointObservation()

    def non_reentrant(function: Any, *values: Any, **kwargs: Any) -> Any:
        if kwargs.get("use_reentrant", False) is not False:
            raise RuntimeError(f"{label} encountered a conflicting reentrant checkpoint call")
        kwargs["use_reentrant"] = False
        observation.calls += 1
        return original(function, *values, **kwargs)

    non_reentrant._mcitlib_olora_checkpoint = True
    checkpoint_module.checkpoint = non_reentrant
    try:
        yield observation
        if checkpoint_module.checkpoint is not non_reentrant:
            raise RuntimeError(f"{label} checkpoint implementation changed during training")
        if observation.calls == 0:
            raise RuntimeError(f"{label} training did not invoke the non-reentrant checkpoint path")
    finally:
        checkpoint_module.checkpoint = original
