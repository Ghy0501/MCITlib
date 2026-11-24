# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Scoped compatibility experiment for DGG's DeepSpeed callback.

This has CPU ordering tests only. It is deliberately not installed by a V4
training entry until the ZeRO-2 bf16 runtime and resumption have GPU evidence.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable


@contextmanager
def optimizer_backward_hook(engine, hook: Callable):
    """Call hook after optimizer.backward and before engine reduction.

    DeepSpeed 0.9.5's ZeRO branch invokes optimizer.backward directly, then
    performs its reduce epilogue. The released DGG fork runs a callback at that
    boundary. Scope the replacement to this engine/optimizer instance.
    """
    optimizer = engine.optimizer
    if optimizer is None or not callable(getattr(optimizer, "backward", None)):
        raise RuntimeError("DGG requires an initialized DeepSpeed optimizer with backward")
    if getattr(optimizer, "_mcit_dgg_hook_active", False):
        raise RuntimeError("DGG hook already installed")
    had_instance_method = "backward" in vars(optimizer)
    old_instance_method = vars(optimizer).get("backward")
    original = optimizer.backward

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        hook(engine)
        return result

    optimizer._mcit_dgg_hook_active = True
    optimizer.backward = wrapped
    try:
        yield
    finally:
        if had_instance_method:
            optimizer.backward = old_instance_method
        else:
            del optimizer.backward
        del optimizer._mcit_dgg_hook_active
