# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Short-lived expert weight cache for a single inference request.

The cache is deliberately not a module buffer and is discarded at the end of
each generate call. Expert weights must not be mutated concurrently during a
request; callers that deliberately mutate them must clear this scope first.
"""
from __future__ import annotations

import functools
import inspect
import weakref
from contextlib import contextmanager
from contextvars import ContextVar

import torch


class _RequestCache:
    def __init__(self):
        self.weights = {}
        self.adapters = {}

    def clear(self):
        self.weights.clear()
        self.adapters.clear()


_ACTIVE = ContextVar('mcitlib_expert_inference_cache', default=None)


@contextmanager
def expert_inference_scope():
    """Cache weights only until this request exits, including on exceptions."""
    parent = _ACTIVE.get()
    current = _RequestCache()
    token = _ACTIVE.set(current)
    try:
        yield
    finally:
        current.clear()
        _ACTIVE.reset(token)
        if parent is not None:
            parent.clear()


def clear_expert_inference_cache():
    """Invalidate the active request after an intentional in-request mutation."""
    current = _ACTIVE.get()
    if current is not None:
        current.clear()


def observe_active_adapter(layer):
    current = _ACTIVE.get()
    if current is None:
        return
    active = layer.active_adapter
    old = current.adapters.get(layer, active)
    if old != active:
        current.clear()
    current.adapters[layer] = active


class InferenceBankInvalidation:
    """Invalidate an active request on explicit mode, device or load changes."""

    def train(self, mode=True):
        clear_expert_inference_cache()
        return super().train(mode)

    def _apply(self, fn):
        clear_expert_inference_cache()
        return super()._apply(fn)

    def _load_from_state_dict(self, *args, **kwargs):
        clear_expert_inference_cache()
        return super()._load_from_state_dict(*args, **kwargs)


def _fingerprint(tensor):
    try:
        version = tensor._version
        pointer = tensor.data_ptr()
    except RuntimeError:
        # Inference tensors do not expose version counters. Recompute rather
        # than risk reusing a weight after an undetectable mutation.
        return None
    return (id(tensor), pointer, version, tensor.dtype, tensor.device,
            tuple(tensor.shape), tuple(tensor.stride()))


def expert_weight(bank, parameters, x, *, extra, build):
    """Build or reuse one scoped, versioned weight; never cache training work."""
    current = _ACTIVE.get()
    if current is None:
        return build()
    if bank.training or torch.is_grad_enabled():
        current.clear()
        return build()
    fingerprints = tuple(_fingerprint(item) for item in parameters)
    if any(item is None for item in fingerprints):
        current.weights.pop(bank, None)
        return build()
    key = (extra, x.device, x.dtype, torch.get_default_dtype(), fingerprints)
    existing = current.weights.get(bank)
    if existing is None or existing[0] != key:
        value = build()
        current.weights[bank] = (key, value)
        return value
    return existing[1]


def install_generation_scope(model):
    """Bound the cache to each actual model.generate call, without a module hook."""
    original = model.generate
    if getattr(original, '_mcit_expert_inference_scope', False):
        return model
    bound_function = getattr(original, '__func__', None)
    if getattr(original, '__self__', None) is model and bound_function is not None:
        model_ref = weakref.ref(model)

        @functools.wraps(bound_function)
        def generate(*args, **kwargs):
            instance = model_ref()
            if instance is None:
                raise RuntimeError('generation model is no longer alive')
            with expert_inference_scope():
                return bound_function(instance, *args, **kwargs)
    else:
        @functools.wraps(original)
        def generate(*args, **kwargs):
            with expert_inference_scope():
                return original(*args, **kwargs)
    generate.__signature__ = inspect.signature(original)
    generate._mcit_expert_inference_scope = True
    model.generate = generate
    return model
