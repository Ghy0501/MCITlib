# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Install InternViT support without changing the vendored LLaVA source."""
from __future__ import annotations

from functools import wraps

from .detect import is_intern_vit, vision_name


def install_vision_factory():
    """Extend the live factory; continue to dispatch CLIP to its original code."""
    from mcitlib.backbones.llava.overrides import import_llava

    import_llava()
    import llava.model.llava_arch as architecture
    import llava.model.multimodal_encoder.builder as builder

    original = builder.build_vision_tower
    if getattr(original, '_mcit_internvl_vision_factory', False):
        architecture.build_vision_tower = original
        return

    @wraps(original)
    def build(config, **kwargs):
        if is_intern_vit(vision_name(config)):
            from .vision_tower import build_intern_vision_tower

            return build_intern_vision_tower(config, **kwargs)
        return original(config, **kwargs)

    build._mcit_internvl_vision_factory = True
    builder.build_vision_tower = build
    # The base architecture imported the factory before this hook was installed.
    architecture.build_vision_tower = build


def install_inference_builder():
    """Route InternVL checkpoint names through the existing LLaVA builder."""
    import llava.model.builder as builder

    original = builder.load_pretrained_model
    if getattr(original, '_mcit_internvl_inference_builder', False):
        return

    @wraps(original)
    def load(model_path, model_base, model_name, *args, **kwargs):
        if 'intern' in model_name.lower() and 'llava' not in model_name.lower():
            model_name = 'llava_' + model_name
        return original(model_path, model_base, model_name, *args, **kwargs)

    load._mcit_internvl_inference_builder = True
    builder.load_pretrained_model = load
