# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Compatibility facade for shared CLIP tower builders."""

from mcitlib.backbones.llava.text_tower import build_text_tower, build_vision_tower

__all__ = ["build_text_tower", "build_vision_tower"]
