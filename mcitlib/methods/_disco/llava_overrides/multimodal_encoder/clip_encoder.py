# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Compatibility facade for the shared CLIP text/feature capability."""

from mcitlib.backbones.llava.text_tower import CLIPTextTower, CLIPVisionTower

__all__ = ["CLIPTextTower", "CLIPVisionTower"]
