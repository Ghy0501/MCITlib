# SPDX-License-Identifier: Apache-2.0
"""Recognize and resolve an InternViT tower without importing model code."""
from __future__ import annotations

import json
import os
from pathlib import Path


def vision_name(config):
    name = getattr(config, 'mm_vision_tower', getattr(config, 'vision_tower', None))
    if os.environ.get('MCITLIB_BACKBONE') == 'internvl' and is_intern_vit(name):
        name = os.environ.get('MCITLIB_INTERNVL_VISION_TOWER') or name
    return name


def is_intern_vit(name):
    if not isinstance(name, (str, os.PathLike)):
        return False
    path = Path(name)
    config = path / 'config.json'
    if config.is_file():
        try:
            value = json.loads(config.read_text(encoding='utf-8'))
            return isinstance(value, dict) and value.get('model_type') == 'intern_vit_6b'
        except (OSError, ValueError):
            return False
    # Only the leaf model identifier is meaningful: a parent directory may
    # happen to contain "internvl" while holding a completely different tower.
    leaf = path.name.lower().replace('-', '').replace('_', '')
    return leaf.startswith('internvit') or leaf.startswith('internvision')
