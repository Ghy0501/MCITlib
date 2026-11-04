# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""NB-6 in-memory text-tower and generation-config normalization."""
from __future__ import annotations

from typing import Any

TEXT_MODEL_OVERRIDES = frozenset({"hide", "disco", "modalprompt"})


def inject_text_tower_config(config: Any, text_tower: str | None = None, select_layer: int = -1) -> Any:
    """Inject the reference CLIP text settings without editing model files."""
    selected = text_tower or getattr(config, "mm_text_tower", None) or getattr(config, "mm_vision_tower", None)
    if selected is None:
        raise ValueError("a text_tower or mm_vision_tower path is required")
    config.mm_text_tower = str(selected)
    config.mm_text_select_layer = int(select_layer)
    return config


def normalize_generation_config(model: Any) -> Any:
    """Match the reference generation config by removing sampling defaults."""
    generation = getattr(model, "generation_config", None)
    if generation is not None:
        generation.__dict__.pop("temperature", None)
        generation.__dict__.pop("top_p", None)
    return model
