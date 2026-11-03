# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""LLaVA inference helpers; CoIN runners are migrated separately."""
from __future__ import annotations

from typing import Any


def generate(model: Any, batch: dict[str, Any], generation_config: dict[str, Any]) -> Any:
    """Delegate generation without imposing a new decoding protocol."""
    return model.generate(**batch, **generation_config)
