# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Resolve only path fields in the JSON files used by KeepLoRA extra steps."""
from __future__ import annotations

import os
from typing import Any

from ...paths import PathResolver


def resolve_fields(values: dict[str, Any], fields: tuple[str, ...], context: str | None = None) -> dict[str, Any]:
    context = context if context is not None else os.environ.get("MCITLIB_PATH_CONTEXT")
    if not context:
        return values
    paths, benchmark = PathResolver.from_context(context)
    resolved = dict(values)
    for field in fields:
        if isinstance(resolved.get(field), str):
            resolved[field] = paths.resolve_templates(resolved[field], benchmark)
    return resolved
