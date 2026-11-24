# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Layered YAML configuration loading."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@lru_cache(maxsize=None)
def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"configuration must contain a mapping: {path}")
    if value.get("format") == "mcitlib-method-v1":
        from .compact import expand_method

        value = expand_method(path.parent.parent, value)
    return value


def load_layered_config(
    config_root: str | Path,
    backbone: str,
    benchmark: str,
    method: str,
    task_id: int | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge defaults < backbone < benchmark < method < task < CLI."""
    root = Path(config_root)
    if backbone == "internvl":
        from ..backbones.internvl.config import load_internvl_config

        return load_internvl_config(root, benchmark, method, task_id, cli_overrides)
    result: dict[str, Any] = {}
    for path in (
        root / "defaults.yaml",
        root / "backbones" / f"{backbone}.yaml",
        root / "benchmarks" / f"{benchmark}.yaml",
        root / "methods" / f"{method}.yaml",
    ):
        result = deep_merge(result, read_yaml(path))

    method_values = read_yaml(root / "methods" / f"{method}.yaml")
    overrides = method_values.get("overrides", {})
    if task_id is not None:
        task_values = (
            overrides.get(backbone, {})
            .get(benchmark, {})
            .get(f"task_{task_id}", {})
        )
        result = deep_merge(result, task_values)
    if cli_overrides:
        result = deep_merge(result, cli_overrides)
    return result


def parse_cli_value(value: str) -> Any:
    """Parse a CLI override using YAML scalar/list/dict syntax."""
    return yaml.safe_load(value)
