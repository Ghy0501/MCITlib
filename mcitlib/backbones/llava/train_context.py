# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Resolve a one-based training context without changing method arguments."""
from __future__ import annotations

from pathlib import Path


def training_task_id(method: str, cur_task: int, task_label: str, *, config_root: Path | None = None) -> int:
    """Map CL-MoE's historical label to its task; other methods use zero-based cur_task.

    CL-MoE's ``--task`` is a string such as ``UCIT_ImgNet``. Its numeric
    context is for diagnostics, while the method argument remains untouched.
    Resolve labels using the same method configuration that declares them,
    and preserve previously accepted numeric labels for direct entry callers.
    """
    if method != "clmoe" or not task_label:
        return int(cur_task) + 1
    try:
        return int(task_label)
    except ValueError:
        pass

    from mcitlib.config.loader import read_yaml
    from mcitlib.paths import repo_root

    root = config_root if config_root is not None else repo_root() / "configs"
    config = read_yaml(Path(root) / "methods" / "clmoe.yaml")
    matches = []
    for tasks in config.get("overrides", {}).get("llava", {}).values():
        for key, value in tasks.items():
            if key.startswith("task_") and value.get("train_config", {}).get("task") == task_label:
                matches.append(int(key.removeprefix("task_")))
    if len(matches) != 1:
        raise ValueError(f"CL-MoE task label {task_label!r} has {len(matches)} matches in {root}/methods/clmoe.yaml")
    return matches[0]
