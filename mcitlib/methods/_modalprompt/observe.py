# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Optional bounded-run evidence for ModalPrompt's training calls."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mcitlib.paths import managed_path, repo_root


def enabled() -> bool:
    return bool(os.environ.get("MCITLIB_MODAL_OBSERVE_DIR"))


def write_event(task: int, step: int, kind: str, **values: Any) -> None:
    root = os.environ.get("MCITLIB_MODAL_OBSERVE_DIR")
    if not root:
        return
    if not Path(root).is_absolute():
        raise ValueError("ModalPrompt observation directory must be absolute")
    project = repo_root()
    path = managed_path(root, project, within=project / "logs", label="ModalPrompt observations")
    rank = int(os.environ.get("RANK", "0"))
    if task < 1 or step < 0 or rank < 0:
        raise ValueError("invalid ModalPrompt observation identity")
    folder = path / f"task{task}"
    folder.mkdir(parents=True, exist_ok=True)
    record = {"event": kind, "task": task, "rank": rank, "step": step, **values}
    with (folder / f"rank{rank}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=True, sort_keys=True) + "\n")


def optimizer_groups(model: Any, master: Any) -> list[dict[str, Any]]:
    """Describe the actual ZeRO subgroups, not the unused trainer groups."""
    names = {id(param): name for name, param in model.named_parameters()}
    groups = [{"group": index, "lr": float(item["lr"]), "parameters": []}
              for index, item in enumerate(master.optimizer.param_groups)]
    for subgroup, parameters in enumerate(master.fp16_groups):
        group = master.sub_group_to_group_id[subgroup]
        for param in parameters:
            name = names.get(id(param))
            if name and any(key in name for key in ("embed_tokens", "prompt_transform", "mm_projector")):
                groups[group]["parameters"].append(name)
    for group in groups:
        group["parameters"].sort()
    return groups
