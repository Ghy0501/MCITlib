# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Construct deepspeed argv without changing v3 training arguments."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ...config.schema import Command, RunConfig, TaskSpec
from ...paths import vendored_pythonpath

ENTRY_MODULE = "mcitlib.backbones.llava.train_entry"
# deepspeed treats the first positional argument as the user script. Running a
# module therefore needs the launcher flag ``--module`` *before* the module name;
# "python -m X" would make every rank execute ``python -u python -m X``.
LAUNCHER_ENTRY = ["--module", ENTRY_MODULE]


def _resolve(value: Any, run_cfg: RunConfig) -> Any:
    paths = run_cfg.paths
    if isinstance(value, list):
        return [_resolve(item, run_cfg) for item in value]
    if not isinstance(value, str) or paths is None:
        return value
    replacements = {
        "${model_root}": str(paths.paths.model_root),
        "${data_root}": str(paths.paths.data_root),
        "${ckpt_root}": str(paths.paths.ckpt_root),
        "${repo_root}": str(paths.paths.repo_root),
    }
    for source, target in replacements.items():
        if value.startswith(source):
            return target + value[len(source):]
    return value


def build_train_command(
    run_cfg: RunConfig, task: TaskSpec, entry: str,
    model_override: str | None = None, method: Any = None,
) -> Command:
    try:
        original = list(run_cfg.values["train_command"]["argv"])
    except (KeyError, TypeError) as error:
        raise ValueError("merged task config lacks train_command.argv") from error
    argv = list(original)
    if method is not None and hasattr(method, "select_train_data"):
        source = method.select_train_data(task)
        argv[argv.index("--data_path") + 1] = source.train_path
        argv[argv.index("--image_folder") + 1] = source.train_folder
    entry_index = next((i for i, value in enumerate(argv) if value.endswith(".py")), None)
    if entry_index is None:
        raise ValueError(f"v3 training entry not found in argv: {original}")
    argv = argv[:entry_index] + LAUNCHER_ENTRY + argv[entry_index + 1:]
    argv = _resolve(argv, run_cfg)
    repo_root = Path(run_cfg.values.get("repo_root", ".")).resolve()
    config_name = run_cfg.values.get("deepspeed_config", "zero2")
    config_path = run_cfg.values["deepspeed_configs"][config_name]
    deepspeed_index = argv.index("--deepspeed") + 1
    argv[deepspeed_index] = str((repo_root / config_path).resolve())
    pythonpath = os.pathsep.join(vendored_pythonpath(repo_root))
    env = {"MCITLIB_TRAIN_ENTRY": entry, "MCITLIB_METHOD": run_cfg.method, "PYTHONPATH": pythonpath}
    if model_override:
        env["MCITLIB_MODEL_OVERRIDE"] = model_override
    return Command(
        argv=argv,
        cwd=str(repo_root),
        env=env,
        kind="train",
        task_id=task.task_id,
        metadata={"source_entry": original[entry_index], "entry": entry},
    )
