# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Opt-in ModalPrompt continuation uses the preceding task checkpoint."""
from __future__ import annotations

from pathlib import Path

import pytest

from mcitlib.config.schema import PathSet
from mcitlib.dataflow import find_gaps
from mcitlib.paths import PathResolver
from mcitlib.runner import build_sequence_commands


@pytest.mark.parametrize(("benchmark", "tasks", "profile"), [
    ("ucit", 6, "four_gpu_validation"),
    ("mllm_dcl", 5, "four_gpu_validation"),
    ("mllm_acl", 4, None),
])
def test_opt_in_previous_checkpoints_are_produced_in_order(
    repo_root: Path, tmp_path: Path, benchmark: str, tasks: int, profile: str | None,
) -> None:
    paths = PathResolver(PathSet(tmp_path / "models", tmp_path / "data", tmp_path / "checkpoints", tmp_path,
                                 tmp_path / "results", tmp_path / "logs"))
    common = {"paths": paths, "execution_profile": profile}
    if profile:
        common["gpus"] = "0,1,2,3"
    legacy = build_sequence_commands(repo_root / "configs", "llava", "modalprompt", benchmark, **common)
    fixed = build_sequence_commands(repo_root / "configs", "llava", "modalprompt", benchmark,
                                    validation_fixes=True, **common)
    assert len(legacy) == len(fixed) == tasks
    assert all("--previous_task_model_path" not in command.argv for command in legacy)
    assert "--previous_task_model_path" not in fixed[0].argv
    for index in range(1, tasks):
        current, previous = fixed[index], fixed[index - 1]
        assert current.argv[current.argv.index("--previous_task_model_path") + 1] == (
            previous.argv[previous.argv.index("--output_dir") + 1]
        )
        assert current.metadata["previous_task_fix"] == "modalprompt_load_previous_task"
    if benchmark == "ucit":
        assert fixed[1].metadata["validation_fix"] == "modalprompt_ucit_task2_output_dir"
    internal = (str(paths.paths.ckpt_root), str(paths.results_root))
    assert not find_gaps(fixed, paths.external_roots(benchmark), internal_roots=internal)
    gaps = find_gaps(fixed[1:], paths.external_roots(benchmark), internal_roots=internal)
    assert any(gap.task_id == 2 and gap.flag == "--previous_task_model_path" for gap in gaps)
