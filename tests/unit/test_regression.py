# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CPU tests for GPU regression log comparison and dry-run preparation."""
from __future__ import annotations

from pathlib import Path

import pytest

from mcitlib.registry import METHODS
from tools.regression.compare import compare_logs, run_combo


def test_compare_logs_relative_tolerance(tmp_path: Path) -> None:
    v3 = tmp_path / "v3.log"
    v4 = tmp_path / "v4.log"
    v3.write_text("{'loss': 1.0}\n{'loss': 0.5}\n", encoding="utf-8")
    v4.write_text("{'loss': 1.0005}\n{'loss': 0.5001}\n", encoding="utf-8")
    result = compare_logs(v3, v4, 1e-3)
    assert result["passed"]
    assert result["count"] == 2


def test_regression_combo_is_dry_run_by_default(repo_root: Path, tmp_path: Path) -> None:
    result = run_combo(repo_root, repo_root.parent / "MCITlib_v3_ref", "lora_ft", "ucit", 1, tmp_path, False, 20)
    assert result["status"] == "dry-run"
    assert not (tmp_path / "v3.log").exists()


@pytest.mark.parametrize("method", METHODS, ids=lambda item: item.id)
def test_regression_uses_registered_model_override(repo_root: Path, tmp_path: Path, method) -> None:
    from unittest.mock import patch

    environments = []

    def fake_process(argv, cwd, env, log, max_steps):
        environments.append(env)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("{'loss': 1.0}\n", encoding="utf-8")
        return 0

    with patch("tools.regression.compare.run_process", side_effect=fake_process):
        result = run_combo(
            repo_root, repo_root.parent / "MCITlib_v3_ref", method.id, "ucit", 1,
            tmp_path, True, 1,
        )
    assert result["passed"]
    assert len(environments) == 2
    assert environments[1].get("MCITLIB_MODEL_OVERRIDE") == method.model_override
