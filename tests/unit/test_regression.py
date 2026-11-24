# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CPU tests for GPU regression log comparison and dry-run preparation."""
from __future__ import annotations

from pathlib import Path

import pytest

from mcitlib.registry import METHODS
from tools.regression.compare import ProcessOutcome, compare_logs, run_combo


def test_compare_logs_relative_tolerance(tmp_path: Path) -> None:
    v3 = tmp_path / "v3.log"
    v4 = tmp_path / "v4.log"
    v3.write_text("{'loss': 1.0}\n{'loss': 0.5}\n", encoding="utf-8")
    v4.write_text("{'loss': 1.0005}\n{'loss': 0.5001}\n", encoding="utf-8")
    result = compare_logs(v3, v4, 1e-3)
    assert result["passed"]
    assert result["count"] == 2


def test_regression_combo_is_dry_run_by_default(repo_root: Path, tmp_path: Path) -> None:
    result = run_combo(repo_root, tmp_path / "v3-reference", "lora_ft", "ucit", 1, tmp_path, False, 20)
    assert result["status"] == "dry-run"
    assert not (tmp_path / "v3.log").exists()


_NATIVE_METHODS = frozenset({"pclr", "same", "sigma", "dgg", "octopus"})
_V3_METHODS = [method for method in METHODS if method.id not in _NATIVE_METHODS]
assert len(_V3_METHODS) == 12


@pytest.mark.parametrize("method", _V3_METHODS, ids=lambda item: item.id)
def test_regression_uses_registered_model_override(repo_root: Path, tmp_path: Path, method) -> None:
    from unittest.mock import patch

    environments = []

    def fake_process(argv, cwd, env, log, max_steps):
        environments.append(env)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("{'loss': 1.0}\n", encoding="utf-8")
        return ProcessOutcome(0)

    with patch("tools.regression.compare.run_process", side_effect=fake_process):
        result = run_combo(
            repo_root, tmp_path / "v3-reference", method.id, "ucit", 1,
            tmp_path, True, 1,
        )
    assert result["passed"]
    assert len(environments) == 2
    assert environments[1].get("MCITLIB_MODEL_OVERRIDE") == method.model_override


@pytest.mark.parametrize("method", sorted(_NATIVE_METHODS))
def test_native_method_reports_missing_v3_reference(repo_root: Path, tmp_path: Path, method) -> None:
    with pytest.raises(ValueError, match="no V3 comparison fixture"):
        run_combo(repo_root, tmp_path / "v3-reference", method, "ucit", 1, tmp_path, True, 1)


def test_task2_uses_cached_shared_previous_and_isolated_outputs(repo_root: Path, tmp_path: Path) -> None:
    from unittest.mock import patch

    calls = []

    def fake_process(argv, cwd, env, log, max_steps):
        calls.append((argv, max_steps))
        output = Path(argv[argv.index("--output_dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        if max_steps is not None:
            previous = Path(argv[argv.index("--previous_task_model_path") + 1])
            assert previous.is_dir()
        count = max_steps or 1
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("".join(f"loss: {1.0 / (index + 1)}\n" for index in range(count)), encoding="utf-8")
        return ProcessOutcome(0)

    with patch("tools.regression.compare.run_process", side_effect=fake_process):
        first = run_combo(repo_root, tmp_path / "v3-reference", "lora_ft", "ucit", 2, tmp_path, True, 2)
        second = run_combo(repo_root, tmp_path / "v3-reference", "lora_ft", "ucit", 2, tmp_path, True, 2)
    assert first["passed"] and second["passed"]
    assert len(calls) == 5  # one cached full task1 preparation + two short runs per invocation
    short_calls = [argv for argv, limit in calls if limit == 2]
    previous = [Path(argv[argv.index("--previous_task_model_path") + 1]) for argv in short_calls]
    outputs = [Path(argv[argv.index("--output_dir") + 1]) for argv in short_calls]
    assert previous[0] == previous[1]
    assert "shared-checkpoints" in str(previous[0])
    assert outputs[0] != outputs[1]
    assert "v3-checkpoints" in str(outputs[0]) and "v4-checkpoints" in str(outputs[1])


def test_fake_training_early_v4_crash_fails(repo_root: Path, tmp_path: Path, monkeypatch) -> None:
    import os
    import sys

    binary = tmp_path / "bin"
    binary.mkdir()
    fake = binary / "deepspeed"
    fake.write_text(f"""#!{sys.executable}
import os
from pathlib import Path
import sys
argv = sys.argv[1:]
Path(argv[argv.index('--output_dir') + 1]).mkdir(parents=True, exist_ok=True)
count = 3 if 'MCITLIB_TRAIN_ENTRY' in os.environ else 20
for index in range(count):
    print(f'loss: {{1.0 / (index + 1)}}', flush=True)
sys.exit(1 if count == 3 else 0)
""", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    v3_reference = tmp_path / "v3-reference"
    (v3_reference / "LLaVA/LoRA-FT").mkdir(parents=True)
    result = run_combo(repo_root, v3_reference, "lora_ft", "ucit", 1, tmp_path, True, 20)
    assert not result["passed"]
    assert result["v3_count"] == 20 and result["v4_count"] == 3
    assert result["v4_exit"] == 1
    assert not result["counts_match"] and not result["exits_acceptable"]


def test_sefe_task2_prepares_merge_and_key_elements_once(repo_root: Path, tmp_path: Path) -> None:
    from unittest.mock import patch

    calls = []

    def fake_process(argv, cwd, env, log, max_steps):
        calls.append((argv, max_steps))
        if "--output_dir" in argv:
            Path(argv[argv.index("--output_dir") + 1]).mkdir(parents=True, exist_ok=True)
        if "--save-model-path" in argv:
            Path(argv[argv.index("--save-model-path") + 1]).mkdir(parents=True, exist_ok=True)
        if any(value.endswith("key_elements_identification.py") for value in argv):
            Path(argv[-2], "Task1_llava_lora_topP.pth").touch()
        if max_steps is not None:
            assert Path(argv[argv.index("--model_name_or_path") + 1]).is_dir()
            assert Path(argv[argv.index("--regularization_info_path") + 1]).is_file()
        count = max_steps or 1
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("".join(f"loss: {1.0 / (index + 1)}\n" for index in range(count)), encoding="utf-8")
        return ProcessOutcome(0)

    with patch("tools.regression.compare.run_process", side_effect=fake_process):
        result = run_combo(repo_root, tmp_path / "v3-reference", "sefe", "ucit", 2, tmp_path, True, 2)
    assert result["passed"]
    preparation = [argv for argv, limit in calls if limit is None]
    assert len(preparation) == 3
    assert preparation[0][0] == "deepspeed"
    assert any("merge_lora_weights.py" in value for argv in preparation for value in argv)
    assert any("key_elements_identification.py" in value for argv in preparation for value in argv)
    short = [argv for argv, limit in calls if limit == 2]
    assert len(short) == 2
    assert short[0][short[0].index("--regularization_info_path") + 1] == short[1][short[1].index("--regularization_info_path") + 1]
