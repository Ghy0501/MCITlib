# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""U1/U2: GPU ids and ports are configurable; evaluation chunks run in parallel, one GPU each."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from mcitlib.config.schema import Command
from mcitlib.gpus import apply_gpu_assignment, parse_gpus
from mcitlib.registry import BENCHMARKS, METHODS
from mcitlib.runner import build_sequence_commands, execute_commands


def _train(commands):
    return [c for c in commands if c.kind == "train"]


def test_training_gpus_and_port_are_replaced_but_the_count_is_enforced(repo_root: Path) -> None:
    commands = build_sequence_commands(repo_root / "configs", "llava", "lora_ft", "ucit", gpus="4,6", master_port=29501)
    for command in _train(commands):
        assert command.argv[command.argv.index("--include") + 1] == "localhost:4,6"
        assert command.argv[command.argv.index("--master_port") + 1] == "29501"
    with pytest.raises(ValueError, match="trains on 2 GPUs"):
        build_sequence_commands(repo_root / "configs", "llava", "lora_ft", "ucit", gpus="0,1,2,3")
    with pytest.raises(ValueError, match="distinct"):
        parse_gpus("1,1")


def test_defaults_match_v3(repo_root: Path) -> None:
    for command in _train(build_sequence_commands(repo_root / "configs", "llava", "lora_ft", "ucit")):
        assert command.argv[command.argv.index("--include") + 1] == "localhost:0,1"
        assert command.argv[command.argv.index("--master_port") + 1] == "9001"


def test_every_evaluation_chunk_is_pinned_to_its_own_gpu(repo_root: Path) -> None:
    for method in METHODS:
        for benchmark in BENCHMARKS:
            if benchmark.id == "general" or (method.supported_benchmarks is not None and benchmark.id not in method.supported_benchmarks):
                continue
            seen: dict[str, set[str]] = {}
            options = {"task_id": 2, "only_eval": True} if method.id == "dgg" else {}
            for command in build_sequence_commands(repo_root / "configs", "llava", method.id, benchmark.id, gpus="2,3", **options):
                if "--chunk-idx" in command.argv:
                    index = int(command.argv[command.argv.index("--chunk-idx") + 1])
                    assert command.env["CUDA_VISIBLE_DEVICES"] == ["2", "3"][index], (method.id, benchmark.id)
                    seen.setdefault(command.metadata["parallel_group"], set()).add(command.env["CUDA_VISIBLE_DEVICES"])
            assert all(devices == {"2", "3"} for devices in seen.values())


def test_default_pinning_reproduces_v3_gpulist(repo_root: Path) -> None:
    chunk_commands = [c for c in build_sequence_commands(repo_root / "configs", "llava", "hide", "ucit") if "--chunk-idx" in c.argv]
    assert chunk_commands
    for command in chunk_commands:
        assert command.env["CUDA_VISIBLE_DEVICES"] == command.argv[command.argv.index("--chunk-idx") + 1]


def test_too_few_gpus_for_the_chunks_is_an_error() -> None:
    command = Command(argv=["python", "x", "--chunk-idx", "1", "--num-chunks", "2"], kind="eval", task_id=1)
    with pytest.raises(ValueError, match="one GPU each"):
        apply_gpu_assignment([command], "0", None)


def _sleeper(path: Path, seconds: float, name: str, group: str | None, code: int = 0) -> Command:
    script = (
        "import sys,time; from pathlib import Path; p=Path(sys.argv[1]);"
        f"p.joinpath('{name}.start').write_text(str(time.time())); time.sleep({seconds});"
        f"p.joinpath('{name}.end').write_text(str(time.time())); sys.exit({code})"
    )
    metadata = {"parallel_group": group} if group else {}
    return Command(argv=[sys.executable, "-c", script, str(path)], kind="eval", task_id=1, metadata=metadata)


def test_chunks_of_a_group_run_concurrently_and_the_next_step_waits(tmp_path: Path) -> None:
    out = tmp_path / "t"
    out.mkdir()
    commands = [
        _sleeper(out, 1.0, "c0", "g"), _sleeper(out, 1.0, "c1", "g"),
        _sleeper(out, 0.0, "after", None),
    ]
    started = time.time()
    results = execute_commands(commands, tmp_path / "logs/run", repo_root=tmp_path)
    elapsed = time.time() - started
    assert [item["status"] for item in results] == ["completed"] * 3
    assert elapsed < 1.9, f"chunks ran one after the other ({elapsed:.1f}s)"
    start = {name: float((out / f"{name}.start").read_text()) for name in ("c0", "c1", "after")}
    end = {name: float((out / f"{name}.end").read_text()) for name in ("c0", "c1")}
    assert abs(start["c0"] - start["c1"]) < 0.7, "chunks must start together"
    assert start["after"] >= max(end.values()) - 0.05, "the next step must wait for every chunk"


def test_one_failed_chunk_lets_the_others_finish_records_all_and_stops(tmp_path: Path) -> None:
    out = tmp_path / "t"
    out.mkdir()
    commands = [
        _sleeper(out, 0.3, "ok", "g"), _sleeper(out, 0.0, "bad", "g", code=3), _sleeper(out, 0.0, "never", None),
    ]
    with pytest.raises(RuntimeError, match="exit 3"):
        execute_commands(commands, tmp_path / "logs/run", repo_root=tmp_path)
    assert (out / "ok.end").exists(), "the healthy chunk must not be killed half-way"
    assert not (out / "never.start").exists()
    states = sorted(
        __import__("json").loads(path.read_text())["status"] for path in (tmp_path / "logs/run/.mcitlib/steps").glob("*.json")
    )
    assert states == ["completed", "failed"]
    results = execute_commands(
        [commands[0], _sleeper(out, 0.0, "bad", "g"), commands[2]], tmp_path / "logs/run", repo_root=tmp_path, resume=True,
    )
    assert [item["status"] for item in results][0] == "skipped"
