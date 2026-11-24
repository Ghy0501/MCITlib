# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""G2/G4/G5: every distinct entry point of every sequence starts in a *clean* process.

Each entry point runs in a fresh interpreter with exactly the environment the runner
would build (no ambient PYTHONPATH) and only imports what the real run imports.  The
only test double is a flash-attn stand-in that cannot execute kernels.

Needs a full LLaVA environment from envs/.  In CI set
MCITLIB_REQUIRE_FULL_ENV=1 so that a missing package fails instead of skipping.
"""
from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pytest

from mcitlib.registry import BENCHMARKS, METHODS, benchmark_supported, get_benchmark
from mcitlib.runner import build_sequence_commands
from mcitlib.selfcheck import distinct_entry_points, run_selfcheck
from tests.conftest import require_module

ROOT = Path(__file__).resolve().parents[2]
FAKE_FLASH = ROOT / "tests" / "fixtures" / "fake_flash_attn"


def _active_environment() -> str | None:
    try:
        import transformers
    except ImportError:
        return None
    if transformers.__version__.startswith("4.31"):
        try:
            deepspeed_version = importlib.metadata.version("deepspeed").split("+")[0]
        except importlib.metadata.PackageNotFoundError:
            deepspeed_version = None
        if deepspeed_version == "0.12.6":
            return "llava-modalprompt-ds126"
        return "llava"
    if transformers.__version__.startswith("4.37"):
        return "llava-t437"
    return None


def _collect() -> list[tuple[str, object, object]]:
    environment = _active_environment()
    found: dict[object, object] = {}
    for method in METHODS:
        if method.env != environment:
            continue
        sequences = [(benchmark.id, "train") for benchmark in BENCHMARKS
                     if benchmark.id != "general" and benchmark_supported(method.id, benchmark.id)]
        if method.id == "mrlora":
            sequences += [(benchmark.id, "router") for benchmark in BENCHMARKS if benchmark.id != "general"]
        for benchmark_id, sequence in sequences:
            commands = build_sequence_commands(ROOT / "configs", "llava", method.id, benchmark_id, sequence=sequence)
            if method.separate_train_and_infer and sequence == "train":
                commands += build_sequence_commands(ROOT / "configs", "llava", method.id, benchmark_id,
                                                    task_id=len(get_benchmark(benchmark_id).tasks), only_eval=True)
            found.update({k: v for k, v in distinct_entry_points(commands).items() if k not in found})
        if benchmark_supported(method.id, "general"):
            general = build_sequence_commands(ROOT / "configs", "llava", method.id, "general", only_eval=True)
            found.update({k: v for k, v in distinct_entry_points(general).items() if k not in found})
    return [(entry.label(), entry, command) for entry, command in found.items()]


CASES = _collect()


def test_collected_entry_points_cover_every_kind() -> None:
    if not CASES:
        pytest.skip("transformers 4.31 or 4.37 required")
    kinds = {entry.kind for _, entry, _ in CASES}
    methods = {entry.method for _, entry, _ in CASES}
    if _active_environment() == "llava-modalprompt-ds126":
        assert methods == {"modalprompt"} and {"train", "eval"} <= kinds
    else:
        assert {"train", "eval", "module"} <= kinds
        assert len(methods) >= 8 or _active_environment() == "llava-t437"


@pytest.mark.entrypoints
@pytest.mark.parametrize("label,entry,command", CASES, ids=[case[0] for case in CASES])
def test_entry_point_starts_in_a_clean_process(label: str, entry, command) -> None:
    require_module("deepspeed")
    require_module("accelerate")
    result = run_selfcheck(entry, command, ROOT, extra_pythonpath=(str(FAKE_FLASH),))
    assert result.ok, f"{label}: exit {result.returncode}: {result.message}"


@pytest.mark.entrypoints
@pytest.mark.parametrize("method", ["moelora", "mrlora", "olora", "clmoe", "smolora"])
def test_evaluation_entry_registers_the_methods_adapter_type(method: str) -> None:
    """G8: the evaluation entry (not only the test harness) must install the method's tuner."""
    import json
    import subprocess
    import sys

    from mcitlib.paths import process_env

    require_module("accelerate")
    env = process_env({"MCITLIB_METHOD": method, "MCITLIB_SELFCHECK": "1"}, ROOT)
    result = subprocess.run(
        [sys.executable, "-m", "mcitlib.backbones.llava.eval_runners.entry",
         "mcitlib.backbones.llava.eval_runners.coin.eval_deepseek_r1"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-400:]
    line = [row for row in result.stdout.splitlines() if row.startswith("{")][-1]
    assert json.loads(line)["adapter_installed"] is True
