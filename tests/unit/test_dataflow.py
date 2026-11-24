# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""G10: every artefact a step reads is written by an earlier step (or is user input).

Catches checkpoint chains that do not close, scorers that read files nobody produced and
similar v3 integration gaps *before* they cost GPU time.  Known v3 gaps are listed
explicitly so that a new one cannot appear unnoticed (each has an entry in drift.md).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mcitlib.dataflow import find_gaps, produced_paths
from mcitlib.paths import PathResolver
from mcitlib.registry import BENCHMARKS, METHODS, get_benchmark, get_method
from mcitlib.runner import build_sequence_commands

# v3 defects kept as-is (drift.md): method, benchmark, sequence -> (flag, expected count).
KNOWN_V3_GAPS = {
    # G10a: the MLLM-DCL evaluation scripts of MR-LoRA have the inference commented out; they
    # only merge and score chunk files that nothing in the sequence writes (25 datasets x tasks).
    ("mrlora", "mllm_dcl", "train"): {"--discover": 25},
    # G10b: ModalPrompt x UCIT task 2 writes Task1_llava (v3 config typo, AR-5 pending). The 4
    # evaluation commands of task 2 (2 datasets x 2 chunks) read Task2_llava, which nothing writes.
    ("modalprompt", "ucit", "train"): {"--model-path": 4},
}


@pytest.fixture(scope="module")
def paths(tmp_path_factory) -> PathResolver:
    root = tmp_path_factory.mktemp("roots")
    config = root / "paths.yaml"
    config.write_text(f"model_root: {root}/models\ndata_root: {root}/data\nckpt_root: {root}/ckpts\n")
    return PathResolver.from_file(config, repo_root=Path(__file__).resolve().parents[2])


def _gaps(repo_root: Path, paths: PathResolver, method: str, benchmark: str, sequence: str):
    roots = (str(paths.paths.model_root), str(paths.paths.data_root))
    commands = build_sequence_commands(repo_root / "configs", "llava", method, benchmark, paths=paths, sequence=sequence)
    if get_method(method).separate_train_and_infer:
        # ModalPrompt: `mcitlib train` only trains; `mcitlib eval` runs afterwards on its output.
        spec = get_benchmark(benchmark)
        for task_id in range(1, len(spec.tasks) + 1):
            commands += build_sequence_commands(
                repo_root / "configs", "llava", method, benchmark, paths=paths, task_id=task_id, only_eval=True,
            )
    before: tuple[str, ...] = ()
    if sequence == "router":
        ordinary = build_sequence_commands(repo_root / "configs", "llava", method, benchmark, paths=paths)
        before = tuple(produced_paths(ordinary))
    return find_gaps(commands, roots, before)


def _all(repo_root: Path):
    for method in METHODS:
        for benchmark in BENCHMARKS:
            if method.supported_benchmarks is not None and benchmark.id not in method.supported_benchmarks:
                continue
            if benchmark.id != "general":
                yield method.id, benchmark.id, "train"
    for benchmark in BENCHMARKS:
        if benchmark.id != "general":
            yield "mrlora", benchmark.id, "router"


def test_only_the_documented_v3_gaps_exist(repo_root: Path, paths: PathResolver) -> None:
    found: dict[tuple[str, str, str], dict[str, int]] = {}
    for method, benchmark, sequence in _all(repo_root):
        gaps = _gaps(repo_root, paths, method, benchmark, sequence)
        if gaps:
            counts: dict[str, int] = {}
            for gap in gaps:
                counts[gap.flag] = counts.get(gap.flag, 0) + 1
            found[(method, benchmark, sequence)] = counts
    assert found == KNOWN_V3_GAPS, "\n".join(f"{key}: {value}" for key, value in sorted(found.items()))


def test_ar5_gap_disappears_when_the_task2_output_dir_is_corrected(repo_root: Path, paths: PathResolver) -> None:
    """Documents what AR-5 fixes: with Task2_llava written, ModalPrompt x UCIT closes."""
    commands = build_sequence_commands(repo_root / "configs", "llava", "modalprompt", "ucit", paths=paths)
    for task_id in range(1, len(get_benchmark("ucit").tasks) + 1):
        commands += build_sequence_commands(
            repo_root / "configs", "llava", "modalprompt", "ucit", paths=paths, task_id=task_id, only_eval=True,
        )
    for command in commands:
        if command.kind == "train" and command.task_id == 2:
            index = command.argv.index("--output_dir") + 1
            assert command.argv[index].endswith("Task1_llava"), "v3 value changed; update AR-5 note"
            command.argv[index] = command.argv[index][: -len("Task1_llava")] + "Task2_llava"
    roots = (str(paths.paths.model_root), str(paths.paths.data_root))
    assert find_gaps(commands, roots) == []
