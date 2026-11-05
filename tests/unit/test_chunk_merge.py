# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""G3: chunked evaluation output is merged before scoring, exactly like v3's ``cat`` loop."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from mcitlib.eval.chunks import MERGE_MODULE, merge_chunks
from mcitlib.registry import BENCHMARKS, METHODS
from mcitlib.runner import build_sequence_commands


def _all_sequences(repo_root: Path):
    for method in METHODS:
        for benchmark in BENCHMARKS:
            if benchmark.id == "general":
                continue
            yield method.id, benchmark.id, build_sequence_commands(repo_root / "configs", "llava", method.id, benchmark.id)
    for benchmark in BENCHMARKS:
        if benchmark.id != "general":
            yield "mrlora", benchmark.id, build_sequence_commands(
                repo_root / "configs", "llava", "mrlora", benchmark.id, sequence="router"
            )


def _arg(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


def test_merge_is_byte_identical_to_the_v3_shell_loop(tmp_path: Path) -> None:
    chunks = []
    for index, payload in enumerate((b'{"a": 1}\n{"a": 2}\n', b"", b'{"b": "caf\xc3\xa9"}\n')):
        path = tmp_path / f"3_{index}.jsonl"
        path.write_bytes(payload)
        chunks.append(path)
    (tmp_path / "merge.jsonl").write_bytes(b"stale content from an earlier run\n")
    expected = tmp_path / "expected.jsonl"
    shell = f'> "{expected}"; for IDX in 0 1 2; do cat "{tmp_path}/3_${{IDX}}.jsonl" >> "{expected}"; done'
    subprocess.run(["bash", "-c", shell], check=True)
    merge_chunks(tmp_path / "merge.jsonl", chunks)
    assert (tmp_path / "merge.jsonl").read_bytes() == expected.read_bytes()


def test_merge_fails_when_a_chunk_is_missing(tmp_path: Path) -> None:
    (tmp_path / "2_0.jsonl").write_text("{}\n")
    with pytest.raises(FileNotFoundError, match="2_1.jsonl"):
        merge_chunks(tmp_path / "merge.jsonl", [tmp_path / "2_0.jsonl", tmp_path / "2_1.jsonl"])
    result = subprocess.run(
        [sys.executable, "-m", MERGE_MODULE, "merge", "--output", str(tmp_path / "m.jsonl"),
         "--inputs", str(tmp_path / "2_0.jsonl"), str(tmp_path / "2_1.jsonl")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2 and "2_1.jsonl" in result.stderr


def test_every_scorer_is_preceded_by_a_complete_merge(repo_root: Path) -> None:
    scorers = merges = 0
    discovered: list[str] = []
    for method, benchmark, commands in _all_sequences(repo_root):
        merged: dict[tuple[int | None, str], int] = {}
        chunk_files: dict[tuple[int | None, str], set[str]] = {}
        for position, command in enumerate(commands):
            where = f"{method}/{benchmark}/task{command.task_id}"
            answers = _arg(command.argv, "--answers-file")
            if answers and Path(answers).name[0].isdigit():
                chunk_files.setdefault((command.task_id, str(Path(answers).parent)), set()).add(answers)
            if command.metadata.get("merge_chunks"):
                output = _arg(command.argv, "--output")
                directory = str(Path(output).parent)
                assert Path(output).name == "merge.jsonl", where
                if "--inputs" in command.argv:
                    inputs = command.argv[command.argv.index("--inputs") + 1:]
                    assert [Path(item).name.split("_")[1] for item in inputs] == [f"{i}.jsonl" for i in range(len(inputs))], where
                    assert set(inputs) == chunk_files[(command.task_id, directory)], f"{where}: merge must cover every chunk"
                else:
                    assert _arg(command.argv, "--discover") == directory, where
                    discovered.append(f"{method}/{benchmark}/task{command.task_id}:{directory}")
                merged[(command.task_id, directory)] = position
                merges += 1
            result_file = _arg(command.argv, "--result-file")
            if result_file and Path(result_file).name == "merge.jsonl":
                key = (command.task_id, str(Path(result_file).parent))
                assert key in merged, f"{where}: scorer reads {result_file} before any merge step"
                assert merged[key] < position, where
                scorers += 1
    assert scorers >= 400 and merges >= 400
    # The only scorers whose chunks are not written by their own sequence: v3's MR-LoRA MLLM-DCL
    # evaluation scripts have the inference commented out (see docs/migration/drift.md, G10).
    assert {item.split("/task")[0] for item in discovered} == {"mrlora/mllm_dcl"}
    assert len(discovered) == 25


def test_discover_mode_requires_exactly_one_complete_chunk_set(tmp_path: Path) -> None:
    from mcitlib.eval.chunks import discover_chunks

    with pytest.raises(FileNotFoundError, match="no chunk files"):
        discover_chunks(tmp_path)
    (tmp_path / "2_0.jsonl").write_text("a\n")
    with pytest.raises(FileNotFoundError, match="complete"):
        discover_chunks(tmp_path)
    (tmp_path / "2_1.jsonl").write_text("b\n")
    assert [item.name for item in discover_chunks(tmp_path)] == ["2_0.jsonl", "2_1.jsonl"]
    for index in range(4):
        (tmp_path / f"4_{index}.jsonl").write_text("stale\n")
    with pytest.raises(FileNotFoundError, match="exactly one"):
        discover_chunks(tmp_path)


def test_merge_covers_all_195_v3_chunked_evaluations(repo_root: Path) -> None:
    """v3 had 195 evaluation scripts that each ended in the same ``cat`` loop."""
    groups = set()
    for method, benchmark, commands in _all_sequences(repo_root):
        for command in commands:
            if command.metadata.get("merge_chunks"):
                groups.add((method, benchmark, command.sequence, command.task_id, _arg(command.argv, "--output")))
    assert len(groups) >= 195
