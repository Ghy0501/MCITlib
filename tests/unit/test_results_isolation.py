# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""G6: two methods never write to the same evaluation folder."""
from __future__ import annotations

from pathlib import Path

from mcitlib.paths import PathResolver
from mcitlib.registry import BENCHMARKS, METHODS
from mcitlib.runner import build_sequence_commands

OUTPUT_FLAGS = ("--answers-file", "--output-dir", "--output_xlsx", "--output_file", "--output")


def _paths(tmp_path: Path) -> PathResolver:
    config = tmp_path / "paths.yaml"
    config.write_text(f"model_root: {tmp_path}/m\ndata_root: {tmp_path}/d\nckpt_root: {tmp_path}/c\n")
    return PathResolver.from_file(config, repo_root=Path(__file__).resolve().parents[2])


def _written(commands) -> set[str]:
    found = set()
    for command in commands:
        for flag in OUTPUT_FLAGS:
            if flag in command.argv:
                found.add(command.argv[command.argv.index(flag) + 1])
    return found


def test_no_two_methods_share_an_output_folder(repo_root: Path, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    owners: dict[str, set[str]] = {}
    for method in METHODS:
        for benchmark in BENCHMARKS:
            if method.supported_benchmarks is not None and benchmark.id not in method.supported_benchmarks:
                continue
            if benchmark.id == "general":
                continue
            commands = build_sequence_commands(repo_root / "configs", "llava", method.id, benchmark.id, paths=paths)
            for target in _written(commands):
                owners.setdefault(str(Path(target).parent), set()).add(method.id)
    shared = {folder: sorted(who) for folder, who in owners.items() if len(who) > 1}
    assert not shared, f"{len(shared)} shared folders, e.g. {next(iter(shared.items()))}"
    root = str(paths.paths.results_root)
    assert all(folder.startswith(root) for folder in owners if "each_dataset" in folder)


def test_results_root_is_configurable_and_defaults_to_project(repo_root: Path, tmp_path: Path) -> None:
    default = _paths(tmp_path)
    assert default.paths.results_root == (repo_root / "results").resolve()
    config = tmp_path / "custom.yaml"
    config.write_text(f"model_root: {tmp_path}/m\ndata_root: {tmp_path}/d\nckpt_root: {tmp_path}/c\nresults_root: {tmp_path}/r\n")
    custom = PathResolver.from_file(config, repo_root=repo_root)
    assert custom.method_results_dir("olora") == (tmp_path / "r" / "olora").resolve()
    commands = build_sequence_commands(repo_root / "configs", "llava", "olora", "ucit", paths=custom)
    assert any(str(tmp_path / "r" / "olora") in " ".join(command.argv) for command in commands)


def test_router_extraction_reads_the_methods_own_results(repo_root: Path, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    commands = build_sequence_commands(repo_root / "configs", "llava", "mrlora", "mllm_acl", paths=paths)
    extract = [c for c in commands if any("router_results" in value for value in c.argv)]
    assert extract and all(c.env["MCITLIB_RESULTS_ROOT"] == str(paths.method_results_dir("mrlora")) for c in extract)
    expected = str(paths.method_results_dir("mrlora") / "ACL" / "each_dataset")
    written = {str(Path(t).parents[2]) for t in _written(commands) if "each_dataset" in t}
    assert written == {expected}, "ACL evaluation must write where the router extraction reads (AR-8)"
