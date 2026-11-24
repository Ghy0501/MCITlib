# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""G1: every training command must be a valid deepspeed launcher invocation.

deepspeed treats the first positional argument as the *user script*. ``deepspeed
... python -m X`` therefore makes each rank run ``python -u python -m X``.  The
module has to be passed with the launcher flag ``--module``.  This test parses
every generated training argv with deepspeed's own argument parser when it is
installed, and with an equivalent structural rule otherwise.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from mcitlib.backbones.llava.args import ENTRY_MODULE
from mcitlib.registry import BENCHMARKS, METHODS
from mcitlib.runner import build_sequence_commands

LAUNCHER_FLAGS_WITH_VALUE = {"--include", "--exclude", "--master_port", "--master_addr", "--num_gpus", "--num_nodes"}


def _training_commands(repo_root: Path):
    for method in METHODS:
        for benchmark in BENCHMARKS:
            if method.supported_benchmarks is not None and benchmark.id not in method.supported_benchmarks:
                continue
            for command in build_sequence_commands(repo_root / "configs", "llava", method.id, benchmark.id):
                if command.kind == "train":
                    yield method, benchmark, command
    router_method = next(item for item in METHODS if item.id == "mrlora")
    for benchmark in BENCHMARKS:
        for command in build_sequence_commands(
            repo_root / "configs", "llava", router_method.id, benchmark.id, sequence="router"
        ):
            if command.kind == "train":
                yield router_method, benchmark, command


def _structural_parse(argv: list[str]) -> tuple[bool, str, list[str]]:
    assert argv[0] == "deepspeed"
    index, module = 1, False
    while index < len(argv):
        token = argv[index]
        if token == "--module":
            module = True
            index += 1
        elif token in LAUNCHER_FLAGS_WITH_VALUE:
            index += 2
        elif token.startswith("--") and token.split("=")[0] in LAUNCHER_FLAGS_WITH_VALUE:
            index += 1
        else:
            return module, token, argv[index + 1:]
    raise AssertionError("deepspeed argv has no user script")


def test_every_training_argv_runs_the_entry_module(repo_root: Path) -> None:
    count = 0
    for method, benchmark, command in _training_commands(repo_root):
        module, user_script, user_args = _structural_parse(list(command.argv))
        where = f"{method.id}/{benchmark.id}/task{command.task_id}"
        assert module, f"{where}: --module missing, deepspeed would run {user_script!r} as a file"
        assert user_script == ENTRY_MODULE, where
        assert "python" not in command.argv[:command.argv.index("--module") + 2], where
        assert "--deepspeed" in user_args and "--output_dir" in user_args, where
        count += 1
    assert count >= 195


def test_every_training_argv_parses_with_deepspeed(repo_root: Path) -> None:
    runner = pytest.importorskip("deepspeed.launcher.runner") if not os.environ.get("MCITLIB_REQUIRE_DEEPSPEED") \
        else __import__("deepspeed.launcher.runner", fromlist=["parse_args"])
    count = 0
    for method, benchmark, command in _training_commands(repo_root):
        args = runner.parse_args(args=list(command.argv)[1:])
        where = f"{method.id}/{benchmark.id}/task{command.task_id}"
        assert args.module is True, where
        assert args.user_script == ENTRY_MODULE, where
        assert args.include and args.master_port, where
        assert "--deepspeed" in args.user_args, where
        count += 1
    assert count >= 195
