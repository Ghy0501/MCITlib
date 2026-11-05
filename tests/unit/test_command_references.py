# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""B1/B4: every generated executable module, script, and deepspeed config exists."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from mcitlib.registry import BENCHMARKS, METHODS
from mcitlib.runner import build_sequence_commands

EVAL_ENTRY = "mcitlib.backbones.llava.eval_runners.entry"


def _sequences(repo_root: Path):
    for method in METHODS:
        for benchmark in BENCHMARKS:
            yield method, benchmark, build_sequence_commands(
                repo_root / "configs", "llava", method.id, benchmark.id
            )
    for benchmark in BENCHMARKS:
        method = next(item for item in METHODS if item.id == "mrlora")
        yield method, benchmark, build_sequence_commands(
            repo_root / "configs", "llava", method.id, benchmark.id, sequence="router"
        )


def test_all_1696_commands_reference_existing_code(repo_root: Path) -> None:
    errors: list[str] = []
    modules: set[str] = set()
    scripts: set[Path] = set()
    ordinary_count = 0
    for method, benchmark, commands in _sequences(repo_root):
        if method.id != "mrlora" or not any(c.sequence == "router" for c in commands):
            ordinary_count += len(commands)
        for command in commands:
            argv = command.argv
            prefix = f"{method.id}/{benchmark.id}/{command.task_id}/{command.kind}"
            if argv and argv[0] in {"python", "python3"} and len(argv) > 1 and argv[1].endswith(".py"):
                path = Path(argv[1])
                path = path if path.is_absolute() else repo_root / path
                scripts.add(path)
                if not path.is_file():
                    errors.append(f"{prefix}: missing script {path}")
            if "--deepspeed" in argv:
                path = Path(argv[argv.index("--deepspeed") + 1])
                if not path.is_file():
                    errors.append(f"{prefix}: missing deepspeed config {path}")
            if "--module" in argv:  # deepspeed launcher form used by training commands
                modules.add(argv[argv.index("--module") + 1])
            if "-m" in argv:
                index = argv.index("-m")
                module = argv[index + 1]
                modules.add(module)
                if module == EVAL_ENTRY:
                    target = argv[index + 2]
                    modules.add(target)
                    assert command.metadata["eval_target"] == target
    assert ordinary_count == 1696
    for module in sorted(modules):
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, AttributeError, ValueError) as error:
            errors.append(f"module {module}: {error}")
        else:
            if spec is None:
                errors.append(f"module not found: {module}")
    assert not errors, "\n".join(errors)
    assert len(modules) >= 20
    assert scripts
