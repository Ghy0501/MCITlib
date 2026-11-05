# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Import-level self-check of the exact process each generated command would start.

``mcitlib doctor`` and the entry-point tests run every *distinct* entry point of a
sequence in a fresh interpreter, with the same environment the runner builds, but
without loading a model.  That catches wrong module names, stale ``llava``/``peft``
installs, missing packages and transformers-version conflicts before any GPU time
is spent.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config.schema import Command
from .paths import merge_pythonpath, process_env

TRAIN_ENTRY_MODULE = "mcitlib.backbones.llava.train_entry"
EVAL_ENTRY_MODULE = "mcitlib.backbones.llava.eval_runners.entry"

_IMPORT_MODULE = "import importlib, sys; importlib.import_module(sys.argv[1]); print('selfcheck ok', sys.argv[1])"
_RUN_PATH = (
    "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__mcitlib_selfcheck__'); "
    "print('selfcheck ok', sys.argv[1])"
)


@dataclass(frozen=True)
class EntryPoint:
    kind: str  # train | eval | module | script
    target: str
    method: str | None = None
    train_entry: str | None = None
    model_override: str | None = None

    def label(self) -> str:
        extra = "/".join(item for item in (self.method, self.train_entry, self.model_override) if item)
        return f"{self.kind}:{self.target}" + (f" [{extra}]" if extra else "")


@dataclass
class CheckResult:
    entry: EntryPoint
    ok: bool
    returncode: int
    message: str


def classify(command: Command) -> EntryPoint | None:
    argv = list(command.argv)
    env = command.env
    method = env.get("MCITLIB_METHOD")
    if not argv:
        return None
    if argv[0] == "deepspeed" and "--module" in argv:
        target = argv[argv.index("--module") + 1]
        return EntryPoint("train", target, method, env.get("MCITLIB_TRAIN_ENTRY"), env.get("MCITLIB_MODEL_OVERRIDE"))
    if argv[0] in {"python", "python3", sys.executable} and len(argv) > 2 and argv[1] == "-m":
        if argv[2] == EVAL_ENTRY_MODULE:
            return EntryPoint("eval", argv[3], method, None, env.get("MCITLIB_MODEL_OVERRIDE"))
        return EntryPoint("module", argv[2], method, None, env.get("MCITLIB_MODEL_OVERRIDE"))
    if argv[0] in {"python", "python3", sys.executable} and len(argv) > 1 and argv[1].endswith(".py"):
        return EntryPoint("script", argv[1], method, None, env.get("MCITLIB_MODEL_OVERRIDE"))
    return None


def distinct_entry_points(commands: list[Command]) -> dict[EntryPoint, Command]:
    result: dict[EntryPoint, Command] = {}
    for command in commands:
        entry = classify(command)
        if entry is not None:
            result.setdefault(entry, command)
    return result


def _argv(entry: EntryPoint) -> list[str]:
    if entry.kind == "train":
        return [sys.executable, "-m", entry.target]
    if entry.kind == "eval":
        return [sys.executable, "-m", EVAL_ENTRY_MODULE, entry.target]
    if entry.kind == "module":
        return [sys.executable, "-c", _IMPORT_MODULE, entry.target]
    return [sys.executable, "-c", _RUN_PATH, entry.target]


def run_selfcheck(
    entry: EntryPoint,
    command: Command,
    repo_root: Path,
    extra_pythonpath: tuple[str, ...] = (),
    timeout: int = 900,
) -> CheckResult:
    env = process_env(command.env, repo_root)
    env["MCITLIB_SELFCHECK"] = "1"
    if extra_pythonpath:
        env["PYTHONPATH"] = merge_pythonpath(env["PYTHONPATH"], *extra_pythonpath, root=repo_root)
    argv = _argv(entry)
    if entry.kind == "script":
        script = Path(entry.target)
        argv[-1] = str(script if script.is_absolute() else repo_root / script)
    process = subprocess.run(
        argv, cwd=repo_root, env=env, capture_output=True, text=True, timeout=timeout,
    )
    text = (process.stderr + "\n" + process.stdout).strip().splitlines()
    errors = [line for line in text if line[:1].isalpha() and line.split(":")[0].endswith(("Error", "Exception"))]
    message = (errors or [line for line in text if line.strip() and "[INFO]" not in line] or [""])[-1][:300]
    return CheckResult(entry, process.returncode == 0, process.returncode, message)
