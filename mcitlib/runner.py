# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Continual-learning orchestration interfaces.

The LLaVA command builder is installed in L2. L1 deliberately keeps execution
behind this small interface so configuration and command-dump tooling can be
tested without importing torch or transformers.
"""
from __future__ import annotations

from typing import Any

from .config.schema import Command, RunConfig, TaskSpec


def build_commands(run_cfg: RunConfig, tasks: list[TaskSpec], dry_run: bool = True) -> list[Command]:
    """Build commands through the selected backbone adapter.

    Adapters register a ``build_commands`` callable on the run configuration in
    later phases. No shell command is executed by this function.
    """
    builder: Any = run_cfg.values.get("command_builder")
    if builder is None:
        raise NotImplementedError("the LLaVA adapter command builder is installed in L2")
    return list(builder(run_cfg, tasks, dry_run=dry_run))


def run(*args: Any, **kwargs: Any) -> list[Command]:
    """Execution entry point; L2 adds the actual deepspeed runner."""
    if not kwargs.get("dry_run", True):
        raise NotImplementedError("actual training execution is installed after the L2 adapter")
    return build_commands(*args, **kwargs)
