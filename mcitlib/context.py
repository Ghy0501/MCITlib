# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Execution context shared by backbone adapters and method plugins."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config.schema import PathSet, TaskSpec


@dataclass
class RunContext:
    backbone: str
    method: str
    benchmark: str
    paths: PathSet
    task: TaskSpec | None = None
    output_dir: Path | None = None
    seed: int = 42
    dry_run: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> int | None:
        return self.task.task_id if self.task else None

    def child(self, task: TaskSpec, output_dir: Path | None = None) -> "RunContext":
        return RunContext(
            backbone=self.backbone,
            method=self.method,
            benchmark=self.benchmark,
            paths=self.paths,
            task=task,
            output_dir=output_dir or self.output_dir,
            seed=self.seed,
            dry_run=self.dry_run,
            diagnostics=dict(self.diagnostics),
        )
