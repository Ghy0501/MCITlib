# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Small, dependency-free configuration and command data structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataSource:
    """The path fields used by the v3 data configuration files."""

    train_path: str | None = None
    replay_path: str | None = None
    router_path: str | None = None
    test_path: str | None = None
    train_folder: str | None = None
    router_folder: str | None = None
    test_folder: str | None = None

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "DataSource":
        names = cls.__dataclass_fields__
        return cls(**{name: values.get(name) for name in names})

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class TaskSpec:
    """One task in a continual-learning benchmark sequence."""

    benchmark: str
    task_id: int
    name: str
    model_config: str
    data_config: str
    train_config: str
    eval_config: str | None = None
    router: bool = False
    data: DataSource | None = None

    @property
    def task_key(self) -> str:
        return f"task_{self.task_id}"


@dataclass
class RunConfig:
    """Merged configuration passed to adapters and method plugins."""

    backbone: str
    method: str
    benchmark: str
    task: TaskSpec | None = None
    values: dict[str, Any] = field(default_factory=dict)
    paths: Any = None

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


@dataclass
class Command:
    """An executable step, retained as argv rather than a shell string."""

    argv: list[str]
    cwd: str = "."
    env: dict[str, str] = field(default_factory=dict)
    kind: str = "other"
    task_id: int | None = None
    sequence: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "cwd": self.cwd,
            "env": dict(self.env),
            "kind": self.kind,
            "task_id": self.task_id,
            "sequence": self.sequence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Step:
    """A non-standard training step supplied by a method plugin."""

    name: str
    command: Command


@dataclass(frozen=True)
class Stage:
    """A named group of steps, such as MR-LoRA router evaluation."""

    name: str
    steps: tuple[Step, ...] = ()


@dataclass(frozen=True)
class PathSet:
    """Roots used to resolve all user-provided paths."""

    model_root: Path
    data_root: Path
    ckpt_root: Path
    repo_root: Path
    results_root: Path | None = None  # evaluation output; default <repo_root>/results
    logs_root: Path | None = None  # run records; default <repo_root>/logs
