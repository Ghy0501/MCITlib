# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Backbone adapter interface; concrete LLaVA implementation arrives in L2."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ..config.schema import Command, DataSource, RunConfig, TaskSpec


class BackboneAdapter(ABC):
    id: ClassVar[str]
    env: ClassVar[str]
    capabilities: ClassVar[frozenset[str]] = frozenset()

    @abstractmethod
    def build_command(self, run_cfg: RunConfig, method: Any, task: TaskSpec) -> Command:
        raise NotImplementedError

    def load_base(self, model_cfg: Any, override_cls: type | None = None) -> Any:
        raise NotImplementedError

    def build_dataset(self, source: DataSource, split: str, entry: str) -> Any:
        raise NotImplementedError

    def make_trainer(self, model: Any, dataset: Any, args: Any, method: Any) -> Any:
        raise NotImplementedError

    def generate(self, model: Any, batch: Any, gen_cfg: Any) -> list[str]:
        raise NotImplementedError
