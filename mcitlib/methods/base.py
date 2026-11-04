# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Method plugin interface shared by all LLaVA methods."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from ..config.schema import DataSource, Stage, Step, TaskSpec
from ..context import RunContext


class ContinualMethod(ABC):
    id: ClassVar[str]
    aliases: ClassVar[tuple[str, ...]] = ()
    env: ClassVar[str] = "llava"
    required_capabilities: ClassVar[frozenset[str]] = frozenset()
    train_entry: ClassVar[str] = "standard"
    deepspeed_config: ClassVar[str] = "zero2"
    separate_train_and_infer: ClassVar[bool] = False
    model_override: ClassVar[str | None] = None

    def __init__(self, cfg: dict[str, Any] | None = None, ctx: Any = None):
        self.cfg = cfg or {}

    @abstractmethod
    def select_train_data(self, task: TaskSpec) -> DataSource:
        raise NotImplementedError

    def extra_cli_args(self, task: TaskSpec) -> dict[str, Any]:
        return {}

    def model_class_override(self) -> type | None:
        return None

    def build_trainable(self, model: Any, backbone: Any) -> Any:
        return model

    def load_previous(self, model: Any, prev: Path) -> Any:
        return model

    def training_overrides(self) -> dict[str, Any]:
        return {}

    def extra_loss(self, model: Any, inputs: Any, outputs: Any, base_loss: Any) -> Any:
        return None

    def before_task(self, ctx: RunContext) -> list[Step]:
        return []

    def after_task(self, ctx: RunContext) -> list[Step]:
        return []

    def prepare_inference(self, model: Any, ctx: RunContext) -> Any:
        return model

    def extra_stages(self) -> list[Stage]:
        return []
