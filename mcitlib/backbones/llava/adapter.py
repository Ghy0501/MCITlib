# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""LLaVA-1.5-7B backbone adapter."""
from __future__ import annotations

from typing import Any, ClassVar

from ...config.schema import Command, DataSource, RunConfig, TaskSpec
from ..base import BackboneAdapter
from .args import build_train_command
from .data import build_dataset


class LlavaAdapter(BackboneAdapter):
    id: ClassVar[str] = "llava"
    env: ClassVar[str] = "llava"
    capabilities: ClassVar[frozenset[str]] = frozenset({"text_tower", "clip_features", "layer_index", "model_class_override"})

    def build_command(self, run_cfg: RunConfig, method: Any, task: TaskSpec) -> Command:
        return build_train_command(run_cfg, task, getattr(method, "train_entry", "standard"))

    def load_base(self, model_cfg: Any, override_cls: type | None = None) -> Any:
        if override_cls is not None:
            raise NotImplementedError("method-private model overrides are installed with SEFE/KeepLoRA")
        from llava.model.builder import load_pretrained_model
        values = model_cfg if isinstance(model_cfg, dict) else vars(model_cfg)
        model_name = values["model_name"]
        return load_pretrained_model(
            model_path=model_name,
            model_base=values.get("model_base"),
            model_name=values.get("model_name", model_name),
            load_8bit=values.get("load_8bit", False),
            load_4bit=values.get("load_4bit", False),
            device_map=values.get("device_map", "auto"),
            device=values.get("device", "cuda"),
        )

    def build_dataset(self, source: DataSource, split: str, entry: str, **kwargs: Any) -> Any:
        return build_dataset(source, split, entry, **kwargs)

    def make_trainer(self, model: Any, dataset: Any, args: Any, method: Any, **kwargs: Any) -> Any:
        from .trainer import LLaVATrainer
        trainer = LLaVATrainer(model=model, args=args, train_dataset=dataset, **kwargs)
        trainer.mcit_method = method
        return trainer

    def generate(self, model: Any, batch: Any, gen_cfg: Any) -> list[str]:
        return model.generate(**batch, **gen_cfg)

    def default_conv_mode(self) -> str:
        return "vicuna_v1"
