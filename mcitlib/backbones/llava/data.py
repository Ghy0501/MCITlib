# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Lazy access to the separate standard and MoE v3 data pipelines."""
from __future__ import annotations

from typing import Any

from ...config.schema import DataSource


def build_dataset(source: DataSource, split: str, entry: str, **kwargs: Any) -> Any:
    """Build the v3 dataset class for one entry without merging its behavior."""
    if entry == "standard":
        from .standard_train import LazySupervisedDataset
    elif entry == "moe":
        from .moe_train import LazySupervisedDataset
    else:
        raise ValueError(f"unknown LLaVA data entry: {entry}")
    if "tokenizer" not in kwargs or "data_args" not in kwargs:
        raise TypeError("build_dataset requires tokenizer and data_args")
    data_args = kwargs["data_args"]
    if split == "train":
        data_path, image_folder = source.train_path, source.train_folder
    elif split in {"test", "eval"}:
        data_path, image_folder = source.test_path, source.test_folder
    else:
        raise ValueError(f"unknown dataset split: {split}")
    if data_path is None:
        raise ValueError(f"data source has no {split}_path")
    # The v3 classes read these fields from data_args. Set them explicitly on a
    # shallow copy so a Replay selection cannot leak into the next task.
    import copy
    data_args = copy.copy(data_args)
    data_args.data_path = data_path
    data_args.image_folder = image_folder
    return LazySupervisedDataset(data_path=data_path, tokenizer=kwargs["tokenizer"], data_args=data_args)
