# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CL-VISTA queries and dataset adaptation for the existing PCLR rank pool."""
from __future__ import annotations

import importlib

import torch
from torch import nn
from torch.nn import functional as F

from ...methods._pclr.core import mean_query
from ...methods._pclr.llava import model_autocast


class VideoQueries:
    def __init__(self, backbone):
        self.backbone = backbone
        self.llava = backbone == "videollava"
        module = "videollava.train.train" if self.llava else "videollama2.train"
        self.native = importlib.import_module(module)
        self.trainer = self.native.LLaVATrainer if self.llava else self.native.VideoLLaMA2Trainer

    def tower(self, model):
        return model.get_video_tower() if self.llava else model.get_vision_tower()

    def query(self, model, input_ids, mask, images, has_image=None):
        if images is None or has_image is not None and not all(has_image):
            raise ValueError("CL-VISTA PCLR requires one real video per query")
        with torch.no_grad(), model_autocast(model):
            text = mean_query(model.get_model().embed_tokens(input_ids), mask)
            tower = self.tower(model)
            if self.llava:
                batch = images if torch.is_tensor(images) else torch.stack(images)
                if batch.ndim != 5 or batch.shape[1] != 3:
                    raise ValueError(f"Expected LanguageBind [B,C,T,H,W]: {batch.shape}")
                frames = tower(batch)[:, :, 0]
            else:
                if not all(isinstance(item, (tuple, list)) and len(item) == 2 and item[1] == "video" for item in images):
                    raise ValueError("PCLR expected VideoLLaMA2's typed video tensor list")
                batch = torch.stack([item[0] for item in images])
                if batch.ndim != 5 or batch.shape[2] != 3:
                    raise ValueError(f"Expected VideoLLaMA2 [B,T,C,H,W]: {batch.shape}")
                b, t = batch.shape[:2]
                feature = tower.select_feature
                try:
                    tower.select_feature = "cls_patch"
                    frames = tower(batch.flatten(0, 1))[:, 0].reshape(b, t, -1)
                finally:
                    tower.select_feature = feature
            if frames.shape[:2] != (text.shape[0], 8):
                raise ValueError(f"PCLR video query expected eight genuine temporal positions: {frames.shape}")
            # Explicit video extension: average native per-frame CLS features in
            # FP32, then L2-normalize the resulting video query once.
            vision = F.normalize(frames.float().mean(dim=1), dim=-1).to(text)
            return text.detach(), vision.detach()

    @staticmethod
    def move_media(value, device):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, list):
            return [VideoQueries.move_media(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(VideoQueries.move_media(item, device) for item in value)
        return value

    @staticmethod
    def targets(model):
        import re

        from .expert_hooks import LLM_TARGETS

        result = {}
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if name.startswith("model.mm_projector."):
                result.setdefault("mm_projector", []).append(name)
            elif re.fullmatch(LLM_TARGETS, name):
                result.setdefault("layer_" + name.split(".")[2], []).append(name)
        if len(result) != 33 or sum(len(items) for key, items in result.items() if key != "mm_projector") != 224:
            raise RuntimeError("PCLR layer/connector groups do not match the native video model")
        return result

    def make_data(self, tokenizer, data_args):
        native = self.native

        class Dataset(native.LazySupervisedDataset):
            def __getitem__(self, index):
                row = self.list_data_dict[index]
                if "video" not in row:
                    raise ValueError("CL-VISTA PCLR row does not have native video data")
                item = super().__getitem__(index)
                question = next(turn["value"] for turn in row["conversations"] if turn["from"] == "human")
                question = question.replace("<video>", "").strip()
                item["instruction_ids"] = tokenizer(
                    question, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length,
                ).input_ids[0]
                return item

        class Collator(native.DataCollatorForSupervisedDataset):
            def __call__(self, instances):
                batch = super().__call__(instances)
                ids = nn.utils.rnn.pad_sequence(
                    [item["instruction_ids"] for item in instances], batch_first=True, padding_value=tokenizer.pad_token_id,
                )
                batch["instruction_ids"] = ids
                batch["instruction_mask"] = ids.ne(tokenizer.pad_token_id)
                batch["has_image"] = [True] * len(instances)
                return batch

        dataset = Dataset(data_path=data_args.data_path, tokenizer=tokenizer, data_args=data_args)
        return {"train_dataset": dataset, "eval_dataset": None, "data_collator": Collator(tokenizer=tokenizer)}


class PCLRHooks:
    method = "pclr"

    def __init__(self, backbone, options):
        self.backbone = backbone
        self.options = options

    def run(self, model, tokenizer, data_args, training_args, previous):
        import json
        import os

        from ...methods._pclr.llava import run

        os.environ.update(
            MCITLIB_PCLR_STAGE=self.options["stage"],
            MCITLIB_PCLR_TASK_ID=str(self.options["task_id"]),
            MCITLIB_PCLR_OPTIONS=json.dumps(self.options.get("pool", {})),
        )
        run(model, tokenizer, data_args, training_args, previous, video=VideoQueries(self.backbone))


def load_for_inference(backbone, base_path, checkpoint):
    from contextlib import contextmanager
    from pathlib import Path

    from ...methods._pclr.core import RankPool
    from ...methods._pclr.llava import RoutedLlava, keep_norms_fp32

    if backbone == "videollava":
        from videollava.model.builder import load_pretrained_model

        tokenizer, native, processor, _ = load_pretrained_model(
            base_path, None, "videollava", device_map={"": "cuda:0"},
        )
    else:
        from videollama2 import model_init

        native, processor, tokenizer = model_init(base_path, None, device_map={"": "cuda:0"})
        native.config.num_frames = checkpoint_frame_count(checkpoint)
    keep_norms_fp32(native)
    pool, phase = RankPool.load_boundary(native, Path(checkpoint) / "pclr.pt")
    if phase not in {"learned", "integrated"} or not len(pool.task_text):
        raise ValueError(f"Invalid CL-VISTA PCLR checkpoint phase: {phase}")

    class Request(RoutedLlava):
        def __init__(self):
            super().__init__(pool, query=VideoQueries(backbone).query)
            self.instruction = None

        @property
        def dtype(self):
            return self.pool.base.dtype

        @contextmanager
        def bind_question(self, question):
            if self.instruction is not None:
                raise RuntimeError("A PCLR request is already bound")
            self.instruction = tokenizer(question, return_tensors="pt", truncation=True,
                                         max_length=2048).input_ids.to(native.device)
            try:
                yield
            finally:
                self.instruction = None

        def generate(self, inputs, images=None, **kwargs):
            return super().generate(inputs, images=images, instruction_ids=self.instruction, **kwargs)

    return tokenizer, Request(), processor


def checkpoint_frame_count(checkpoint):
    import json
    from pathlib import Path

    config = json.loads((Path(checkpoint) / "config.json").read_text())
    frames = config.get("num_frames")
    if frames != 8:
        raise ValueError(f"Unexpected saved native PCLR video frame count: {frames}")
    return frames
