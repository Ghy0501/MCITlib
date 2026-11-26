# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""SAME's separate CLIP prior for native eight-frame CL-VISTA samples."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor

from ...methods._same.prior import PriorRuntime, PriorState, load_clip_runtime
from .checkpointing import video_non_reentrant_scope
from .expert_hooks import ExpertHooks

POOLING = "eight_native_indices_clip_image_embeds_fp32_mean_then_l2"


class VideoCLIP:
    def __init__(self, backbone, path):
        self.backbone = backbone
        self.processor = CLIPImageProcessor.from_pretrained(path, local_files_only=True)

    def frames(self, path):
        from decord import VideoReader

        video = VideoReader(str(path), num_threads=1)
        if self.backbone == "videollava":
            indices = np.linspace(0, len(video) - 1, 8, dtype=int)
        else:
            from videollama2.mm_utils import frame_sample

            indices = frame_sample(len(video), mode="uniform", num_frames=8)
        raw = video.get_batch(indices)
        values = raw.cpu().numpy() if torch.is_tensor(raw) else raw.asnumpy()
        if values.ndim != 4 or len(values) != 8:
            raise RuntimeError(f"SAME expected eight decoded RGB frames from {path}")
        pixels = self.processor.preprocess([Image.fromarray(frame) for frame in values],
                                           return_tensors="pt")["pixel_values"]
        return pixels


class VideoPriorRuntime(PriorRuntime):
    def __init__(self, current, clip, backbone):
        super().__init__(current.state, current.tokenizer, current.clip_tokenizer,
                         current.text_tower, current.image_tower, current.task_id)
        self.clip = clip
        self.backbone = backbone
        self.pixels = None

    def _extract(self, input_ids, images):
        frames = self.pixels
        if frames is None or frames.ndim != 5 or frames.shape[:2] != (input_ids.shape[0], 8):
            raise RuntimeError("SAME requires separate CLIP-normalized RGB frames for every native video")
        visual_param = next(self.image_tower.parameters())
        text_param = next(self.text_tower.parameters())
        with torch.no_grad():
            features = self.image_tower(pixel_values=frames.flatten(0, 1).to(visual_param)).image_embeds
            visual = F.normalize(features.float().view(frames.shape[0], 8, -1).mean(dim=1), dim=-1)
            ids = input_ids.masked_fill(input_ids.lt(0), self.tokenizer.pad_token_id or 0).cpu()
            decoded = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
            clean = ["\n".join(value.split("\n")[1:]) for value in decoded]
            if self.backbone == "videollava":
                clean = [value.split(" ASSISTANT")[0] for value in clean]
            else:
                clean = [value.split("<</SYS>>")[-1].split("[/INST]")[0].strip() for value in clean]
            tokens = self.clip_tokenizer(clean, padding="longest", truncation=True, max_length=77,
                                         return_tensors="pt").to(text_param.device)
            text = self.text_tower(**tokens).pooler_output
        return visual.to(text.device), text

    @contextmanager
    def bind(self, frames):
        if self.pixels is not None:
            raise RuntimeError("A separate CLIP frame batch is already bound")
        self.pixels = frames
        try:
            yield
        finally:
            self.pixels = None

    def install_training_hook(self, model):
        def before_forward(module, args, kwargs):
            frames = kwargs.pop("same_video_frames", None)
            if module.training:
                self.state.synchronize_once()
            with self.bind(frames):
                input_ids = kwargs.get("input_ids", args[0] if args else None)
                self.prepare(module, input_ids, kwargs.get("images"),
                             training=module.training, grad_enabled=torch.is_grad_enabled())

        return model.register_forward_pre_hook(before_forward, with_kwargs=True)

    def install_generate_hook(self, model):
        super().install_generate_hook(model)

        @contextmanager
        def request(path):
            with self.bind(self.clip.frames(path).unsqueeze(0)):
                yield

        model.bind_clip_video = request
        return model


class SameHooks(ExpertHooks):
    def __init__(self, backbone, options):
        super().__init__("same", options)
        self.backbone, self.task = backbone, int(options["cur_task"])
        self.path = options["clip_path"]
        self.state = None
        self.wrapped = None

    def build(self, model, config):
        self.state = PriorState(int(self.options["expert_num"]), device=model.device)
        self.wrapped = super().build(model, config)
        self.wrapped.config.same_video_clip_path = self.path
        self.wrapped.config.same_video_pooling = POOLING
        return self.wrapped

    def install_video_prior(self, tokenizer, previous, rank):
        if previous is not None:
            self.state.load(previous)
            self.state.require_observed_through(self.task - 1)
            self.plugin.load_previous(self.wrapped, Path(previous))
        elif self.task != 0:
            raise ValueError("SAME video continuation requires a prior task checkpoint")
        tower = self.wrapped.get_video_tower() if self.backbone == "videollava" else self.wrapped.get_vision_tower()
        runtime = load_clip_runtime(self.state, tokenizer, self.path, device=tower.device,
                                    dtype=tower.dtype, task_id=self.task)
        self.runtime = VideoPriorRuntime(runtime, VideoCLIP(self.backbone, self.path), self.backbone)
        self.runtime.install_training_hook(self.wrapped)

    def wrap_data(self, data, data_args):
        native = data["train_dataset"]
        root = Path(data_args.video_folder if self.backbone == "videollava" else data_args.data_folder)
        clip = self.runtime.clip
        original = data["data_collator"]

        class Frames(Dataset):
            def __getitem__(self, index):
                record = native[index]
                path = root / native.list_data_dict[index]["video"]
                return record, clip.frames(path)

            def __len__(self):
                return len(native)

            @property
            def modality_lengths(self):
                return native.modality_lengths

        def collate(items):
            rows, frames = zip(*items, strict=True)
            result = original(rows)
            result["same_video_frames"] = torch.stack(frames)
            return result

        return {**data, "train_dataset": Frames(), "data_collator": collate}

    def checkpoint_scope(self, args):
        return video_non_reentrant_scope(args, "SAME video")

    def before_checkpoint(self, model):
        self.plugin.finish_task(model)
        self.state.complete_task(self.task)

    def after_checkpoint(self, output):
        self.state.save(output)


def install_same_inference(model, tokenizer, backbone, checkpoint):
    import json

    directory = Path(checkpoint)
    adapter = json.loads((directory / "adapter_config.json").read_text())
    path = model.config.same_video_clip_path
    if model.config.same_video_pooling != POOLING:
        raise ValueError("SAME video CLIP recipe differs from the saved model")
    active = int(adapter["cur_task"])
    state = PriorState(int(adapter["expert_num"]), device=model.device)
    state.load(directory)
    state.require_observed_through(active)
    runtime = load_clip_runtime(state, tokenizer, path, device=model.device, dtype=model.dtype, task_id=active)
    wrapper = VideoPriorRuntime(runtime, VideoCLIP(backbone, path), backbone)
    wrapper.install_generate_hook(model)
