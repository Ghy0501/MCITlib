# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Recover the native DISCO and HiDe video routing from their actual visual forwards."""
from __future__ import annotations

import inspect
import json
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, CLIPTextModel

from ...tuners.inference_cache import expert_inference_scope
from ...tuners.variants.disco import MCITDiscoLoraConfig, MCITDiscoLoraLinear
from ...tuners.variants.hide import MCITHiDeLoraConfig, MCITHiDeLoraLinear

STATE_NAME = "video_route_state.pt"


def top_k(method, probabilities, maximum=3):
    values, indices = torch.topk(probabilities, min(maximum, len(probabilities)))
    total = values.sum()
    if method == "hide":
        values = values / (total + 1e-6)
    elif total > 1e-6:
        values = values / total
    return [(index, value) for index, value in zip(indices.tolist(), values.tolist(), strict=True)
            if method == "hide" or value > 1e-6]


class VideoPrior:
    def __init__(self, backbone, method, experts, *, initialization=True):
        self.backbone, self.method, self.experts = backbone, method, experts
        hidden = {"image": 1024, "text": 768}
        slots = max(10, experts) if method == "hide" else experts
        initializer = torch.randn if method == "hide" and initialization else torch.zeros
        self.means = {name: initializer(slots, n, dtype=torch.float32) for name, n in hidden.items()}
        if method == "hide" and initialization:
            self.means = {name: mean * .1 for name, mean in self.means.items()}
        self.count = {name: torch.ones(slots, dtype=torch.int64) for name in hidden}
        self.completed_task = -1

    def restore(self, directory, expected_task):
        path = Path(directory) / STATE_NAME
        state = torch.load(path, map_location="cpu")
        if (state.get("kind"), state.get("backbone"), state.get("expert_num"), state.get("task")) != (
            self.method, self.backbone, self.experts, expected_task,
        ):
            raise ValueError(f"Incompatible video prototype state: {path}")
        for name, prototype in self.means.items():
            value = state["means"][name]
            count = state["counts"][name]
            if value.shape != prototype.shape or value.dtype != torch.float32 or not value.isfinite().all():
                raise ValueError(f"Invalid {name} prototype state: {path}")
            if count.shape != (prototype.shape[0],) or count.dtype != torch.int64 or (count < 1).any():
                raise ValueError(f"Invalid {name} boundary state: {path}")
            if expected_task >= 0 and not (count[:expected_task + 1] > 1).all():
                raise ValueError(f"Unobserved historical {name} prototype: {path}")
            self.means[name] = value
            self.count[name] = count
        self.completed_task = expected_task

    def save(self, directory, task):
        if any(count[task].item() <= 1 for count in self.count.values()):
            raise RuntimeError("No real video/text samples contributed to the route prototype")
        path = Path(directory) / STATE_NAME
        with path.open("xb") as handle:
            torch.save({"kind": self.method, "backbone": self.backbone, "expert_num": self.experts,
                        "task": task, "means": self.means, "counts": self.count}, handle)

    def order(self, module, image, text, current):
        if image.shape[0] != text.shape[0]:
            raise ValueError("The image and text prototype batches differ")
        observations = {"image": image.float(), "text": text.float()}
        if module.training:
            for key, live in observations.items():
                total = live.sum(0)
                count = torch.tensor(len(live), device=total.device, dtype=torch.int64)
                if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                    torch.distributed.all_reduce(total)
                    torch.distributed.all_reduce(count)
                prior = self.means[key][current]
                n = self.count[key][current]
                self.means[key][current] = (prior * n + total.cpu()) / (n + count.cpu())
                self.count[key][current] += count.cpu()
            return None
        sims = []
        for key, live in observations.items():
            anchor = self.means[key][:self.experts].to(live.device)
            if self.method == "disco":
                sim = F.cosine_similarity(anchor, live.mean(0, keepdim=True), dim=-1)
            else:
                sim = F.cosine_similarity(anchor.unsqueeze(0), live.unsqueeze(1), dim=-1).max(0).values
            sims.append(sim)
        temperature = .05 if self.method == "disco" else .1
        return F.softmax((sims[0] + sims[1]) * (.5 / temperature), dim=0)

    def install(self, model, tokenizer, text_path, current):
        if getattr(model, "_video_prior_installed", False):
            raise RuntimeError("Video prior hooks may only be installed once")
        llava = self.backbone == "videollava"
        tower = model.get_video_tower() if llava else model.get_vision_tower()
        feature_model = tower.video_tower if llava else tower.vision_tower
        device, dtype = tower.device, tower.dtype
        clip_tokenizer = AutoTokenizer.from_pretrained(text_path, local_files_only=True)
        text_model = CLIPTextModel.from_pretrained(text_path, local_files_only=True).to(device=device, dtype=dtype)
        text_model.requires_grad_(False).eval()
        if (int(text_model.config.hidden_size), int(tower.hidden_size)) != (768, 1024):
            raise ValueError("CL-VISTA route tower geometry differs from the original video method")
        variant = MCITDiscoLoraLinear if self.method == "disco" else MCITHiDeLoraLinear
        layers = [m for m in model.modules() if isinstance(m, variant)]
        if len(layers) != 224:
            raise ValueError(f"Video expert route count is {len(layers)}, expected 224 language targets")
        prior_prepare = model.prepare_inputs_labels_for_multimodal
        signature = inspect.signature(prior_prepare)

        def prepare(base, *pos, **kwargs):
            inputs = signature.bind_partial(*pos, **kwargs).arguments
            tokens, media = inputs["input_ids"], inputs.get("images")
            if media is None or tokens.shape[1] == 1:
                return prior_prepare(*pos, **kwargs)
            visual = []

            def capture(_module, _inputs, result):
                visual.append(result.pooler_output.detach())

            with feature_model.register_forward_hook(capture):
                expanded = prior_prepare(*pos, **kwargs)
            if len(visual) != 1:
                raise RuntimeError(f"Expected one native video forward for a CL-VISTA batch, got {len(visual)}")
            image = visual[0]
            if not llava:
                image = image.reshape(tokens.shape[0], model.config.num_frames, -1).mean(dim=1)
            if image.shape != (tokens.shape[0], 1024):
                raise ValueError(f"Unexpected video routing pools: {image.shape}")
            with torch.no_grad():
                cleaned = tokens.masked_fill(tokens.lt(0), tokenizer.pad_token_id or 0).cpu()
                decoded = tokenizer.batch_decode(cleaned, skip_special_tokens=True)
                questions = ["\n".join(value.split("\n")[1:]) for value in decoded]
                if llava:
                    questions = [value.split(" ASSISTANT")[0] for value in questions]
                else:
                    questions = [value.split("<</SYS>>")[-1].split("[/INST]")[0].strip() for value in questions]
                text_input = clip_tokenizer(questions, padding="longest", max_length=77,
                                            truncation=True, return_tensors="pt").to(device)
                text = text_model(**text_input).pooler_output
                if text.device != image.device:
                    raise RuntimeError(f"Video route towers are on different devices: {text.device}, {image.device}")
                weights = self.order(base, image, text, current)
                if weights is not None:
                    probabilities = weights.tolist()
                    selection = top_k(self.method, weights)
                    print(json.dumps({"event": "video_expert_route", "method": self.method,
                                      "mode": "top3" if self.method == "disco" or llava else "layer31_full",
                                      "experts": [i for i, _ in selection] if self.method == "disco" or llava else list(range(self.experts))}),
                          flush=True)
                    for layer in layers:
                        if self.method == "disco":
                            layer.mask_signal = probabilities
                            layer.video_selection = selection
                        elif int(layer.layer) == 31:
                            layer.expert_weight = probabilities
                            layer.video_selection = selection if llava else None
            return expanded

        model.prepare_inputs_labels_for_multimodal = MethodType(prepare, model)
        model._video_prior_installed = True

        @contextmanager
        def request(_question):
            try:
                with expert_inference_scope():
                    yield
            finally:
                for layer in layers:
                    layer.video_selection = None
                    if self.method == "disco":
                        layer.mask_signal = [0.0] * self.experts
                    elif int(layer.layer) == 31:
                        layer.expert_weight = []
        model.bind_question = request


class VideoPriorHooks:
    def __init__(self, method, backbone, options):
        self.method, self.backbone, self.options = method, backbone, options
        self.task = int(options["cur_task"])
        self.prior = VideoPrior(backbone, method, int(options["expert_num"]))
        self.base = None

    def build(self, model, config):
        from peft import get_peft_model

        from .expert_hooks import LLM_TARGETS

        self.base = model
        config_cls = MCITDiscoLoraConfig if self.method == "disco" else MCITHiDeLoraConfig
        top = 3 if self.method == "disco" or self.backbone == "videollava" else None
        adapted = get_peft_model(model, config_cls(
            r=config.r, lora_alpha=config.lora_alpha, target_modules=LLM_TARGETS,
            lora_dropout=config.lora_dropout, bias=config.bias, task_type="CAUSAL_LM",
            expert_num=self.prior.experts, cur_task=self.task, video_top_k=top,
        ))
        # Each native training forward invokes only the current expert.
        # Keep the inactive bank in the checkpoint and outside the optimizer.
        self.active_banks = 0
        for layer in adapted.modules():
            if not isinstance(layer, (MCITDiscoLoraLinear, MCITHiDeLoraLinear)):
                continue
            a, b = layer.lora_A["default"], layer.lora_B["default"]
            for index, pair in enumerate(zip(a.loraA, b.loraB, strict=True)):
                for factor in pair:
                    factor.requires_grad_(index == self.task)
            self.active_banks += 1
        if self.active_banks != 224:
            raise ValueError("Video prior did not bind the active bank on 224 language targets")
        return adapted

    def install_video_prior(self, tokenizer, previous, rank):
        if self.task > 0:
            if previous is None:
                raise FileNotFoundError("Video routing continuation requires the immediately preceding adapter")
            self.prior.restore(previous, self.task - 1)
        self.base.config.mm_text_tower = self.options["text_tower"]
        self.prior.install(self.base, tokenizer, self.options["text_tower"], self.task)

    def after_checkpoint(self, output):
        self.prior.save(output, self.task)

    def checkpoint_scope(self, args):
        from contextlib import nullcontext
        return nullcontext()

    def loss(self, model, inputs, outputs, base_loss):
        return None


def install_inference_prior(model, tokenizer, backbone, method, checkpoint):

    checkpoint = Path(checkpoint)
    settings = json.loads((checkpoint / "adapter_config.json").read_text())
    current = settings["cur_task"]
    prior = VideoPrior(backbone, method, settings["expert_num"], initialization=False)
    prior.restore(checkpoint, current)
    prior.install(model, tokenizer, model.config.mm_text_tower, current)
