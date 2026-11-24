# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""PCLR rank pools, progressive consolidation, and task-boundary state.

Independent implementation informed by Meng et al. and the authors' LLaVA code.
Each group shares multimodal keys across all linear projections in one layer.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PoolConfig:
    text_dim: int
    vision_dim: int
    new_rank: int = 64
    top_rank: int = 64
    alpha: float = 128.0
    share_rank: int = 32
    skip_interval: int = 4
    query_weight: float = 0.2

    def __post_init__(self):
        if min(self.text_dim, self.vision_dim, self.new_rank, self.top_rank, self.skip_interval) <= 0:
            raise ValueError("PCLR dimensions, top rank and skip interval must be positive")
        if self.skip_interval < 2 or self.share_rank < 0:
            raise ValueError("invalid compression interval or shared rank")


def mean_query(embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        mean = embeddings.mean(dim=1)
    else:
        mask = mask.to(embeddings.dtype).unsqueeze(-1)
        mean = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return F.normalize(mean, dim=-1)


def distillation_loss(teacher: torch.Tensor, student: torch.Tensor, top_k: int = 100,
                      temperature: float = 2.0) -> torch.Tensor:
    """Upstream's KL of temperature-smoothed teacher top-k probabilities."""
    if temperature <= 0 or top_k <= 0 or teacher.shape != student.shape:
        raise ValueError("incompatible logits or distillation settings")
    indices = teacher.detach().topk(min(top_k, teacher.shape[-1]), dim=-1).indices
    t = teacher.detach().gather(-1, indices).float()
    s = student.gather(-1, indices).float()
    # softmax(logits) ** (1/T), normalized, equals softmax(logits/T).
    p = F.softmax(t / temperature, dim=-1)
    return (p * (F.log_softmax(t / temperature, -1) - F.log_softmax(s / temperature, -1))).sum(-1).mean()


def retained_indices(n: int, skip: int, device=None) -> torch.Tensor:
    if n < 0 or skip < 2:
        raise ValueError("invalid rank count or compression interval")
    i = torch.arange(n, device=device)
    return i[(i + 1) % skip != 0]


class Keys(nn.Module):
    def __init__(self, config: PoolConfig, static: int, train: int, device, dtype):
        super().__init__()
        self.config = config
        for kind, count, grad in (("static", static, False), ("new", train, True)):
            for modality, dim in (("text", config.text_dim), ("vision", config.vision_dim)):
                value = F.normalize(torch.randn(count, dim, device=device, dtype=dtype), dim=-1)
                setattr(self, f"{kind}_{modality}", nn.Parameter(value, requires_grad=grad))

    def scores(self, text: torch.Tensor, vision: torch.Tensor, part: str) -> torch.Tensor:
        kt = getattr(self, f"{part}_text")
        kv = getattr(self, f"{part}_vision")
        return text.to(kt.dtype) @ F.normalize(kt, dim=-1).T + (
            self.config.vision_dim / self.config.text_dim
        ) * (vision.to(kv.dtype) @ F.normalize(kv, dim=-1).T)

    def select(self, text: torch.Tensor, vision: torch.Tensor, part: str, count: int) -> torch.Tensor:
        scores = self.scores(text, vision, part)
        return scores.sum(0).topk(min(count, scores.shape[1])).indices


class RankLinear(nn.Module):
    def __init__(self, base: nn.Linear, config: PoolConfig, static: int, train: int, shared: int,
                 group: str, runtime):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.config, self.group, self.runtime = config, group, runtime
        d, o = base.in_features, base.out_features
        device, dtype = base.weight.device, base.weight.dtype
        for name, count, grad in (("static", static, False), ("new", train, True),
                                  ("shared", shared, False)):
            a = torch.empty(d, count, device=device, dtype=dtype)
            nn.init.normal_(a, std=1 / math.sqrt(d))
            b = torch.zeros(count, o, device=device, dtype=dtype)
            setattr(self, f"{name}_a", nn.Parameter(a, requires_grad=grad))
            setattr(self, f"{name}_b", nn.Parameter(b, requires_grad=grad))

    def forward(self, x):
        result = self.base(x)
        state = self.runtime
        if state.text is None or state.vision is None:
            raise RuntimeError("PCLR routing query was not established for this request")
        if state.teacher:
            old = state.teacher_layers[self.group][id(self)]
            kt, kv = state.teacher_keys[self.group]
            shared_a, shared_b, a, b = old
            if shared_a.shape[1]:
                result = result + ((x @ shared_a) @ shared_b) * (self.config.alpha / shared_a.shape[1])
            parts = ((state.split_at, a.shape[1]), (0, state.split_at)) if state.part == "new" else ((0, state.split_at),)
            remaining = self.config.top_rank
            for lo, hi in parts:
                if hi <= lo or remaining <= 0:
                    continue
                scores = state.text.to(kt.dtype) @ F.normalize(kt[lo:hi], dim=-1).T
                scores += (self.config.vision_dim / self.config.text_dim) * state.vision.to(kv.dtype) @ F.normalize(kv[lo:hi], dim=-1).T
                idx = lo + scores.sum(0).topk(min(remaining, hi - lo)).indices
                result = result + ((x @ a[:, idx]) @ b[idx]) * (self.config.alpha / self.config.top_rank)
                remaining -= len(idx)
            return result
        keys = state.groups[self.group]
        if self.shared_a.shape[1]:
            result = result + ((x @ self.shared_a) @ self.shared_b) * (self.config.alpha / self.shared_a.shape[1])
        # During learning and latest-task generation, new ranks take priority.
        remaining = self.config.top_rank
        parts = ("new", "static") if state.part == "new" else ("static",)
        for part in parts:
            a, b = getattr(self, f"{part}_a"), getattr(self, f"{part}_b")
            if not a.shape[1] or remaining <= 0:
                continue
            chosen = keys.select(state.text, state.vision, part, remaining)
            result = result + ((x @ a[:, chosen]) @ b[chosen]) * (self.config.alpha / self.config.top_rank)
            remaining -= len(chosen)
        return result


class Runtime:
    def __init__(self):
        self.text = self.vision = None
        self.part = "new"
        self.groups = None
        self.teacher = False
        self.teacher_keys = {}
        self.teacher_layers = {}
        self.split_at = 0


class RankPool(nn.Module):
    """Adapter around target projections. Base weights remain frozen."""
    def __init__(self, base: nn.Module, targets: Mapping[str, list[str]], config: PoolConfig,
                 static: int = 0, train: int | None = None, shared: int = 0):
        super().__init__()
        self.base = base
        self.config = config
        self.targets = {k: list(v) for k, v in targets.items()}
        self.runtime = Runtime()
        self.groups = nn.ModuleDict()
        self.runtime.groups = self.groups
        self.register_buffer("task_text", torch.empty(0, config.text_dim))
        self.register_buffer("task_vision", torch.empty(0, config.vision_dim))
        train = config.new_rank if train is None else train
        self.base.requires_grad_(False)
        for i, (_, names) in enumerate(self.targets.items()):
            if not names:
                raise ValueError("empty rank pool group")
            group = f"g{i}"
            first = base.get_submodule(names[0])
            self.groups[group] = Keys(config, static, train, first.weight.device, first.weight.dtype)
            for name in names:
                parent_name, _, leaf = name.rpartition(".")
                parent = base.get_submodule(parent_name) if parent_name else base
                old = getattr(parent, leaf)
                if not isinstance(old, nn.Linear):
                    raise TypeError(f"{name} is not a linear module")
                setattr(parent, leaf, RankLinear(old, config, static, train, shared, group, self.runtime))
        self.task_text = self.task_text.to(device=next(base.parameters()).device, dtype=next(base.parameters()).dtype)
        self.task_vision = self.task_vision.to(device=self.task_text.device, dtype=self.task_text.dtype)

    def layers(self):
        return ((n, m) for n, m in self.base.named_modules() if isinstance(m, RankLinear))

    def set_query(self, text, vision, part="new"):
        if part not in ("new", "static") or text.shape[0] != vision.shape[0]:
            raise ValueError("invalid routing query")
        self.runtime.text, self.runtime.vision, self.runtime.part = text.detach(), vision.detach(), part

    def alignment_loss(self):
        q, v = self.runtime.text, self.runtime.vision
        if q is None or v is None:
            raise RuntimeError("query is required")
        loss = []
        for keys in self.groups.values():
            if keys.new_text.shape[0]:
                t = q.to(keys.new_text.dtype) @ F.normalize(keys.new_text, dim=-1).T
                z = v.to(keys.new_vision.dtype) @ F.normalize(keys.new_vision, dim=-1).T
                image_present = v.norm(dim=-1, keepdim=True).ne(0)
                loss.append((1 - t).mean() + ((1 - z) * image_present).mean())
        return torch.stack(loss).mean() if loss else q.new_zeros(())

    @torch.no_grad()
    def init_new_keys(self, text, vision, initialize_values=False):
        for keys in self.groups.values():
            keys.new_text.copy_(text.reshape(1, -1).expand_as(keys.new_text))
            keys.new_vision.copy_(vision.reshape(1, -1).expand_as(keys.new_vision))
        if initialize_values:
            for _, layer in self.layers():
                keys = self.groups[layer.group]
                idx = keys.select(text.reshape(1, -1), vision.reshape(1, -1), "static", layer.new_a.shape[1])
                if len(idx) != layer.new_a.shape[1]:
                    raise ValueError("not enough static ranks for nearest-value initialization")
                layer.new_a.copy_(layer.static_a[:, idx])
                layer.new_b.copy_(layer.static_b[idx])

    @torch.no_grad()
    def append_task_query(self):
        text = F.normalize(torch.cat([g.new_text for g in self.groups.values()]).mean(0), dim=0)
        vision = F.normalize(torch.cat([g.new_vision for g in self.groups.values()]).mean(0), dim=0)
        self.task_text = torch.cat((self.task_text, text[None]), 0)
        self.task_vision = torch.cat((self.task_vision, vision[None]), 0)

    @torch.no_grad()
    def merge_new(self):
        for keys in self.groups.values():
            for modality in ("text", "vision"):
                setattr(keys, f"static_{modality}", nn.Parameter(torch.cat((getattr(keys, f"static_{modality}"),
                    getattr(keys, f"new_{modality}")), 0), requires_grad=False))
                setattr(keys, f"new_{modality}", nn.Parameter(getattr(keys, f"new_{modality}")[:0], requires_grad=False))
        for _, layer in self.layers():
            layer.static_a = nn.Parameter(torch.cat((layer.static_a, layer.new_a), 1), requires_grad=False)
            layer.static_b = nn.Parameter(torch.cat((layer.static_b, layer.new_b), 0), requires_grad=False)
            layer.new_a = nn.Parameter(layer.new_a[:, :0], requires_grad=False)
            layer.new_b = nn.Parameter(layer.new_b[:0], requires_grad=False)

    @torch.no_grad()
    def compress(self, split_at: int):
        """Drop every interval-th vector in each teacher part, initialize student keys.

        Student keys receive the task queries selecting them and stay frozen; selected
        student values and shared values remain trainable for logit distillation.
        """
        self.runtime.split_at = split_at
        self.runtime.teacher_keys = {name: (keys.static_text.detach().clone(), keys.static_vision.detach().clone())
                                     for name, keys in self.groups.items()}
        self.runtime.teacher_layers = {name: {} for name in self.groups}
        for _, layer in self.layers():
            self.runtime.teacher_layers[layer.group][id(layer)] = tuple(x.detach().clone() for x in
                (layer.shared_a, layer.shared_b, layer.static_a, layer.static_b))
        for group, keys in self.groups.items():
            n = keys.static_text.shape[0]
            if not 0 < split_at < n:
                raise ValueError("compression needs nonempty old and newest teacher pools")
            idx = torch.cat((retained_indices(split_at, self.config.skip_interval, keys.static_text.device),
                             split_at + retained_indices(n - split_at, self.config.skip_interval, keys.static_text.device)))
            if len(idx) < self.config.top_rank:
                raise ValueError("compressed pool is smaller than routing top-k")
            kt = F.normalize(keys.static_text[idx].clone(), dim=-1)
            kv = F.normalize(keys.static_vision[idx].clone(), dim=-1)
            scores = self.task_text.to(kt.dtype) @ kt.T + (self.config.vision_dim / self.config.text_dim) * self.task_vision.to(kv.dtype) @ kv.T
            selected = scores.topk(self.config.top_rank, dim=1).indices
            for task, chosen in enumerate(selected):
                kt[chosen] += self.task_text[task].to(kt.dtype)
                kv[chosen] += self.task_vision[task].to(kv.dtype)
            keys.static_text = nn.Parameter(F.normalize(kt, dim=-1), requires_grad=False)
            keys.static_vision = nn.Parameter(F.normalize(kv, dim=-1), requires_grad=False)
            for _, layer in self.layers():
                if layer.group != group:
                    continue
                layer.static_a = nn.Parameter(layer.static_a[:, idx].clone(), requires_grad=True)
                layer.static_b = nn.Parameter(layer.static_b[idx].clone(), requires_grad=True)
                if not layer.shared_a.shape[1] and self.config.share_rank:
                    d, o = layer.base.in_features, layer.base.out_features
                    layer.shared_a = nn.Parameter(torch.randn(d, self.config.share_rank, device=layer.base.weight.device,
                                                            dtype=layer.base.weight.dtype) / math.sqrt(d))
                    layer.shared_b = nn.Parameter(torch.zeros(self.config.share_rank, o, device=layer.base.weight.device,
                                                              dtype=layer.base.weight.dtype))
                else:
                    layer.shared_a.requires_grad_(True)
                    layer.shared_b.requires_grad_(True)

    def adapter_state(self):
        names = ("static_", "new_", "shared_", "task_text", "task_vision")
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if any(x in k for x in names) or ("mm_projector" in k and ".base." in k)}

    def save_boundary(self, path: str | Path, phase: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "mcitlib-pclr-v1", "config": asdict(self.config), "targets": self.targets,
                    "phase": phase, "state": self.adapter_state()}, path)

    @classmethod
    def load_boundary(cls, base: nn.Module, path: str | Path):
        data = torch.load(path, map_location="cpu")
        if data.get("format") != "mcitlib-pclr-v1" or not data.get("targets"):
            raise ValueError("invalid PCLR boundary checkpoint")
        state = data["state"]
        static = state["groups.g0.static_text"].shape[0]
        train = state["groups.g0.new_text"].shape[0]
        shared_keys = [key for key in state if key.endswith(".shared_a")]
        if not shared_keys:
            raise ValueError("PCLR checkpoint has no rank vectors")
        shared = state[shared_keys[0]].shape[1]
        model = cls(base, data["targets"], PoolConfig(**data["config"]), static, train, shared)
        model.task_text = state["task_text"].to(model.task_text)
        model.task_vision = state["task_vision"].to(model.task_vision)
        expected = set(model.adapter_state())
        if expected != set(state):
            raise ValueError(f"PCLR checkpoint adapter keys differ: {sorted(expected ^ set(state))}")
        model.load_state_dict(state, strict=False)
        return model, data["phase"]

    @torch.no_grad()
    def begin_task(self, text, vision):
        if any(g.new_text.shape[0] for g in self.groups.values()):
            self.merge_new()
        rank = self.config.new_rank
        for keys in self.groups.values():
            for modality in ("text", "vision"):
                old = getattr(keys, f"static_{modality}")
                setattr(keys, f"new_{modality}", nn.Parameter(old.new_empty(rank, old.shape[1])))
        for _, layer in self.layers():
            a = layer.static_a.new_empty(layer.base.in_features, rank)
            nn.init.normal_(a, std=1 / math.sqrt(layer.base.in_features))
            layer.new_a = nn.Parameter(a)
            layer.new_b = nn.Parameter(layer.static_b.new_zeros(rank, layer.base.out_features))
            layer.static_a.requires_grad_(False)
            layer.static_b.requires_grad_(False)
            layer.shared_a.requires_grad_(False)
            layer.shared_b.requires_grad_(False)
        self.init_new_keys(text, vision, initialize_values=bool(next(iter(self.groups.values())).static_text.shape[0] and
                                                                 next(self.layers())[1].shared_a.shape[1]))
