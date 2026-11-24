# SPDX-License-Identifier: MIT
# SAME algorithm adapted from PRISM, Copyright (c) LAMDA-CL, MIT license.
# See docs/same_sigma_provenance.md for the precise upstream revision.
"""SAME's routed expert layers and persistent spectral statistics."""
from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
from peft.utils import PeftType
from torch import nn
from torch.nn import functional as F

from ..mcit_expert_lora import MCITExpertLoraConfig, MCITExpertLoraLinear, MCITExpertLoraModel, transpose


def _recomputing() -> bool:
    frame = inspect.currentframe()
    try:
        while frame:
            if frame.f_code.co_name in {"recompute_fn", "_run_fn_with_dynamo_disabled", "unpack"} and "torch/utils/checkpoint" in frame.f_code.co_filename.replace("\\", "/"):
                return True
            frame = frame.f_back
    finally:
        del frame
    return False


def _rank(s, ratio):
    energy = s.float().square()
    if energy.sum().item() <= 1e-12:
        return 0
    return min(s.numel(), int(((energy.cumsum(0) / energy.sum()) <= ratio).sum().item()) + 1)


@dataclass
class SAMEConfig(MCITExpertLoraConfig):
    cur_task: int = 0
    tau_score: float = 0.1
    curvature_mu: float = 0.9
    window_size: int = 3
    max_components: int = 64
    cumulative_energy_ratio: float = 0.9
    counter_schema: str = 'int64-v1'

    def __post_init__(self):
        if self.counter_schema != 'int64-v1':
            raise ValueError(f'unsupported SAME counter schema: {self.counter_schema!r}')
        self.peft_type = PeftType.MOE_LORA_SAME


class SAMEState(nn.ParameterDict):
    """Persistent SAME state; reject legacy floating counters before any dtype cast."""

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # A non-LoRA partial load may contain none of this state. If any SAME
        # state is present, require both exact counters before PyTorch can copy
        # a saturated legacy BF16/FP16 value into an integer destination.
        names = tuple(self.keys())
        if any(prefix + name in state_dict for name in names):
            for name in ('step', 'count'):
                key = prefix + name
                value = state_dict.get(key)
                if (not isinstance(value, torch.Tensor) or value.dtype != torch.int64 or
                        value.ndim != 0 or value.item() < 0):
                    raise RuntimeError(
                        f'{key}: SAME requires a nonnegative scalar int64 counter '
                        '(int64-v1); legacy floating or incomplete counters may be saturated and cannot be resumed or loaded'
                    )
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class SAMELinear(MCITExpertLoraLinear):
    def __init__(self, *args, cur_task=0, tau_score=0.1, curvature_mu=0.9,
                 window_size=3, max_components=64, cumulative_energy_ratio=0.9, **kwargs):
        self.cur_task = int(cur_task)
        self.tau_score = float(tau_score)
        self.curvature_mu = float(curvature_mu)
        self.window_size = int(window_size)
        self.max_components = int(max_components)
        self.cumulative_energy_ratio = float(cumulative_energy_ratio)
        super().__init__(*args, **kwargs)
        n = min(self.max_components, self.in_features)
        # Non-trainable parameters live under the adapter namespace so PEFT 0.4
        # saves them alongside all experts and restores them before continuation.
        state = SAMEState({
            'u': nn.Parameter(torch.zeros(self.in_features, n), requires_grad=False),
            's': nn.Parameter(torch.zeros(n), requires_grad=False),
            'u_prev': nn.Parameter(torch.zeros(self.in_features, n), requires_grad=False),
            's_prev': nn.Parameter(torch.zeros(n), requires_grad=False),
            'valid': nn.Parameter(torch.zeros(()), requires_grad=False),
            'count': nn.Parameter(torch.zeros((), dtype=torch.int64), requires_grad=False),
            'step': nn.Parameter(torch.zeros((), dtype=torch.int64), requires_grad=False),
            'done': nn.Parameter(torch.zeros(()), requires_grad=False),
            'util': nn.Parameter(torch.full((self.expert_num,), 1 / self.expert_num), requires_grad=False),
            'importance': nn.Parameter(torch.full((self.expert_num,), 1 / self.expert_num), requires_grad=False),
            'mask': nn.Parameter(torch.ones(self.expert_num), requires_grad=False),
        })
        self.lora_state = nn.ModuleDict({self.active_adapter: state})
        self.lora_state.to(device=self.weight.device)
        # The request-scoped global prior is runtime state, not a PEFT tensor.
        # Inference must restore a complete prototype sidecar and set it first.
        self.router = None
        self.lora_router[self.active_adapter].weight.register_hook(self._router_grad)
        for expert in self.lora_A[self.active_adapter].loraA:
            expert.weight.register_hook(self._expert_grad)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # PEFT loads adapter dictionaries with strict=False. If this layer's
        # adapter is present, all persistent SAME state must accompany it;
        # otherwise inference could silently use freshly initialized state.
        adapter_prefixes = tuple(prefix + part for part in ('lora_A.', 'lora_B.', 'lora_router.'))
        if any(key.startswith(adapter_prefixes) for key in state_dict):
            for adapter, state in self.lora_state.items():
                missing = [prefix + f'lora_state.{adapter}.{name}' for name in state.keys()
                           if prefix + f'lora_state.{adapter}.{name}' not in state_dict]
                if missing:
                    raise RuntimeError(f'SAME adapter state is incomplete; missing {missing[0]}')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    @property
    def state(self):
        return self.lora_state[self.active_adapter]

    def freeze_state(self):
        for param in self.lora_state.parameters():
            param.requires_grad_(False)

    def _router_grad(self, grad):
        state = self.state
        if not self.training or state['step'].item() < 10:
            return grad
        k = _rank(state['s'], self.cumulative_energy_ratio)
        if not k:
            return grad
        u = state['u'][:, :k].to(grad)
        s = state['s'][:k].to(grad)
        smooth = torch.stack([s[max(0, i - self.window_size + 1):i + 1].mean() for i in range(k)])
        scale = 1 / (smooth + 1e-6)
        scale = (scale / scale.max()).clamp(0.8, 1.2)
        projection = grad @ u
        return grad + ((projection * scale) - projection) @ u.T

    def _expert_grad(self, grad):
        state = self.state
        if not self.training or self.cur_task == 0:
            return grad
        if not state['valid'].item():
            raise RuntimeError('SAME continuation requires a valid prior covariance snapshot')
        k = _rank(state['s_prev'], self.cumulative_energy_ratio)
        if not k:
            raise RuntimeError('SAME covariance snapshot has zero energy')
        u = state['u_prev'][:, :k].to(grad)
        s = state['s_prev'][:k].to(grad)
        parallel = grad @ u
        return (grad - parallel @ u.T) / self.curvature_mu + (parallel / (s + self.curvature_mu)) @ u.T

    @torch.no_grad()
    def _observe(self, x, probs):
        state = self.state
        step = int(state['step'].item())
        if state['count'].item() == 0 or step % 20 == 0:
            centered = x.float() - x.float().mean(0, keepdim=True)
            k = state['s'].numel()
            q = torch.randn(x.shape[-1], k, device=x.device, dtype=torch.float32)
            u, _ = torch.linalg.qr(centered.T @ (centered @ q), mode='reduced')
            s = ((centered @ u).square().sum(0) / max(1, x.shape[0] - 1)).sqrt()
            state['u'].copy_(u)
            state['s'].copy_(s)
        state['count'].add_(x.shape[0])
        state['step'].add_(1)
        p = probs.float()
        state['util'].mul_(0.95).add_(p.mean(0), alpha=0.05)
        importance = (p * x.float().square().sum(-1, keepdim=True)).mean(0)
        state['importance'].mul_(0.95).add_(importance, alpha=0.05)
        if not state['done'].item():
            def norm(v):
                return (v - v.min()) / (v.max() - v.min() + 1e-8)
            scores = norm(state['util']) - norm(state['importance'])
            current = min(self.cur_task, self.expert_num - 1)
            mask = (scores >= self.tau_score).float()
            mask[current] = 1
            state['mask'].copy_(mask)
            if mask.sum().item() == 1:
                state['done'].fill_(1)

    @torch.no_grad()
    def snapshot(self):
        state = self.state
        if _rank(state['s'], self.cumulative_energy_ratio) == 0:
            raise RuntimeError('SAME cannot save a task boundary without nonzero covariance')
        state['u_prev'].copy_(state['u'])
        state['s_prev'].copy_(state['s'])
        state['valid'].fill_(1)

    @torch.no_grad()
    def next_task(self):
        state = self.state
        if not state['valid'].item():
            raise RuntimeError('SAME previous checkpoint has no covariance snapshot')
        state['count'].zero_()
        state['step'].zero_()
        state['done'].zero_()
        state['util'].fill_(1 / self.expert_num)
        state['mask'].fill_(1)

    def merge(self):
        # SAME remains dynamically routed after the generic LLaVA builder's
        # merge_and_unload call; make the newly injected layer an eval layer.
        self.eval()
        self.merged = True

    def set_global_prior(self, prior):
        if (not isinstance(prior, torch.Tensor) or prior.ndim != 1 or
                prior.numel() != self.expert_num or not torch.isfinite(prior).all() or
                (prior < 0).any() or prior.sum() <= 0):
            raise ValueError('SAME global prior must be a finite nonnegative expert vector')
        self.router = prior.detach().to(device=self.weight.device, dtype=torch.float32)

    def forward(self, x, **kwargs):
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), self.bias)
        if self.disable_adapters or self.active_adapter not in self.lora_A:
            return result
        dtype = x.dtype
        x = x.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)
        flat = x.reshape(-1, x.shape[-1])
        logits = self.lora_router[self.active_adapter](flat)
        probs = (logits / (1 if self.training else self.expert_num)).softmax(-1)
        route = probs.mean(0)
        recomputing = self.training and _recomputing()
        if self.training:
            if recomputing:
                mask = self._forward_mask
            else:
                mask = self.state['mask'].detach().clone()
                self._forward_mask = mask
            route = route * mask.to(route)
            route = route / route.sum().clamp_min(1e-8)
        else:
            if self.router is None:
                raise RuntimeError('SAME inference requires a restored global image/text prior')
            # PRISM multiplies by its FP32 global prior, normalizes, and only
            # then retains/renormalizes the two highest per-layer means.
            route = route * self.router.to(device=route.device)
            route = route / (route.sum() + 1e-8)
            if self.expert_num > 2:
                selected = route.topk(2).indices
                mask = torch.zeros_like(route).scatter_(0, selected, 1)
                route = route * mask
                route = route / (route.sum() + 1e-8)
            route = route.to(dtype=x.dtype)
        # PRISM weights the A and B banks separately, retaining cross-expert terms.
        a = sum(route[i] * e(x) for i, e in enumerate(self.lora_A[self.active_adapter].loraA))
        b = sum(route[i] * e(a) for i, e in enumerate(self.lora_B[self.active_adapter].loraB))
        if self.training and not recomputing:
            self._observe(flat.detach(), probs.detach())
        return (result + b * self.scaling[self.active_adapter]).to(dtype)


class SAMEModel(MCITExpertLoraModel):
    def _create_new_module(self, config, adapter_name, target, **context):
        if getattr(self.model, 'is_loaded_in_4bit', False) or getattr(self.model, 'is_loaded_in_8bit', False):
            raise ValueError('SAME currently supports only 16-bit unquantized LLaVA')
        if not isinstance(target, nn.Linear):
            raise ValueError('SAME requires a torch.nn.Linear target')
        return super()._create_new_module(config, adapter_name, target, **context)

    def _linear_class(self):
        return SAMELinear

    def _extra_module_kwargs(self, cfg):
        return {k: getattr(cfg, k) for k in ('cur_task', 'tau_score', 'curvature_mu', 'window_size', 'max_components', 'cumulative_energy_ratio')}

    def add_adapter(self, adapter_name, config=None):
        super().add_adapter(adapter_name, config)
        for layer in self.model.modules():
            if isinstance(layer, SAMELinear):
                layer.freeze_state()
