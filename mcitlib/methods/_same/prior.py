# SPDX-License-Identifier: MIT
# Adapted from PRISM RouterIntegration, Copyright (c) LAMDA-CL.
"""SAME's image/text prototype prior for LLaVA training and generation.

The small prototype state is intentionally external to the model module tree:
it remains FP32 across model precision casts and is saved in an explicit,
versioned sidecar. PRISM creates FP32 non-trainable ParameterLists after model
precision setup, but an additional later dtype cast would change their dtype;
preserving FP32 here avoids low-precision boundary-counter saturation.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

SIDECAR = 'same_prior_state.pt'
SCHEMA = 'same-prior-v1'
FEATURE_DIM = 768
MIXTURE_LOGIT_SCALE = 28.0
IMAGE_TOKEN_INDEX = -200


def clip_snapshot_identity(directory: str | Path):
    """Identify the local HF CLIP snapshot by the content actually selected.

    No absolute path is included, so a copied snapshot can be used elsewhere.
    Hashing the selected weight file(s) incurs a full read on first bind.
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f'SAME CLIP snapshot directory is missing: {directory}')
    from transformers.utils import is_safetensors_available

    names = {'config.json', 'preprocessor_config.json'}
    names.update(name for name in ('tokenizer.json', 'tokenizer_config.json',
                                   'special_tokens_map.json', 'added_tokens.json',
                                   'vocab.json', 'merges.txt') if (directory / name).is_file())
    if 'tokenizer.json' not in names and not {'vocab.json', 'merges.txt'} <= names:
        raise FileNotFoundError('SAME CLIP snapshot lacks tokenizer.json or vocab.json/merges.txt')
    # transformers' from_pretrained prefers safetensors when supported, then
    # its sharded index, otherwise the corresponding PyTorch files.
    formats = [('model.safetensors', 'model.safetensors.index.json')] if is_safetensors_available() else []
    formats.append(('pytorch_model.bin', 'pytorch_model.bin.index.json'))
    weight = next(((single, index) for single, index in formats
                   if (directory / single).is_file() or (directory / index).is_file()), None)
    if weight is None:
        raise FileNotFoundError('SAME CLIP snapshot has no supported model weight file or index')
    single, index = weight
    if (directory / single).is_file():
        names.add(single)
    else:
        names.add(index)
        with (directory / index).open(encoding='utf8') as handle:
            weight_map = json.load(handle).get('weight_map')
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError('SAME CLIP weight index has no weight_map')
        names.update(weight_map.values())
    hashes = {}
    for name in sorted(names):
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise ValueError('invalid file name in SAME CLIP snapshot')
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise FileNotFoundError(f'SAME CLIP snapshot file is missing or outside its directory: {name}')
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    canonical = json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode('utf8')
    return {'schema': 'same-clip-sha256-v1', 'files': hashes,
            'sha256': hashlib.sha256(canonical).hexdigest()}


class PriorState:
    """Per-rank running image/text anchor means and FP32 sample counts."""

    def __init__(self, expert_num: int, feature_dim: int = FEATURE_DIM, *, device=None):
        if expert_num < 1 or feature_dim < 1:
            raise ValueError('SAME prior dimensions must be positive')
        self.expert_num, self.feature_dim = int(expert_num), int(feature_dim)
        # Match PRISM's RNG draw order: all image anchors, then all text anchors.
        self.image_anchors = torch.cat([0.1 * torch.randn(1, feature_dim) for _ in range(expert_num)])
        self.text_anchors = torch.cat([0.1 * torch.randn(1, feature_dim) for _ in range(expert_num)])
        self.image_boundary = torch.ones(expert_num, dtype=torch.float32)
        self.text_boundary = torch.ones(expert_num, dtype=torch.float32)
        self.prior = None
        self.completed_task = -1
        self.clip_identity = None
        self._bound_clip_path = None
        self._synchronized = False
        if device is not None:
            self.to(device)

    def to(self, device):
        for key in ('image_anchors', 'text_anchors', 'image_boundary', 'text_boundary'):
            setattr(self, key, getattr(self, key).to(device=device, dtype=torch.float32))
        if self.prior is not None:
            self.prior = self.prior.to(device=device, dtype=torch.float32)
        return self

    @torch.no_grad()
    def update(self, image: torch.Tensor | None, text: torch.Tensor, task_id: int):
        if not 0 <= task_id < self.expert_num:
            raise ValueError(f'invalid SAME task index: {task_id}')
        if text.ndim == 1:
            text = text.unsqueeze(0)
        if text.ndim != 2 or text.shape[1] != self.feature_dim:
            raise ValueError('SAME text feature shape does not match the anchors')
        batch = int(text.shape[0])
        if batch < 1:
            raise ValueError('SAME cannot update anchors from an empty batch')
        self.to(text.device)
        # PRISM uses text batch size for both boundary updates.
        if image is not None:
            if image.ndim == 1:
                image = image.unsqueeze(0)
            if image.ndim != 2 or image.shape != text.shape:
                raise ValueError('SAME image/text feature shapes must match')
            n = self.image_boundary[task_id] + batch
            self.image_anchors[task_id].copy_(
                (self.image_anchors[task_id] * self.image_boundary[task_id] + image.sum(dim=0)) / n)
            self.image_boundary[task_id].copy_(n)
        n = self.text_boundary[task_id] + batch
        self.text_anchors[task_id].copy_(
            (self.text_anchors[task_id] * self.text_boundary[task_id] + text.sum(dim=0)) / n)
        self.text_boundary[task_id].copy_(n)

    def mixture(self, image: torch.Tensor | None, text: torch.Tensor):
        if text.ndim == 1:
            text = text.unsqueeze(0)
        self.to(text.device)
        text_sims = torch.stack([
            F.cosine_similarity(text.unsqueeze(1), a.unsqueeze(0), dim=2).max()
            for a in self.text_anchors.split(1)
        ]).to(device=text.device, dtype=torch.float32)
        if image is not None:
            if image.ndim == 1:
                image = image.unsqueeze(0)
            image_sims = torch.stack([
                F.cosine_similarity(image.unsqueeze(1), a.unsqueeze(0), dim=2).max()
                for a in self.image_anchors.split(1)
            ]).to(device=text.device, dtype=torch.float32)
            text_sims = 0.2 * image_sims + 0.8 * text_sims
        return F.softmax(text_sims.reshape(-1) * MIXTURE_LOGIT_SCALE, dim=0)

    def payload(self):
        self.require_observed_through(self.completed_task)
        if self.clip_identity is None:
            raise ValueError('SAME prior cannot be saved without a verified CLIP snapshot identity')
        result = {'schema': SCHEMA, 'expert_num': self.expert_num, 'feature_dim': self.feature_dim,
                  'completed_task': self.completed_task, 'clip_identity': self.clip_identity}
        for key in ('image_anchors', 'text_anchors', 'image_boundary', 'text_boundary', 'prior'):
            value = getattr(self, key)
            result[key] = None if value is None else value.detach().to(device='cpu', dtype=torch.float32).clone()
        return result

    def bind_clip(self, directory: str | Path):
        directory = Path(directory).resolve()
        if self._bound_clip_path == directory and self.clip_identity is not None:
            return
        identity = clip_snapshot_identity(directory)
        if self.clip_identity is not None and self.clip_identity != identity:
            raise ValueError('SAME checkpoint CLIP snapshot content does not match the configured tower')
        self.clip_identity = identity
        self._bound_clip_path = directory

    def complete_task(self, task_id: int):
        if task_id != self.completed_task + 1 or not 0 <= task_id < self.expert_num:
            raise ValueError('SAME prior task completion must follow the previous completed task')
        if (self.text_boundary[:task_id + 1] <= 1).any():
            raise ValueError('SAME prior is missing observed text anchors for a completed task')
        if ((self.text_boundary[task_id + 1:] != 1).any() or
                (self.image_boundary[task_id + 1:] != 1).any()):
            raise ValueError('SAME prior has observations for tasks after the completed task')
        self.completed_task = task_id

    def synchronize_once(self):
        """Match the initial rank-0 parameter broadcast in data parallel runs.

        There is deliberately no per-step collective: PRISM's prototype
        updates are process-local and its checkpoint saves rank 0's state.
        """
        if self._synchronized:
            return
        import torch.distributed as dist

        world_env = int(os.environ.get('WORLD_SIZE', '1'))
        if not dist.is_available() or not dist.is_initialized():
            if world_env > 1:
                raise RuntimeError('SAME prior requires an initialized data-parallel group before training')
            self._synchronized = True
            return
        for name in ('image_anchors', 'text_anchors', 'image_boundary', 'text_boundary'):
            dist.broadcast(getattr(self, name), src=0)
        present = torch.tensor([int(self.prior is not None)], device=self.image_anchors.device, dtype=torch.int64)
        dist.broadcast(present, src=0)
        if present.item():
            if self.prior is None:
                self.prior = torch.zeros(self.expert_num, device=self.image_anchors.device, dtype=torch.float32)
            dist.broadcast(self.prior, src=0)
        else:
            self.prior = None
        self._synchronized = True

    def restore(self, payload: dict[str, Any]):
        if not isinstance(payload, dict) or payload.get('schema') != SCHEMA:
            raise ValueError('SAME prior sidecar has an unsupported or missing schema')
        if (payload.get('expert_num'), payload.get('feature_dim')) != (self.expert_num, self.feature_dim):
            raise ValueError('SAME prior sidecar dimensions do not match the adapter')
        completed = payload.get('completed_task')
        if not isinstance(completed, int) or isinstance(completed, bool) or not 0 <= completed < self.expert_num:
            raise ValueError('SAME prior sidecar has no valid completed task identity')
        identity = payload.get('clip_identity')
        if (not isinstance(identity, dict) or identity.get('schema') != 'same-clip-sha256-v1' or
                not isinstance(identity.get('files'), dict) or not identity['files'] or
                not isinstance(identity.get('sha256'), str)):
            raise ValueError('SAME prior sidecar has no valid CLIP snapshot identity')
        canonical = json.dumps(identity['files'], sort_keys=True, separators=(',', ':')).encode('utf8')
        if hashlib.sha256(canonical).hexdigest() != identity['sha256']:
            raise ValueError('SAME prior sidecar CLIP identity is invalid')
        if self.clip_identity is not None and self.clip_identity != identity:
            raise ValueError('SAME checkpoint CLIP snapshot content does not match the configured tower')
        shapes = {'image_anchors': (self.expert_num, self.feature_dim),
                  'text_anchors': (self.expert_num, self.feature_dim),
                  'image_boundary': (self.expert_num,), 'text_boundary': (self.expert_num,)}
        checked = {}
        for name, shape in shapes.items():
            value = payload.get(name)
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or
                    tuple(value.shape) != shape or not torch.isfinite(value).all()):
                raise ValueError(f'invalid SAME prior state: {name}')
            if name.endswith('boundary') and (value < 1).any():
                raise ValueError(f'invalid SAME prior boundary: {name}')
            checked[name] = value.detach().clone()
        if (checked['text_boundary'][:completed + 1] <= 1).any():
            raise ValueError('SAME prior is missing observed text anchors for a completed task')
        if ((checked['text_boundary'][completed + 1:] != 1).any() or
                (checked['image_boundary'][completed + 1:] != 1).any()):
            raise ValueError('SAME prior has observations for tasks after the completed task')
        prior = payload.get('prior')
        if prior is not None:
            if (not isinstance(prior, torch.Tensor) or prior.dtype != torch.float32 or
                    tuple(prior.shape) != (self.expert_num,) or not torch.isfinite(prior).all() or
                    (prior < 0).any() or prior.sum() <= 0):
                raise ValueError('invalid SAME cached prior')
            prior = prior.detach().clone()
        device = self.image_anchors.device
        for name, value in checked.items():
            setattr(self, name, value.to(device))
        self.prior = None if prior is None else prior.to(device)
        self.completed_task = completed
        self.clip_identity = identity

    def save(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / SIDECAR
        temporary = directory / f'.{SIDECAR}.{os.getpid()}.tmp'
        try:
            torch.save(self.payload(), temporary)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target

    def load(self, directory: str | Path):
        target = Path(directory) / SIDECAR
        if not target.is_file():
            raise FileNotFoundError(
                f'SAME checkpoint lacks {SIDECAR}: {target}. Its historical image/text anchors '
                'cannot be reconstructed from the old adapter; use the frozen legacy source for old results.')
        self.restore(torch.load(target, map_location='cpu', weights_only=True))
        return target

    def require_observed_through(self, task_id: int):
        if not 0 <= task_id < self.expert_num:
            raise ValueError(f'invalid SAME checkpoint task index: {task_id}')
        if self.completed_task != task_id:
            raise ValueError('SAME prior completed task does not match the checkpoint/continuation task')
        if (self.text_boundary[:task_id + 1] <= 1).any():
            raise ValueError('SAME prior is missing observed text anchors for a completed task')
        if ((self.text_boundary[task_id + 1:] != 1).any() or
                (self.image_boundary[task_id + 1:] != 1).any()):
            raise ValueError('SAME prior has observations for tasks after the completed task')


def _module_device_dtype(module):
    parameter = next(module.parameters())
    return parameter.device, parameter.dtype


class PriorRuntime:
    """Input lifecycle and CLIP features for a single model process/rank."""

    def __init__(self, state: PriorState, tokenizer, clip_tokenizer, text_tower, image_tower, task_id: int):
        self.state = state
        self.tokenizer, self.clip_tokenizer = tokenizer, clip_tokenizer
        self.text_tower, self.image_tower = text_tower, image_tower
        self.task_id = int(task_id)

    def _extract(self, input_ids: torch.Tensor, images):
        device, text_dtype = _module_device_dtype(self.text_tower)
        image_features = None
        if images is not None and self.image_tower is not None:
            try:
                image_device, image_dtype = _module_device_dtype(self.image_tower)
                with torch.no_grad():
                    if isinstance(images, list):
                        outputs = [self.image_tower(pixel_values=image.to(image_device, image_dtype).unsqueeze(0))
                                   .image_embeds.to(image.dtype) for image in images]
                        image_features = torch.cat(outputs, dim=0)
                    else:
                        image_features = self.image_tower(pixel_values=images.to(image_device, image_dtype))\
                            .image_embeds.to(images.dtype)
            except RuntimeError:
                # PRISM explicitly falls back to the text feature only.
                image_features = None
        pad = self.tokenizer.pad_token_id if self.tokenizer is not None else self.clip_tokenizer.pad_token_id
        token_ids = input_ids.detach().cpu().clone()
        token_ids[token_ids == IMAGE_TOKEN_INDEX] = pad
        tok = self.tokenizer or self.clip_tokenizer
        decoded = tok.batch_decode(token_ids.numpy(), skip_special_tokens=True)
        decoded = ['\n'.join(value.split('\n')[1:]).split(' ASSISTANT')[0] for value in decoded]
        clip_inputs = self.clip_tokenizer(decoded, padding='longest', max_length=77,
                                          truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            text = self.text_tower(**clip_inputs).pooler_output.to(text_dtype)
        if text.ndim == 1:
            text = text.unsqueeze(0)
        return image_features, text

    def _write(self, model, prior):
        from ...tuners.variants.same import SAMELinear

        count = 0
        for layer in model.modules():
            if isinstance(layer, SAMELinear):
                layer.set_global_prior(prior)
                count += 1
        if count == 0:
            raise RuntimeError('SAME prior runtime found no routed SAME layers')
        self.state.prior = prior.detach().to(dtype=torch.float32).clone()

    def prepare(self, model, input_ids, images, *, training: bool, grad_enabled: bool):
        if self.clip_tokenizer is None or self.text_tower is None:
            return False
        # Upstream towers are children of the model and inherit train/eval
        # mode even though their parameters are frozen and features use no_grad.
        self.text_tower.train(training)
        if self.image_tower is not None:
            self.image_tower.train(training)
        if input_ids is None or (hasattr(input_ids, 'shape') and input_ids.shape[1] <= 1):
            if self.state.prior is not None:
                self._write(model, self.state.prior)
            return False
        image, text = self._extract(input_ids, images)
        if training and grad_enabled and 0 <= self.task_id < self.state.expert_num:
            self.state.update(image, text, self.task_id)
            prior = torch.zeros(self.state.expert_num, device=text.device, dtype=torch.float32)
            prior[self.task_id] = 1.0
        else:
            prior = self.state.mixture(image, text)
        self._write(model, prior)
        return True

    def install_training_hook(self, model):
        def before_forward(module, args, kwargs):
            # All data-parallel ranks enter this before any data-dependent
            # image/token/gradient branch, preventing asymmetric collectives.
            if module.training:
                self.state.synchronize_once()
            input_ids = kwargs.get('input_ids', args[0] if args else None)
            images = kwargs.get('images')
            if kwargs.get('inputs_embeds') is None and images is not None:
                self.prepare(module, input_ids, images, training=module.training,
                             grad_enabled=torch.is_grad_enabled())
        return model.register_forward_pre_hook(before_forward, with_kwargs=True)

    def install_generate_hook(self, model):
        original = model.generate
        if getattr(original, '_same_prior_hook', False):
            raise RuntimeError('SAME generation prior hook is already installed')

        @functools.wraps(original)
        def generate(*args, **kwargs):
            input_ids = kwargs.get('input_ids', kwargs.get('inputs', args[0] if args else None))
            self.prepare(model, input_ids, kwargs.get('images'), training=False, grad_enabled=False)
            return original(*args, **kwargs)

        generate._same_prior_hook = True
        model.generate = generate
        return model


def load_clip_runtime(state: PriorState, tokenizer, clip_path: str, *, device, dtype, task_id: int):
    if not clip_path or not Path(clip_path).is_dir():
        raise FileNotFoundError(f'SAME requires a local CLIP routing/text tower directory: {clip_path!r}')
    state.bind_clip(clip_path)
    from transformers import AutoTokenizer, CLIPTextModel, CLIPVisionModelWithProjection

    clip_tokenizer = AutoTokenizer.from_pretrained(clip_path, use_fast=True, local_files_only=True)
    text = CLIPTextModel.from_pretrained(clip_path, local_files_only=True).to(device=device, dtype=dtype)
    vision = CLIPVisionModelWithProjection.from_pretrained(clip_path, local_files_only=True).to(device=device, dtype=dtype)
    text.requires_grad_(False).eval()
    vision.requires_grad_(False).eval()
    if text.config.hidden_size != state.feature_dim or vision.config.projection_dim != state.feature_dim:
        raise ValueError('SAME CLIP feature dimensions do not match the checkpoint prior')
    state.to(device)
    return PriorRuntime(state, tokenizer, clip_tokenizer, text, vision, task_id)
