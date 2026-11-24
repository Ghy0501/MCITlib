# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""LLaVA routing, supervision, and CIL training for PCLR."""
from __future__ import annotations

import json
import os
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .core import PoolConfig, RankPool, distillation_loss, mean_query


def keep_norms_fp32(base):
    """Keep the release's FP32 norm parameters without moving a dispatched model."""
    for name, module in base.named_modules():
        if 'norm' in name:
            module.to(dtype=torch.float32)


def prepare_base_precision(base, device, dtype):
    """Match the release: mixed-precision base with FP32 norm modules."""
    base.to(device=device, dtype=dtype)
    keep_norms_fp32(base)


def model_autocast(base):
    """Use the model compute dtype even outside Trainer's autocast scope.

    Query initialization and raw generation run outside Trainer. CLIP's FP32
    layer norms require autocast when the remaining tower is BF16 or FP16.
    A CPU BF16 context also makes the small precision regression portable.
    """
    weight = base.get_model().embed_tokens.weight
    if weight.device.type == 'cuda' and weight.dtype in (torch.bfloat16, torch.float16):
        return torch.autocast(device_type='cuda', dtype=weight.dtype)
    if weight.device.type == 'cpu' and weight.dtype == torch.bfloat16:
        return torch.autocast(device_type='cpu', dtype=weight.dtype)
    return nullcontext()


def target_groups(model):
    result = {}
    for name, layer in list(model.named_modules()):
        if not isinstance(layer, nn.Linear) or any(x in name for x in ("vision_tower", "vision_resampler", "lm_head")):
            continue
        pieces = name.split('.')
        if 'mm_projector' in pieces:
            key = 'mm_projector'
        elif 'layers' in pieces:
            key = f"layer_{pieces[pieces.index('layers') + 1]}"
        else:
            raise ValueError(f"unrecognized LLaVA projection for PCLR: {name}")
        result.setdefault(key, []).append(name)
    if not result:
        raise ValueError("no LLaVA projections found for PCLR")
    return result


def queries(base, instruction_ids, instruction_mask, images, has_image=None):
    with torch.no_grad(), model_autocast(base):
        text = mean_query(base.get_model().embed_tokens(instruction_ids), instruction_mask)
        tower = base.get_vision_tower()
        if images is None:
            vision = text.new_zeros((text.shape[0], tower.hidden_size))
        else:
            if not isinstance(images, torch.Tensor) or images.ndim != 4:
                raise ValueError("PCLR currently supports one image per example")
            original = tower.select_feature
            try:
                tower.select_feature = 'cls_patch'
                vision = F.normalize(tower(images)[:, 0], dim=-1).to(text)
            finally:
                tower.select_feature = original
            if has_image is not None:
                vision = vision * torch.as_tensor(has_image, device=vision.device, dtype=vision.dtype)[:, None]
    return text.detach(), vision.detach()


def make_data(tokenizer, data_args):
    from ...backbones.llava.standard_train import DataCollatorForSupervisedDataset, LazySupervisedDataset

    class Dataset(LazySupervisedDataset):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            row = self.list_data_dict[index]
            first = next(x['value'] for x in row['conversations'] if x['from'] == 'human')
            first = first.replace('<image>', '').strip()
            item['instruction_ids'] = tokenizer(first, return_tensors='pt', truncation=True,
                                                max_length=tokenizer.model_max_length).input_ids[0]
            item['has_image'] = 'image' in row
            return item

    class Collator(DataCollatorForSupervisedDataset):
        def __call__(self, instances):
            batch = super().__call__(instances)
            ids = nn.utils.rnn.pad_sequence([row['instruction_ids'] for row in instances], batch_first=True,
                                            padding_value=tokenizer.pad_token_id)
            batch['instruction_ids'] = ids
            batch['instruction_mask'] = ids.ne(tokenizer.pad_token_id)
            batch['has_image'] = [row['has_image'] for row in instances]
            return batch

    dataset = Dataset(data_path=data_args.data_path, tokenizer=tokenizer, data_args=data_args)
    return {'train_dataset': dataset, 'eval_dataset': None, 'data_collator': Collator(tokenizer=tokenizer)}


@torch.no_grad()
def task_initial_query(base, data, batch_size, device):
    """Deterministic full-task mean; the release's precomputed key files are unavailable."""
    loader = DataLoader(data['train_dataset'], batch_size=batch_size, shuffle=False,
                        collate_fn=data['data_collator'], num_workers=0)
    total_text, total_vision, count = None, None, 0
    for batch in loader:
        images = batch.get('images')
        if images is not None:
            images = images.to(device)
        text, vision = queries(base, batch['instruction_ids'].to(device), batch['instruction_mask'].to(device),
                               images, batch['has_image'])
        total_text = text.float().sum(0) if total_text is None else total_text + text.float().sum(0)
        total_vision = vision.float().sum(0) if total_vision is None else total_vision + vision.float().sum(0)
        count += text.shape[0]
    if not count:
        raise ValueError("cannot initialize PCLR keys from an empty task")
    return F.normalize(total_text, dim=0), F.normalize(total_vision, dim=0), count


class RoutedLlava(nn.Module):
    def __init__(self, pool, stage='learn'):
        super().__init__()
        self.pool, self.stage = pool, stage
        self.config = pool.base.config
        self.register_buffer('probability', torch.ones(len(pool.task_text), device=pool.task_text.device))

    def gradient_checkpointing_enable(self, **kwargs):
        self.pool.base.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.pool.base.gradient_checkpointing_disable()

    def forward(self, input_ids=None, attention_mask=None, labels=None, images=None,
                instruction_ids=None, instruction_mask=None, has_image=None, **kwargs):
        with model_autocast(self.pool.base):
            return self._forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, images=images,
                                 instruction_ids=instruction_ids, instruction_mask=instruction_mask,
                                 has_image=has_image, **kwargs)

    def _forward(self, input_ids=None, attention_mask=None, labels=None, images=None,
                 instruction_ids=None, instruction_mask=None, has_image=None, **kwargs):
        base = self.pool.base
        args = dict(input_ids=input_ids, attention_mask=attention_mask, images=images, **kwargs)
        if self.stage == 'learn':
            text, vision = queries(base, instruction_ids, instruction_mask, images, has_image)
            self.pool.set_query(text, vision, 'new')
            output = base(labels=labels, **args)
            return {'loss': output.loss + self.pool.config.query_weight * self.pool.alignment_loss(),
                    'logits': output.logits}
        if self.stage != 'integrate':
            raise ValueError(f"unknown PCLR stage {self.stage}")
        if not len(self.probability):
            raise ValueError("PCLR integration needs task query prototypes")
        task = int(torch.multinomial(self.probability, 1).item())
        text = self.pool.task_text[task].unsqueeze(0).expand(input_ids.shape[0], -1)
        vision = self.pool.task_vision[task].unsqueeze(0).expand(input_ids.shape[0], -1)
        if has_image is not None:
            vision = vision * torch.as_tensor(has_image, dtype=vision.dtype, device=vision.device)[:, None]
        self.pool.set_query(text, vision, 'new' if task == len(self.probability) - 1 else 'static')
        self.pool.runtime.teacher = True
        try:
            with torch.no_grad():
                teacher = base(**args).logits
        finally:
            self.pool.runtime.teacher = False
        self.pool.runtime.part = 'static'
        student = base(**args).logits
        size = min(teacher.shape[1], student.shape[1])
        loss = distillation_loss(teacher[:, :size], student[:, :size])
        with torch.no_grad():
            self.probability[task] = (loss.detach().float().clamp_min(0) + 1e-4).sqrt()
        return {'loss': loss, 'logits': student}

    @torch.no_grad()
    def generate(self, inputs, images=None, instruction_ids=None, **kwargs):
        with model_autocast(self.pool.base):
            return self._generate(inputs, images, instruction_ids, **kwargs)

    def _generate(self, inputs, images=None, instruction_ids=None, **kwargs):
        if instruction_ids is None:
            raise ValueError("PCLR generation requires plain instruction_ids")
        mask = torch.ones_like(instruction_ids, dtype=torch.bool)
        text, vision = queries(self.pool.base, instruction_ids, mask, images)
        part = 'new'
        if len(self.pool.task_text) and next(iter(self.pool.groups.values())).new_text.shape[0]:
            score = text.to(self.pool.task_text.dtype) @ self.pool.task_text.T + vision.to(self.pool.task_vision.dtype) @ self.pool.task_vision.T
            if score.sum(0).argmax().item() != len(self.pool.task_text) - 1:
                part = 'static'
        self.pool.set_query(text, vision, part)
        return self.pool.base.generate(inputs, images=images, **kwargs)


def run(base, tokenizer, data_args, training_args, previous):
    from ...backbones.llava.trainer import LLaVATrainer
    stage = os.environ.get('MCITLIB_PCLR_STAGE', 'learn')
    task_id = int(os.environ['MCITLIB_PCLR_TASK_ID'])
    if task_id < 1:
        raise ValueError('PCLR task identity must be positive')
    device = training_args.device
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32
    prepare_base_precision(base, device, dtype)
    data = make_data(tokenizer, data_args)
    options = json.loads(os.environ.get('MCITLIB_PCLR_OPTIONS', '{}'))
    if set(options) - {'new_rank', 'top_rank', 'alpha', 'share_rank', 'skip_interval'}:
        raise ValueError('unknown PCLR pool setting')
    if previous:
        pool, phase = RankPool.load_boundary(base, Path(previous) / 'pclr.pt')
        expected_history = task_id - 1 if stage == 'learn' else task_id
        if (phase not in {'learned', 'integrated'} or len(pool.task_text) != expected_history or
                len(pool.task_vision) != expected_history or not torch.isfinite(pool.task_text).all() or
                not torch.isfinite(pool.task_vision).all()):
            raise ValueError('PCLR checkpoint task history or phase does not match this stage')
        if any(getattr(pool.config, key) != value for key, value in options.items()
               if key not in {'new_rank', 'skip_interval'}):
            raise ValueError('PCLR settings differ from the previous task checkpoint')
        # Progressive learning changes capacity and retention at task boundaries.
        pool.config = replace(pool.config, **{k: v for k, v in options.items()
                                              if k in {'new_rank', 'skip_interval'}})
        for keys in pool.groups.values():
            keys.config = pool.config
        for _, layer in pool.layers():
            layer.config = pool.config
    else:
        if stage != 'learn' or task_id != 1:
            raise ValueError("PCLR continuation or integration requires a prior checkpoint")
        pool = RankPool(base, target_groups(base), PoolConfig(base.config.hidden_size, base.get_vision_tower().hidden_size,
                                                             **options))
        phase = None
    if stage == 'learn':
        text, vision, count = task_initial_query(base, data, max(1, training_args.per_device_train_batch_size), device)
        if previous:
            pool.begin_task(text, vision)
        else:
            pool.init_new_keys(text, vision)
        print(f"PCLR key initialization used {count} task examples", flush=True)
    elif stage == 'integrate':
        if phase != 'learned':
            raise ValueError("PCLR integration must follow a learned checkpoint")
        split = next(iter(pool.groups.values())).static_text.shape[0]
        pool.merge_new()
        pool.compress(split)
    else:
        raise ValueError(f"invalid PCLR stage {stage}")
    model = RoutedLlava(pool, stage)
    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data)
    trainer.train()
    trainer.save_state()
    if stage == 'learn':
        pool.append_task_query()
    if training_args.local_rank in (-1, 0):
        Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
        pool.save_boundary(Path(training_args.output_dir) / 'pclr.pt', 'learned' if stage == 'learn' else 'integrated')
        base.config.save_pretrained(training_args.output_dir)
