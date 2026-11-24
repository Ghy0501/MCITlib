# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""Collect full-model current-data GPWC for each cumulative history model.

Only the sum needed by LoRA-target regularization is persisted. Per-history
full-model means and their hashes are collected before releasing memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import MethodType

import torch
from torch.utils.data import DataLoader, Subset

from .core import sample_indices
from .state import load_manifest


def _target_weight(name):
    suffixes = ('q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight',
                'gate_proj.weight', 'up_proj.weight', 'down_proj.weight')
    return name.startswith('model.layers.') and name.endswith(suffixes)


def _hash_tensor(value):
    # NumPy has no native bfloat16; byte-view keeps the exact serialized bits.
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _open_vision_gradients(model):
    tower = model.get_vision_tower()
    forward = getattr(tower.forward, '__wrapped__', None)
    if forward is None:
        raise RuntimeError('LLaVA vision tower forward is not the expected no-grad wrapper')
    tower.forward = MethodType(forward, tower)


def _enable_flash_attention():
    # Match the native LLaVA training entry and avoid storing a quadratic
    # attention matrix at the 4096-token GPWC context length.
    from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

    replace_llama_attn_with_flash_attn()


def _move_batch(batch, device):
    out = {}
    for key, value in batch.items():
        if key == 'images':
            out[key] = [x.to(device=device, dtype=torch.bfloat16) for x in value] if isinstance(value, list) else value.to(device=device, dtype=torch.bfloat16)
        elif torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def supervised_token_counts(model, batch):
    """Count shifted labels after the LLaVA image expansion and length cap."""
    labels, ids = batch['labels'], batch['input_ids']
    attention = batch.get('attention_mask', torch.ones_like(ids, dtype=torch.bool)).bool()
    images = batch.get('images')
    multimodal = images is not None and hasattr(model, 'get_vision_tower')
    if multimodal:
        tower = model.get_vision_tower()
        features = tower.num_patches + (tower.select_feature == 'cls_patch')
        limit = getattr(model.config, 'tokenizer_model_max_length', None)
    else:
        features, limit = 0, None
    result = []
    for row in range(labels.shape[0]):
        row_labels = labels[row][attention[row]]
        row_ids = ids[row][attention[row]]
        if not len(row_labels) or row_labels[0] != -100:
            raise ValueError('GPWC requires a masked first label in every sample')
        positions = (row_ids == -200).nonzero(as_tuple=True)[0].tolist() if multimodal else []
        if len(positions) > 1:
            raise ValueError('GPWC expects at most one image placeholder per sample')
        if positions:
            image = images[row]
            if image.ndim == 3:
                chunks = 1
            elif image.ndim == 4:
                chunks = image.shape[0]
            else:
                raise ValueError('unsupported image layout for GPWC token counting')
            position = positions[0]
            padding = row_labels.new_full((features * chunks,), -100)
            row_labels = torch.cat((row_labels[:position], padding, row_labels[position + 1:]))
        if limit is not None:
            row_labels = row_labels[:limit]
        result.append(int(row_labels[1:].ne(-100).sum().item()))
    return result


def collect_mean(model, dataloader, device, microbatch):
    """Mean one full-model gradient per logical batch, offloading to CPU.

    The V4 loader may split a logical batch to fit memory. Each microbatch loss
    is weighted by its non-ignored label count, preserving the aggregate CE
    objective; bf16 accumulation order can differ from a single large batch.
    """
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    means = {}
    count = 0
    for batch in dataloader:
        labels = batch['labels']
        weights = supervised_token_counts(model, batch)
        denominator = sum(weights)
        if denominator <= 0:
            raise ValueError('GPWC logical batch has no supervised tokens')
        model.zero_grad(set_to_none=True)
        size = labels.shape[0]
        for start in range(0, size, microbatch):
            stop = min(size, start + microbatch)
            part = {key: value[start:stop] if torch.is_tensor(value) or isinstance(value, list) else value
                    for key, value in batch.items()}
            part = _move_batch(part, device)
            weight = sum(weights[start:stop]) / denominator
            if weight:
                (model(**part).loss * weight).backward()
        count += 1
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach().cpu()
            if count == 1:
                means[name] = grad.clone()
            elif name not in means:
                raise ValueError(f'GPWC parameter became active only after batch one: {name}')
            else:
                previous = means[name]
                means[name] = previous + (grad - previous) / count
        if count > 1 and {name for name, p in model.named_parameters() if p.grad is not None} != set(means):
            raise ValueError('GPWC active full-model parameter set changed between batches')
    if not count:
        raise ValueError('GPWC dataset produced no batches')
    return means, count


def reduce_history(histories, keys, device):
    """Use the release's stacked history reduction in the original dtype."""
    result = {}
    for name in sorted(keys):
        with torch.no_grad():
            stack = torch.stack([entry[name].to(device) for entry in histories])
            result[name] = stack.sum(dim=0).detach().cpu()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-base', required=True)
    parser.add_argument('--vision-tower', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--image_folder', required=True)
    parser.add_argument('--history', action='append', required=True)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--logical-batch', type=int, default=16)
    parser.add_argument('--microbatch', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.samples, args.logical_batch, args.microbatch) < 1:
        raise ValueError('GPWC sizes must be positive')
    if args.output.exists():
        raise FileExistsError('GPWC output already exists')
    if not torch.cuda.is_available():
        raise RuntimeError('full-model GPWC requires a CUDA GPU')
    from mcitlib.backbones.llava.overrides import import_llava
    import_llava()
    _enable_flash_attention()
    from llava import conversation as conversation_lib
    from llava.model import LlavaLlamaForCausalLM
    from llava.model.language_model.llava_llama import LlavaConfig
    from peft import PeftModel
    from transformers import AutoTokenizer

    from mcitlib.backbones.llava.standard_train import DataArguments, make_supervised_data_module

    device = torch.device('cuda:0')
    config = LlavaConfig.from_pretrained(args.model_base)
    config.mm_vision_tower = args.vision_tower
    model = LlavaLlamaForCausalLM.from_pretrained(args.model_base, config=config, torch_dtype=torch.bfloat16,
                                                  low_cpu_mem_usage=True).to(device)
    model.config.use_cache = False
    model.config.tokenizer_model_max_length = 4096
    tower = model.get_vision_tower()
    if not tower.is_loaded:
        tower.load_model()
    tower.to(device=device, dtype=torch.bfloat16)
    _open_vision_gradients(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=True, model_max_length=4096,
                                               padding_side='right')
    tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates['v1']
    data_args = DataArguments(data_path=args.data_path, image_folder=args.image_folder, lazy_preprocess=True,
                              is_multimodal=True, image_aspect_ratio='pad')
    data_args.image_processor = tower.image_processor
    data_args.mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
    data_args.mm_use_im_patch_token = getattr(model.config, 'mm_use_im_patch_token', False)
    data = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    indexes = sample_indices(len(data['train_dataset']), args.samples)
    generator = torch.Generator().manual_seed(42)
    order = torch.randperm(len(indexes), generator=generator).tolist()
    subset = Subset(data['train_dataset'], [indexes[i] for i in order])
    loader = DataLoader(subset, batch_size=args.logical_batch, shuffle=False,
                        collate_fn=data['data_collator'], num_workers=0)
    histories, audits = [], []
    history = [str(Path(path).resolve()) for path in args.history]
    for index, path in enumerate(history):
        if load_manifest(Path(path)) != [Path(p) for p in history[:index]]:
            raise ValueError('GPWC history is not chronological')
        model = PeftModel.from_pretrained(model, path).merge_and_unload()
        means, batches = collect_mean(model, loader, device, args.microbatch)
        signatures = {name: {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': _hash_tensor(value)}
                      for name, value in means.items()}
        keys = {name for name in means if _target_weight(name)}
        if not keys:
            raise RuntimeError('GPWC full-model gradients contained no LoRA-target weights')
        if histories and keys != {name for name in histories[0] if _target_weight(name)}:
            raise RuntimeError('GPWC LoRA-target keys changed between history models')
        audits.append({'history': path, 'batches': batches, 'full_gradient_tensors': signatures})
        histories.append(means)
        model.zero_grad(set_to_none=True)
    # Match the release: stack BF16 histories on CUDA, then torch.sum over
    # the history dimension. Keep full means in host RAM rather than writing
    # one ~14-GiB file per historical model.
    gradients = reduce_history(histories, keys, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'version': 1, 'history': history, 'gradients': gradients,
                'samples': args.samples, 'logical_batch': args.logical_batch, 'microbatch': args.microbatch}, args.output)
    from .state import digest
    audit = {'version': 1, 'history': history, 'gpwc_sha256': digest(args.output),
             'samples': args.samples, 'logical_batch': args.logical_batch, 'microbatch': args.microbatch,
             'reduction': 'cuda_stack_sum_original_dtype', 'full_gradient_audits': audits}
    audit_path = args.output.with_suffix(args.output.suffix + '.audit.json')
    with audit_path.open('x', encoding='utf-8') as handle:
        json.dump(audit, handle, indent=2)



if __name__ == '__main__':
    main()
