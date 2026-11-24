# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""Bounded PCLR precision check using tiny real LLaVA and CLIP models.

No checkpoint, dataset, download, distributed launcher, or optimizer state is
required. Set CUDA_VISIBLE_DEVICES before selecting ``--device cuda:0``.
"""
from __future__ import annotations

import argparse
import json

import torch
from torch import nn
from transformers import CLIPVisionConfig, CLIPVisionModel

from mcitlib.backbones.llava.overrides import import_llava
from mcitlib.methods._pclr.core import PoolConfig, RankPool
from mcitlib.methods._pclr.llava import RoutedLlava, prepare_base_precision, target_groups, task_initial_query


def run(device: str = 'cpu') -> dict:
    import_llava()
    from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

    torch.manual_seed(42)
    target = torch.device(device)
    config = LlavaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
                         bos_token_id=1, eos_token_id=2, pad_token_id=0)
    config.mm_use_im_start_end = False
    config.tune_mm_mlp_adapter = False
    base = LlavaLlamaForCausalLM(config)
    # The LLaVA tower is constructed without loading any external checkpoints;
    # its forward and the enclosed Transformers CLIP implementation are real.
    tower = CLIPVisionTower.__new__(CLIPVisionTower)
    nn.Module.__init__(tower)
    tower.is_loaded, tower.select_layer, tower.select_feature = True, -2, 'patch'
    tower.vision_tower = CLIPVisionModel(CLIPVisionConfig(hidden_size=8, intermediate_size=16,
                                                        num_hidden_layers=2, num_attention_heads=2,
                                                        image_size=4, patch_size=2))
    base.model.vision_tower = tower
    base.model.mm_projector = nn.Linear(8, 16)
    prepare_base_precision(base, target, torch.bfloat16)
    norm = tower.vision_tower.vision_model.pre_layrnorm
    patch_weight = tower.vision_tower.vision_model.embeddings.patch_embedding.weight
    assert norm.weight.dtype == torch.float32 and patch_weight.dtype == torch.bfloat16
    assert base.model.norm.weight.dtype == torch.float32

    observed = []
    def observe(module, inputs):
        enabled = torch.is_autocast_enabled() if target.type == 'cuda' else torch.is_autocast_cpu_enabled()
        observed.append((enabled, str(inputs[0].dtype), str(module.weight.dtype)))
    hook = norm.register_forward_pre_hook(observe)
    try:
        ids = torch.tensor([[1, -200, 4, 5]], device=target)
        labels = torch.tensor([[-100, -100, 4, 5]], device=target)
        instruction = torch.tensor([[1, 4, 5]], device=target)
        images = torch.randn(1, 3, 4, 4, device=target)
        batch = {'instruction_ids': instruction, 'instruction_mask': torch.ones_like(instruction),
                 'images': images, 'has_image': [True]}
        text, vision, count = task_initial_query(base, {'train_dataset': [0], 'data_collator': lambda _: batch},
                                                  1, target)
        assert count == 1 and torch.isfinite(text).all() and torch.isfinite(vision).all()
        query_calls = len(observed)
        pool = RankPool(base, target_groups(base), PoolConfig(text_dim=16, vision_dim=8, new_rank=4,
                        top_rank=3, alpha=6, share_rank=2, skip_interval=2))
        pool.init_new_keys(text, vision)
        model = RoutedLlava(pool, 'learn')
        output = model(input_ids=ids, labels=labels, images=images, instruction_ids=instruction,
                       instruction_mask=torch.ones_like(instruction), has_image=[True])
        assert torch.isfinite(output['loss'])
        output['loss'].backward()
        changed = sum(layer.new_b.grad is not None and layer.new_b.grad.abs().sum().item() > 0
                      for _, layer in pool.layers())
        assert changed > 0
        training_calls = len(observed) - query_calls
        pool.append_task_query()
        model.eval()
        generated = model.generate(ids, images=images, instruction_ids=instruction, max_new_tokens=2,
                                   do_sample=False, num_beams=1)
        assert generated.shape[0] == 1 and 1 <= generated.shape[1] - ids.shape[1] <= 2
        inference_calls = len(observed) - query_calls - training_calls
        assert query_calls >= 1 and training_calls >= 2 and inference_calls >= 2
        assert all(enabled and weight == 'torch.float32' for enabled, _, weight in observed)
        assert tower.select_feature == 'patch'
        return {'device': str(target), 'compute_dtype': str(patch_weight.dtype),
                'clip_norm_dtype': str(norm.weight.dtype), 'query_calls': query_calls,
                'training_calls': training_calls, 'generation_calls': inference_calls,
                'projections_with_gradient': changed, 'generated_tokens': generated.shape[1] - ids.shape[1]}
    finally:
        hook.remove()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', help='cpu or a visible CUDA device, e.g. cuda:0')
    args = parser.parse_args()
    print(json.dumps(run(args.device), sort_keys=True))


if __name__ == '__main__':
    main()
