# SPDX-License-Identifier: Apache-2.0
"""Real tiny LLaVA forward, continuation, consolidation and generation on CPU."""
import copy

import torch
from torch import nn

from mcitlib.backbones.llava.overrides import import_llava
from mcitlib.methods._pclr.core import PoolConfig, RankPool
from mcitlib.methods._pclr.llava import RoutedLlava, prepare_base_precision, target_groups


class TinyVision(nn.Module):
    hidden_size = 3
    select_feature = 'patch'

    def forward(self, images):
        features = images.mean(dim=(-1, -2)).unsqueeze(1)
        # [CLS] plus two patches, or two patches in the ordinary LLaVA path.
        return features.expand(-1, 3 if self.select_feature == 'cls_patch' else 2, -1)


def make_base():
    import_llava()
    from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
    config = LlavaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
                         bos_token_id=1, eos_token_id=2, pad_token_id=0)
    config.mm_use_im_start_end = False
    config.tune_mm_mlp_adapter = False
    model = LlavaLlamaForCausalLM(config)
    model.model.vision_tower = TinyVision()
    model.model.mm_projector = nn.Linear(3, 16)
    return model


def test_real_llava_two_tasks_compression_and_reload(tmp_path):
    torch.manual_seed(42)
    base = make_base()
    prepare_base_precision(base, torch.device('cpu'), torch.bfloat16)
    assert base.model.norm.weight.dtype == torch.float32
    assert base.model.layers[0].input_layernorm.weight.dtype == torch.float32
    assert base.model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
    initial = copy.deepcopy(base.state_dict())
    config = PoolConfig(text_dim=16, vision_dim=3, new_rank=4, top_rank=3,
                        alpha=6, share_rank=2, skip_interval=2)
    pool = RankPool(base, target_groups(base), config)
    text = torch.randn(16)
    vision = torch.randn(3)
    pool.init_new_keys(text, vision)
    routed = RoutedLlava(pool, 'learn')
    base.config.use_cache = False
    base.enable_input_require_grads()
    routed.gradient_checkpointing_enable()
    ids = torch.tensor([[1, -200, 4, 5]])
    labels = torch.tensor([[-100, -100, 4, 5]])
    images = torch.randn(1, 3, 4, 4)
    instruction = torch.tensor([[1, 4, 5]])
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        output = routed(input_ids=ids, labels=labels, images=images, instruction_ids=instruction,
                        instruction_mask=torch.ones_like(instruction), has_image=[True])
    output['loss'].backward()
    assert any(layer.new_b.grad is not None and layer.new_b.grad.abs().sum() > 0 for _, layer in pool.layers())
    pool.append_task_query()
    p1 = tmp_path / 'task1.pt'
    pool.save_boundary(p1, 'learned')
    base2 = make_base()
    prepare_base_precision(base2, torch.device('cpu'), torch.bfloat16)
    base2.load_state_dict(initial)
    pool2, phase = RankPool.load_boundary(base2, p1)
    assert phase == 'learned'
    pool2.begin_task(torch.randn(16), torch.randn(3))
    pool2.append_task_query()
    pool2.merge_new()
    pool2.compress(4)
    integration = RoutedLlava(pool2, 'integrate')
    base2.config.use_cache = False
    base2.enable_input_require_grads()
    integration.gradient_checkpointing_enable()
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        result = integration(input_ids=ids, labels=labels, images=images, instruction_ids=instruction,
                             instruction_mask=torch.ones_like(instruction), has_image=[True])
    result['loss'].backward()
    assert any(layer.static_b.grad is not None for _, layer in pool2.layers())
    out = tmp_path / 'task2.pt'
    pool2.save_boundary(out, 'integrated')
    base3 = make_base()
    prepare_base_precision(base3, torch.device('cpu'), torch.bfloat16)
    base3.load_state_dict(initial)
    pool3, phase = RankPool.load_boundary(base3, out)
    assert phase == 'integrated' and pool3.task_text.shape[0] == 2
    restored = RoutedLlava(pool3, 'eval').eval()
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        generated = restored.generate(ids, images=images, instruction_ids=instruction, max_new_tokens=2,
                                      do_sample=False, num_beams=1)
    assert generated.shape[0] == 1 and 1 <= generated.shape[1] - ids.shape[1] <= 2
