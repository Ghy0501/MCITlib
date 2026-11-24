# SPDX-License-Identifier: Apache-2.0
"""Exercise actual method models against real tiny InternViT and CLIP towers."""
from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import BatchEncoding, CLIPConfig, CLIPModel

from mcitlib.backbones.internvl.intern_vit.configuration_intern_vit import InternVisionConfig
from mcitlib.backbones.internvl.intern_vit.modeling_intern_vit import InternVisionModel
from mcitlib.backbones.llava.overrides import import_llava


@pytest.fixture(scope='module')
def tiny_assets(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp('internvl-method-tiny')
    vit = root / 'InternViT-tiny'
    clip = root / 'separate-clip'
    torch.manual_seed(17)
    InternVisionModel(InternVisionConfig(image_size=224, patch_size=14, hidden_size=16,
        intermediate_size=32, num_hidden_layers=4, num_attention_heads=4,
        use_flash_attn=False)).save_pretrained(vit)
    clip_config = CLIPConfig(
        text_config={'vocab_size': 32, 'hidden_size': 768, 'intermediate_size': 64, 'num_hidden_layers': 1,
         'num_attention_heads': 12, 'max_position_embeddings': 77, 'bos_token_id': 1,
         'eos_token_id': 2, 'pad_token_id': 0},
        vision_config={'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 1, 'num_attention_heads': 4,
         'image_size': 336, 'patch_size': 14, 'projection_dim': 768}, projection_dim=768)
    CLIPModel(clip_config).save_pretrained(clip)
    return vit, clip


class FakeTokenizer:
    pad_token_id = 0

    def batch_decode(self, ids, skip_special_tokens=True):
        return ['USER:\nWhat is in the picture? ASSISTANT:'] * len(ids)

    def __call__(self, texts, **_kwargs):
        return BatchEncoding({'input_ids': torch.tensor([[1, 4, 2]] * len(texts)),
                              'attention_mask': torch.ones(len(texts), 3, dtype=torch.long)})


def make_model(method, tiny_assets, *, load_towers=True):
    import_llava()
    vit, clip = tiny_assets
    module = importlib.import_module(f'mcitlib.methods._{method}.llava_overrides.language_model.llava_llama')
    if method == 'modalprompt':
        cls = importlib.import_module('mcitlib.methods._modalprompt.llava_overrides').MCITModalPrompt
    else:
        cls = module.LlavaLlamaForCausalLM
    torch.manual_seed(2026)
    config = module.LlavaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=32, num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
        mm_vision_tower=str(vit), mm_text_tower=str(clip), mm_hidden_size=16,
        mm_projector_type='mlp2x_gelu', mm_vision_select_layer=-4, mm_text_select_layer=-1,
        mm_vision_select_feature='patch', tokenizer_model_max_length=1024, use_cache=True)
    model = cls(config)
    if load_towers:
        model.get_vision_tower().load_model()
        model.get_text_tower().load_model()
    model.tokenizer = FakeTokenizer()
    model.clip_tokenizer = FakeTokenizer()
    if method == 'modalprompt':
        model.cur_task, model.prefix_len, model.num_tasks = 1, 2, 1
        model.proto_ids_list = torch.tensor([[8, 9]])
        model.prompt_transform = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(16, 16), torch.nn.SiLU(), torch.nn.Linear(16, 768))])
    else:
        model.set_eval(2)
    model.eval()
    return model


@pytest.mark.parametrize('method', ['disco', 'hide', 'modalprompt'])
def test_method_model_uses_real_internvit_patches_and_clip_guides(tiny_assets, method):
    model = make_model(method, tiny_assets)
    assert len(model.model.layers) == 32
    assert model.get_vision_tower().select_layer == -4
    images = torch.randn(1, 3, 336, 336)
    with torch.inference_mode():
        guide, patches = model.encode_images(images)
    assert guide.shape == (1, 768)
    assert patches.shape == (1, 576, 16)
    assert torch.isfinite(guide).all() and torch.isfinite(patches).all()
    assert model.get_vision_tower().vision_tower_name != model.get_vision_tower().guide_tower_name

    # The actual configs route 3200-wide patches through a 4096-wide projector;
    # check the full geometry on meta while keeping the live CPU tower tiny.
    config = SimpleNamespace(mm_hidden_size=3200, hidden_size=4096, mm_projector_type='mlp2x_gelu')
    with torch.device('meta'):
        projector = importlib.import_module(
            f'mcitlib.methods._{method}.llava_overrides.multimodal_projector.builder').build_vision_projector(config)
    assert projector[0].weight.shape == (4096, 3200)
    assert projector[2].weight.shape == (4096, 4096)


@pytest.mark.parametrize('method', ['disco', 'hide'])
def test_method_adapter_cache_save_and_reload_with_real_towers(tiny_assets, tmp_path, method):
    from peft import PeftModel, get_peft_model

    from mcitlib.tuners._peft_registry import install_method_adapter
    from mcitlib.tuners.inference_cache import expert_inference_scope

    install_method_adapter(method)
    variant = importlib.import_module(f'mcitlib.tuners.variants.{method}')
    config_class = variant.MCITDiscoLoraConfig if method == 'disco' else variant.MCITHiDeLoraConfig
    config = config_class(r=4, lora_alpha=8, expert_num=2, cur_task=1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0, task_type='CAUSAL_LM')
    # Production creates/loads the adapter before the delayed vision and text
    # towers.  The ordering also keeps the frozen CLIP projections outside LoRA.
    model = get_peft_model(make_model(method, tiny_assets, load_towers=False), config)
    model.get_vision_tower().load_model()
    model.get_text_tower().load_model()
    model.eval()
    folder = tmp_path / method
    model.save_pretrained(folder, safe_serialization=False)
    reloaded = PeftModel.from_pretrained(make_model(method, tiny_assets, load_towers=False), folder)
    reloaded.get_vision_tower().load_model()
    reloaded.get_text_tower().load_model()
    reloaded.eval()
    ids = torch.tensor([[1, -200, 2, 3]])
    mask = torch.ones_like(ids)
    images = torch.randn(1, 3, 336, 336)
    with torch.inference_mode(), expert_inference_scope():
        before = model(input_ids=ids, attention_mask=mask, images=images).logits
    with torch.inference_mode(), expert_inference_scope():
        after = reloaded(input_ids=ids, attention_mask=mask, images=images).logits
    assert before.shape == after.shape == (1, 579, 32)
    assert torch.isfinite(before).all() and torch.isfinite(after).all()
    torch.testing.assert_close(before, after)


def test_olora_nonreentrant_checkpoint_restores_and_backpropagates():
    import torch.utils.checkpoint as checkpoint

    from mcitlib.backbones.llava.olora_checkpoint import olora_checkpoint_scope

    method = SimpleNamespace(id='olora', training_overrides=lambda: {'gradient_checkpointing_kwargs': {'use_reentrant': False}})
    args = SimpleNamespace(gradient_checkpointing=True)
    original = checkpoint.checkpoint
    x = torch.tensor([2.0], requires_grad=True)
    with olora_checkpoint_scope(method, args) as observation:
        y = checkpoint.checkpoint(lambda value: value.square(), x)
        y.sum().backward()
        assert observation.calls == 1
    assert checkpoint.checkpoint is original and torch.equal(x.grad, torch.tensor([4.0]))


@pytest.mark.parametrize(('benchmark', 'digest'), [
    ('ucit', '2a4595386ca010a104d3dd2109f2da52c6cabe562aa5d6304e3a469880d4bc37'),
    ('dcl', 'ea251206f6ef970a2207e150810e416c134246c3904cffac1b2c91abfcf01721'),
])
def test_smolora_real_internvit_instruction_training_save_and_reload(tiny_assets, tmp_path, benchmark, digest):
    from peft import PeftModel

    from mcitlib.backbones.internvl.overrides import install_vision_factory
    from mcitlib.methods.smolora import SMoLoRAMethod, load_instruction_embeddings
    from mcitlib.tuners._peft_registry import install_method_adapter

    import_llava()
    install_vision_factory()
    install_method_adapter('smolora')
    from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM

    vit, _ = tiny_assets
    asset = Path(__file__).resolve().parents[2] / 'assets' / 'smolora' / f'ins_emb_multi_{benchmark}.pkl'
    embeddings = load_instruction_embeddings(asset, digest)
    assert torch.as_tensor(embeddings).ndim == 2 and torch.as_tensor(embeddings).shape[1] == 384

    def base():
        torch.manual_seed(2026)
        config = LlavaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=32, num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
            mm_vision_tower=str(vit), mm_hidden_size=16, mm_projector_type='mlp2x_gelu',
            mm_vision_select_layer=-4, mm_vision_select_feature='patch',
            tokenizer_model_max_length=1024, use_cache=False)
        return LlavaLlamaForCausalLM(config)

    method = SMoLoRAMethod({'r': 4, 'lora_alpha': 8, 'expert_num': 4, 'ins_type': 0,
                            'ins_emb': str(asset), 'ins_emb_sha256': digest, 'lora_dropout': 0,
                            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                               'gate_proj', 'up_proj', 'down_proj']})
    model = method.build_trainable(base(), None)
    assert not model.get_vision_tower().is_loaded
    model.get_vision_tower().load_model()
    assert all(not param.requires_grad for param in model.get_vision_tower().parameters())
    gates = [module for name, module in model.named_modules() if name.endswith('lora_ins_gate.default')]
    assert gates and all(module.in_features == 384 for module in gates)
    ids = torch.tensor([[1, -200, 2, 3]])
    mask = torch.ones_like(ids)
    images = torch.randn(1, 3, 336, 336)
    model.train()
    result = model(input_ids=ids, attention_mask=mask, images=images, labels=ids)
    assert result.logits.shape == (1, 579, 32) and torch.isfinite(result.loss)
    result.loss.backward()
    gradients = [param.grad for name, param in model.named_parameters() if 'lora_B' in name and param.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    assert any(torch.count_nonzero(grad) for grad in gradients)
    assert all(param.grad is None for param in model.get_vision_tower().parameters())
    model.eval()
    path = tmp_path / benchmark
    model.save_pretrained(path, safe_serialization=False)
    reloaded = PeftModel.from_pretrained(base(), path)
    assert not reloaded.get_vision_tower().is_loaded
    reloaded.get_vision_tower().load_model()
    reloaded.eval()
    with torch.inference_mode():
        before = model(input_ids=ids, attention_mask=mask, images=images).logits
        after = reloaded(input_ids=ids, attention_mask=mask, images=images).logits
    assert torch.isfinite(before).all() and torch.isfinite(after).all()
    torch.testing.assert_close(before, after)
