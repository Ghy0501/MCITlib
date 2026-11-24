# SPDX-License-Identifier: Apache-2.0
"""CPU integration coverage for InternVL's InternViT tower."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from mcitlib.backbones.internvl.intern_vit.configuration_intern_vit import InternVisionConfig
from mcitlib.backbones.internvl.intern_vit.modeling_intern_vit import InternVisionModel


def tiny_tower(path):
    config = InternVisionConfig(
        image_size=224, patch_size=112, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4, use_flash_attn=False,
    )
    torch.manual_seed(109)
    InternVisionModel(config).save_pretrained(path)
    return config


def test_real_internvit_vision_factory_and_llava_projector(tmp_path):
    from mcitlib.backbones.llava.overrides import import_llava

    tiny_tower(tmp_path)
    import_llava()
    from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM

    from mcitlib.backbones.internvl.overrides import install_vision_factory

    install_vision_factory()
    config = LlavaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, mm_vision_tower=str(tmp_path),
        mm_hidden_size=16, mm_projector_type='mlp2x_gelu', mm_vision_select_layer=-2,
        mm_vision_select_feature='patch',
    )
    model = LlavaLlamaForCausalLM(config)
    vision = model.get_vision_tower()
    assert not vision.is_loaded and vision.hidden_size == 16
    # t437's private SEFE/Keep builders call this with an explicit device map.
    vision.load_model(device_map='cpu')
    assert vision.image_processor.crop_size == {'height': 336, 'width': 336}
    images = torch.randn(1, 3, 336, 336)
    with torch.no_grad():
        patches = vision(images)
        projected = model.encode_images(images)
    # Interpolates the 2x2 position grid to the 3x3 evaluation grid.
    assert patches.shape == (1, 9, 16) and projected.shape == (1, 9, 16)
    assert vision.num_patches == 9
    assert torch.isfinite(projected).all()
    assert all(not parameter.requires_grad for parameter in vision.parameters())


def test_guide_factory_uses_distinct_clip_projection(tmp_path, monkeypatch):
    from mcitlib.backbones.internvl import vision_tower
    from mcitlib.backbones.llava.text_tower import build_vision_tower

    tiny_tower(tmp_path)
    seen = []

    class FakeGuide(torch.nn.Module):
        def forward(self, pixels, output_hidden_states):
            assert output_hidden_states
            return SimpleNamespace(image_embeds=torch.full((pixels.shape[0], 768), 2.0))

    def load(path):
        seen.append(path)
        return FakeGuide()

    monkeypatch.setattr(vision_tower.CLIPVisionModelWithProjection, 'from_pretrained', load)
    config = SimpleNamespace(mm_vision_tower=str(tmp_path), mm_text_tower='separate-clip',
                             mm_vision_select_layer=-2, mm_vision_select_feature='patch')
    vision = build_vision_tower(config)
    guide, patches = vision(torch.randn(2, 3, 336, 336))
    assert seen == ['separate-clip']
    assert guide.shape == (2, 768) and torch.all(guide == 2)
    assert patches.shape == (2, 9, 16)


def test_vision_path_override_is_in_memory_only(tmp_path, monkeypatch):
    from mcitlib.backbones.internvl.detect import vision_name
    from mcitlib.backbones.internvl.vision_tower import build_intern_vision_tower

    tiny_tower(tmp_path)
    original = (tmp_path / 'config.json').read_bytes()
    config = SimpleNamespace(mm_vision_tower='/old/InternViT-6B-224px', mm_vision_select_layer=-2)
    monkeypatch.setenv('MCITLIB_BACKBONE', 'internvl')
    monkeypatch.setenv('MCITLIB_INTERNVL_VISION_TOWER', str(tmp_path))
    assert vision_name(config) == str(tmp_path)
    tower = build_intern_vision_tower(config, delay_load=True)
    assert tower.vision_tower_name == config.mm_vision_tower == str(tmp_path)
    assert (tmp_path / 'config.json').read_bytes() == original


def test_installed_factory_keeps_clip_tower_when_internvl_is_also_configured(tmp_path, monkeypatch):
    from mcitlib.backbones.internvl.overrides import install_vision_factory
    from mcitlib.backbones.llava.overrides import import_llava

    tiny_tower(tmp_path)
    import_llava()
    import llava.model.multimodal_encoder.builder as builder
    import llava.model.multimodal_encoder.clip_encoder as clip

    monkeypatch.setattr(clip.CLIPVisionConfig, 'from_pretrained', lambda name: SimpleNamespace(hidden_size=12))
    monkeypatch.setenv('MCITLIB_BACKBONE', 'internvl')
    monkeypatch.setenv('MCITLIB_INTERNVL_VISION_TOWER', str(tmp_path))
    install_vision_factory()
    install_vision_factory()
    clip_path = tmp_path / 'InternVL-parent' / 'clip'
    clip_path.mkdir(parents=True)
    (clip_path / 'config.json').write_text('{"model_type": "clip_vision_model"}')
    config = SimpleNamespace(mm_vision_tower=str(clip_path), mm_vision_select_layer=-2)
    vision = builder.build_vision_tower(config, delay_load=True)
    assert isinstance(vision, clip.CLIPVisionTower)
    assert vision.vision_tower_name == config.mm_vision_tower == str(clip_path)
    assert vision.hidden_size == 12
    from mcitlib.backbones.internvl.detect import is_intern_vit

    assert not is_intern_vit(tmp_path / 'InternVL-parent' / 'missing-clip')
    assert is_intern_vit('/unavailable/InternViT-6B-224px')


def test_tiny_internvl_lora_training_save_and_reload(tmp_path):
    import transformers
    from torch.utils.data import Dataset

    from mcitlib.backbones.internvl.overrides import install_vision_factory
    from mcitlib.backbones.llava.overrides import import_llava
    from mcitlib.backbones.llava.trainer import LLaVATrainer

    vision_path = tmp_path / 'vision'
    tiny_tower(vision_path)
    import_llava()
    install_vision_factory()
    from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
    from peft import LoraConfig, PeftModel, get_peft_model

    def make_base():
        torch.manual_seed(45)
        config = LlavaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32,
            mm_vision_tower=str(vision_path), mm_hidden_size=16, mm_projector_type='mlp2x_gelu',
            mm_vision_select_layer=-2, mm_vision_select_feature='patch', use_cache=False,
        )
        model = LlavaLlamaForCausalLM(config)
        model.get_vision_tower().load_model()
        model.enable_input_require_grads()
        return model

    class Examples(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            ids = torch.tensor([1, -200, 2, 3])
            return {'input_ids': ids, 'labels': ids.clone(), 'images': torch.ones(3, 336, 336)}

    model = get_peft_model(make_base(), LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj'],
                                                   lora_dropout=0, task_type='CAUSAL_LM'))
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path / 'trainer'), no_cuda=True, max_steps=1, per_device_train_batch_size=1,
        gradient_checkpointing=True, remove_unused_columns=False, save_strategy='no', report_to=[],
        disable_tqdm=True, learning_rate=0.001,
    )
    args.group_by_modality_length, args.is_SAT, args.mm_projector_lr = False, False, None
    trainer = LLaVATrainer(model=model, args=args, train_dataset=Examples())
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if parameter.requires_grad}
    assert trainer.train().global_step == 1
    assert any(not torch.equal(before[name], parameter.detach()) for name, parameter in model.named_parameters()
               if name in before)
    adapter = tmp_path / 'adapter'
    model.save_pretrained(adapter)
    reloaded = PeftModel.from_pretrained(make_base(), adapter)
    model.eval()
    reloaded.eval()
    ids = torch.tensor([[1, -200, 2, 3]])
    images = torch.ones(1, 3, 336, 336)
    with torch.no_grad():
        original = model(input_ids=ids, images=images).logits
        actual = reloaded(input_ids=ids, images=images).logits
    assert original.shape == actual.shape == (1, 12, 32)
    torch.testing.assert_close(original, actual)
    assert torch.isfinite(actual).all()


def test_inference_builder_routes_internvl_names_without_changing_llava(monkeypatch):
    from mcitlib.backbones.internvl.overrides import install_inference_builder
    from mcitlib.backbones.llava.overrides import import_llava

    import_llava()
    import llava.model.builder as builder

    calls = []

    def original(path, base, name, **kwargs):
        calls.append((path, base, name, kwargs))
        return name

    monkeypatch.setattr(builder, 'load_pretrained_model', original)
    install_inference_builder()
    install_inference_builder()
    assert builder.load_pretrained_model('adapter', 'base', 'Task1_internvl_lora', device='cpu') == (
        'llava_Task1_internvl_lora'
    )
    assert builder.load_pretrained_model('adapter', 'base', 'Task1_llava_lora', device='cpu') == 'Task1_llava_lora'
    assert calls == [('adapter', 'base', 'llava_Task1_internvl_lora', {'device': 'cpu'}),
                     ('adapter', 'base', 'Task1_llava_lora', {'device': 'cpu'})]
