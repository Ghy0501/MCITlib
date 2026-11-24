# SPDX-License-Identifier: Apache-2.0
"""Private SEFE/Keep entry points must select a multimodal InternVL model."""
from __future__ import annotations

import json
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch
import transformers
from packaging.version import Version

pytestmark = pytest.mark.skipif(Version(transformers.__version__) < Version('4.36'),
                                reason='private SEFE/Keep builders require the t437 environment')


@pytest.mark.parametrize('method', ['sefe', 'keeplora'])
def test_direct_merge_and_gradient_entry_use_multimodal_builder(tmp_path, monkeypatch, method):
    from mcitlib.backbones.llava.overrides import allow_t437_llava_registration, import_llava

    import_llava()
    with allow_t437_llava_registration():
        builder = import_module(f'mcitlib.methods._{method}.llava_overrides.builder')
        merge = import_module(f'mcitlib.methods._{method}.merge')
    actual_builder = import_module(builder.load_pretrained_model.__module__)
    import llava.model.language_model.llava_llama as vendor_llama
    import peft

    calls = []

    class Tokenizer:
        def __len__(self):
            return 16

        def save_pretrained(self, path):
            calls.append(('save_tokenizer', path))

    class Tower:
        is_loaded = False
        image_processor = object()

        def load_model(self, device_map=None):
            calls.append(('vision_load', device_map))
            self.is_loaded = True

        def to(self, **kwargs):
            return self

    class Model:
        def __init__(self):
            self.config = SimpleNamespace(mm_use_im_patch_token=False, mm_use_im_start_end=False)
            self.lm_head = torch.nn.Linear(2, 16, bias=False)
            self.tower = Tower()

        def get_vision_tower(self):
            return self.tower

        def resize_token_embeddings(self, count):
            assert count == 16

        def load_state_dict(self, state, strict):
            assert state == {} and not strict

        def to(self, *args, **kwargs):
            return self

        def train(self):
            return self

        def save_pretrained(self, path):
            calls.append(('save_model', path))

    def multimodal(*args, **kwargs):
        calls.append(('multimodal_model', args[0]))
        return Model()

    def text_only(*args, **kwargs):
        raise AssertionError('InternVL was sent to a text-only model')

    tokenizer = Tokenizer()
    monkeypatch.setattr(actual_builder.AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(actual_builder.AutoModelForCausalLM, 'from_pretrained', text_only)
    monkeypatch.setattr(actual_builder.LlavaLlamaForCausalLM, 'from_pretrained', multimodal)
    monkeypatch.setattr(actual_builder.LlavaConfig, 'from_pretrained', lambda *args, **kwargs: object())
    monkeypatch.setattr(vendor_llama.LlavaConfig, 'from_pretrained', lambda *args, **kwargs: object())
    monkeypatch.setattr(peft.PeftModel, 'from_pretrained',
                        lambda model, path: SimpleNamespace(merge_and_unload=lambda: model))
    adapter = tmp_path / 'Task2_internvl_lora'
    adapter.mkdir()
    torch.save({}, adapter / 'non_lora_trainables.bin')
    merge.merge_lora(SimpleNamespace(model_path=str(adapter), model_base='internvl-base',
                                     save_model_path=str(tmp_path / 'merged')))
    assert ('multimodal_model', 'internvl-base') in calls
    assert ('vision_load', 'cpu') in calls
    assert ('save_model', str(tmp_path / 'merged')) in calls

    if method == 'keeplora':
        # Exercise the actual direct import in gradient extraction, through model
        # loading; stop at the dataset boundary without allocating a GPU model.
        with allow_t437_llava_registration():
            extract = import_module('mcitlib.methods._keeplora.extract_gradients')
        assert extract.load_pretrained_model is builder.load_pretrained_model

        class DatasetReached(Exception):
            pass

        def stop_at_dataset(**kwargs):
            raise DatasetReached

        monkeypatch.setattr(extract, 'disable_torch_init', lambda: None)
        monkeypatch.setattr(extract, 'make_supervised_data_module', stop_at_dataset)
        for name, payload in (
            ('data', {'train_path': 'dataset', 'train_folder': 'images'}),
            ('model', {'model_name': 'internvl-base'}),
            ('train', {'model_path': str(adapter)}),
        ):
            (tmp_path / f'{name}.json').write_text(json.dumps(payload))
        args = SimpleNamespace(data_config=str(tmp_path / 'data.json'), model_config=str(tmp_path / 'model.json'),
                               train_config=str(tmp_path / 'train.json'))
        with pytest.raises(DatasetReached):
            extract.extract_gradients(args)
        assert calls.count(('multimodal_model', 'internvl-base')) == 2
        assert ('vision_load', 'auto') in calls


def test_sefe_internvit_key_elements_produce_real_regularization_gradients(tmp_path):
    from mcitlib.backbones.internvl.intern_vit.configuration_intern_vit import InternVisionConfig
    from mcitlib.backbones.internvl.intern_vit.modeling_intern_vit import InternVisionModel
    from mcitlib.backbones.llava.overrides import allow_t437_llava_registration, import_llava
    from mcitlib.methods._sefe.key_elements import identify_key_elements

    import_llava()
    with allow_t437_llava_registration():
        llama = import_module('mcitlib.methods._sefe.llava_overrides.language_model.llava_llama')
    from peft import LoraConfig, get_peft_model

    vision_path = tmp_path / 'vision'
    vision_config = InternVisionConfig(image_size=224, patch_size=112, hidden_size=16, intermediate_size=32,
                                       num_hidden_layers=4, num_attention_heads=4, use_flash_attn=False)
    InternVisionModel(vision_config).save_pretrained(vision_path)
    config = llama.LlavaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                               num_attention_heads=4, num_key_value_heads=4, mm_vision_tower=str(vision_path),
                               mm_hidden_size=16, mm_projector_type='mlp2x_gelu', mm_vision_select_layer=-2,
                               mm_vision_select_feature='patch', use_cache=False)
    targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

    def make_model(regularization=None):
        model = llama.LlavaLlamaForCausalLM(config, regularization_info_path=regularization)
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=targets,
                                                lora_dropout=0, task_type='CAUSAL_LM'))
        model.get_vision_tower().load_model(device_map='cpu')
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if 'lora_A.' in name or 'lora_B.' in name:
                    parameter.normal_(0, 0.1)
        return model

    torch.manual_seed(59)
    first = make_model()
    first.save_pretrained(tmp_path / 'Task1_internvl_lora')
    top1 = identify_key_elements(0, tmp_path, 'internvl')
    indices = torch.load(top1)
    assert len(indices) == 7 and all(item['top2'].shape[0] > 0 for item in indices.values())
    second = make_model(str(top1))
    ids = torch.tensor([[1, -200, 2, 3]])
    images = torch.ones(1, 3, 336, 336)
    labels = ids.clone()
    regularized_output = second(input_ids=ids, labels=labels, images=images)
    with_regularization = regularized_output.loss
    assert torch.isfinite(with_regularization), f'non-finite logits={not torch.isfinite(regularized_output.logits).all()}'
    backbone = second.get_base_model().get_model()
    layer = backbone.layers[0]
    saved = layer.regularization_info
    saved_path = backbone.regularization_info_path
    layer.regularization_info = None
    backbone.regularization_info_path = None
    without_regularization = second(input_ids=ids, labels=labels, images=images).loss
    assert torch.isfinite(without_regularization)
    layer.regularization_info = saved
    backbone.regularization_info_path = saved_path
    penalty = with_regularization - without_regularization
    assert torch.isfinite(penalty) and penalty.item() > 0
    penalty.backward()
    grads = [parameter.grad for name, parameter in second.named_parameters() if 'lora_B.' in name]
    assert len(grads) == 7 and all(gradient is not None and torch.count_nonzero(gradient) for gradient in grads)
    second.save_pretrained(tmp_path / 'Task2_internvl_lora')
    top2 = identify_key_elements(1, tmp_path, 'internvl')
    accumulated = torch.load(top2)
    assert all(accumulated[name]['top2'].shape[0] == 2 * indices[name]['top2'].shape[0] for name in indices)
