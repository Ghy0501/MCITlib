# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Regressions for DISCO and HiDe default LLaVA inference."""

import copy
import hashlib
import importlib.machinery
import importlib.util
import inspect
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from mcitlib.tuners import inference_cache as cache
from mcitlib.tuners.variants import disco, hide

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures/llava_mainline'


def legacy_variants():
    """Load frozen v4 source text solely as a regression oracle."""
    expected = {
        'disco': 'f10d15866afbb8bd203160744da7b025a94bc28d8e548a18eba41e75615c56b5',
        'hide': '6e6f71c2665ba2da84c3aeeee9eb638e05e48c07fefe34e82e1ba24af3009641',
    }
    modules = []
    for kind, sha256 in expected.items():
        path = FIXTURES / f'v4-{kind}-legacy.txt'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha256
        name = f'mcitlib.tuners.variants.{kind}_legacy_test_oracle'
        loader = importlib.machinery.SourceFileLoader(name, str(path))
        spec = importlib.util.spec_from_loader(name, loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


@pytest.fixture(scope='module')
def modules():
    old_disco, old_hide = legacy_variants()
    return cache, old_disco, old_hide, disco, hide


def layer(cls, *, kind, layer_id=3, experts=2):
    options = {'expert_num': experts, 'cur_task': 1, 'train_signal': False, 'bias': False}
    if kind == 'hide':
        options.update(layer=layer_id, expert_weight=[0.2, 0.8])
    return cls('default', 4, 5, r=4, lora_alpha=4, lora_dropout=0, **options)


def test_official_modules_and_frozen_oracles_are_distinct():
    old_disco, old_hide = legacy_variants()
    root = Path(__file__).resolve().parents[2]
    assert Path(cache.__file__).resolve() == root / 'mcitlib/tuners/inference_cache.py'
    assert Path(disco.__file__).resolve() == root / 'mcitlib/tuners/variants/disco.py'
    assert Path(hide.__file__).resolve() == root / 'mcitlib/tuners/variants/hide.py'
    assert all(FIXTURES in Path(module.__file__).resolve().parents for module in (old_disco, old_hide))


@pytest.mark.parametrize('kind,layer_id', [('disco', 3), ('hide', 3), ('hide', 31)])
@pytest.mark.parametrize('dtype,amp', [(torch.float32, False), (torch.float16, True)])
def test_real_layer_matches_existing_eval_and_mixed_precision(modules, kind, layer_id, dtype, amp):
    cache, old_disco, old_hide, disco, hide = modules
    old_cls = old_disco.MCITDiscoLoraLinear if kind == 'disco' else old_hide.MCITHiDeLoraLinear
    new_cls = disco.MCITDiscoLoraLinear if kind == 'disco' else hide.MCITHiDeLoraLinear
    torch.manual_seed(93)
    old = layer(old_cls, kind=kind, layer_id=layer_id).eval()
    new = layer(new_cls, kind=kind, layer_id=layer_id).eval()
    new.load_state_dict(copy.deepcopy(old.state_dict()))
    # Production commonly keeps expert adapters in fp32 with an fp16 base.
    old.weight.data = old.weight.data.to(dtype)
    new.weight.data = new.weight.data.to(dtype)
    x = torch.randn(2, 3, 4).to(dtype)
    autocast = torch.autocast('cpu', dtype=torch.bfloat16) if amp else nullcontext()
    # Production eval disables temporary Linear's reset_parameters globally.
    with patch.object(nn.Linear, 'reset_parameters', lambda _self: None), torch.no_grad(), autocast:
        expected = old(x)
        with cache.expert_inference_scope():
            actual1, actual2 = new(x), new(x)
    torch.testing.assert_close(actual1, expected, rtol=1e-5 if not amp else 1e-2, atol=1e-6 if not amp else 1e-2)
    torch.testing.assert_close(actual2, expected, rtol=1e-5 if not amp else 1e-2, atol=1e-6 if not amp else 1e-2)


@pytest.mark.parametrize('kind', ['disco-a', 'disco-b', 'hide-a', 'hide-b'])
def test_fp16_experts_keep_legacy_fp32_temporary_weight_semantics(modules, kind):
    cache, old_disco, old_hide, disco, hide = modules
    table = {
        'disco-a': (old_disco.MCITDiscoBankA, disco.MCITDiscoBankA, (4, 4, 2, 0, False), 4),
        'disco-b': (old_disco.MCITDiscoBankB, disco.MCITDiscoBankB, (4, 5, 2, 0, False), 4),
        'hide-a': (old_hide.MCITHiDeBankA, hide.MCITHiDeBankA, (4, 4, 2, 1, False, 3, []), 4),
        'hide-b': (old_hide.MCITHiDeBankB, hide.MCITHiDeBankB, (4, 5, 2, 1, False, 3, []), 2),
    }
    old_cls, new_cls, arguments, features = table[kind]
    old = old_cls(*arguments).eval().half()
    new = new_cls(*arguments).eval().half()
    new.load_state_dict(copy.deepcopy(old.state_dict()))
    x = torch.randn(2, features)
    with patch.object(nn.Linear, 'reset_parameters', lambda _self: None), torch.no_grad(), cache.expert_inference_scope():
        torch.testing.assert_close(new(x), old(x))


@pytest.mark.parametrize('kind,layer_id', [('disco', 3), ('hide', 3), ('hide', 31)])
@pytest.mark.parametrize('training', [True, False])
def test_gradient_enabled_paths_preserve_outputs_and_gradients(modules, kind, layer_id, training):
    cache, old_disco, old_hide, disco, hide = modules
    old_cls = old_disco.MCITDiscoLoraLinear if kind == 'disco' else old_hide.MCITHiDeLoraLinear
    new_cls = disco.MCITDiscoLoraLinear if kind == 'disco' else hide.MCITHiDeLoraLinear
    torch.manual_seed(18)
    old = layer(old_cls, kind=kind, layer_id=layer_id).train(training)
    new = layer(new_cls, kind=kind, layer_id=layer_id).train(training)
    new.load_state_dict(copy.deepcopy(old.state_dict()))
    before = torch.randn(2, 4, requires_grad=True)
    after = before.detach().clone().requires_grad_()
    torch.manual_seed(88)
    expected = old(before)
    expected.square().sum().backward()
    torch.manual_seed(88)
    with cache.expert_inference_scope():
        actual = new(after)
        actual.square().sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(after.grad, before.grad)
    old_params, new_params = dict(old.named_parameters()), dict(new.named_parameters())
    assert old_params.keys() == new_params.keys()
    for name in old_params:
        left, right = old_params[name].grad, new_params[name].grad
        assert (left is None) == (right is None), name
        if left is not None:
            torch.testing.assert_close(left, right)


def test_hide_separate_sums_preserve_cross_terms(modules):
    cache, _, _, _, hide = modules
    model = layer(hide.MCITHiDeLoraLinear, kind='hide').eval()
    a = model.lora_A['default']
    b = model.lora_B['default']
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        a.loraA[0].weight.fill_(1)
        a.loraA[1].weight.fill_(2)
        b.loraB[0].weight.fill_(3)
        b.loraB[1].weight.fill_(4)
        x = torch.ones(1, 4)
        with cache.expert_inference_scope():
            actual = model(x)
        # (3 + 4) * (1 + 2) contains the cross-expert terms.
        assert torch.all(actual == 168)


def test_disco_mask_keeps_previously_enabled_block_when_flag_becomes_zero(modules):
    cache, old_disco, _, disco, _ = modules
    old = layer(old_disco.MCITDiscoLoraLinear, kind='disco').eval()
    new = layer(disco.MCITDiscoLoraLinear, kind='disco').eval()
    new.load_state_dict(copy.deepcopy(old.state_dict()))
    old.mask_signal = new.mask_signal = [1., 0., 0., 0., 0., 0., 0., 0.]
    x = torch.randn(1, 4)
    with torch.no_grad(), cache.expert_inference_scope():
        torch.testing.assert_close(new(x), old(x))
        old.mask_signal = new.mask_signal = [0., 1., 0., 0., 0., 0., 0., 0.]
        torch.testing.assert_close(new(x), old(x))
        torch.testing.assert_close(new.lora_AB['default'].weight[:2, :2], torch.eye(2))
        torch.testing.assert_close(new.lora_AB['default'].weight, old.lora_AB['default'].weight)


def test_hide_last_layer_uses_updated_dynamic_expert_weights(modules):
    cache, _, old_hide, _, hide = modules
    old = layer(old_hide.MCITHiDeLoraLinear, kind='hide', layer_id=31).eval()
    new = layer(hide.MCITHiDeLoraLinear, kind='hide', layer_id=31).eval()
    new.load_state_dict(copy.deepcopy(old.state_dict()))
    x = torch.randn(1, 4)
    with torch.no_grad(), cache.expert_inference_scope():
        torch.testing.assert_close(new(x), old(x))
        new.expert_weight[:] = [0.9, 0.1]
        old.expert_weight[:] = [0.9, 0.1]
        torch.testing.assert_close(new(x), old(x))


def test_request_cache_avoids_linear_and_repeated_cat_then_clears(modules):
    cache, _, _, disco, _ = modules
    bank = disco.MCITDiscoBankA(4, 4, 2, 0, False).eval()
    x = torch.randn(2, 4)
    original = torch.cat
    with patch.object(nn, 'Linear', side_effect=AssertionError('temporary Linear in inference')), \
         patch.object(torch, 'cat', wraps=original) as cat, torch.no_grad():
        with cache.expert_inference_scope():
            first, second = bank(x), bank(x)
            assert cat.call_count == 1
        with cache.expert_inference_scope():
            third = bank(x)
            assert cat.call_count == 2
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, third)


def test_hide_sums_cached_once_per_request_and_cur_task_invalidates(modules):
    cache, _, _, _, hide = modules
    bank = hide.MCITHiDeBankA(4, 4, 2, 0, False, 3, []).eval()
    x = torch.randn(2, 4)
    original = torch.zeros
    with patch.object(nn, 'Linear', side_effect=AssertionError('temporary Linear in inference')), \
         patch.object(torch, 'zeros', wraps=original) as zeros, torch.no_grad(), cache.expert_inference_scope():
        before, repeat = bank(x), bank(x)
        assert zeros.call_count == 1
        bank.cur_task = 1
        after = bank(x)
        assert zeros.call_count == 2 and not torch.equal(before, after)
    torch.testing.assert_close(before, repeat)


def test_reload_dtype_inplace_and_cross_request_data_mutation_invalidate(modules):
    cache, _, _, disco, _ = modules
    bank = disco.MCITDiscoBankA(4, 4, 2, 0, False).eval()
    x = torch.randn(2, 4)
    with torch.no_grad(), cache.expert_inference_scope():
        first = bank(x)
        bank.loraA[0].mlp.weight.add_(1)
        second = bank(x)
        assert not torch.equal(first, second)
        saved = copy.deepcopy(bank.state_dict())
        bank.loraA[0].mlp.weight.add_(2)
        assert not torch.equal(bank(x), second)
        bank.load_state_dict(saved)
        torch.testing.assert_close(bank(x), second)
        bank.to(dtype=torch.float64)
        torch.testing.assert_close(bank(x), second)
    with torch.no_grad():
        bank.loraA[0].mlp.weight.data.add_(3)
        with cache.expert_inference_scope():
            changed = bank(x)
        assert not torch.equal(changed, second)
        with cache.expert_inference_scope():
            old = bank(x)
            bank.loraA[0].mlp.weight.data.add_(1)
            cache.clear_expert_inference_cache()
            assert not torch.equal(bank(x), old)


def test_grad_mode_transition_clears_cached_weights(modules):
    cache, _, _, disco, _ = modules
    bank = disco.MCITDiscoBankA(4, 4, 2, 0, False).eval()
    x = torch.randn(2, 4)
    original = torch.cat
    with patch.object(torch, 'cat', wraps=original) as cat, cache.expert_inference_scope():
        with torch.no_grad():
            bank(x)
            bank(x)
            assert cat.call_count == 1
        with torch.enable_grad():
            bank(x)
            assert cat.call_count == 2
        with torch.no_grad():
            bank(x)
            assert cat.call_count == 3


def test_train_eval_switch_invalidates_even_if_mutation_uses_data(modules):
    cache, _, _, disco, _ = modules
    bank = disco.MCITDiscoBankA(4, 4, 2, 0, False).eval()
    x = torch.randn(2, 4)
    with torch.no_grad(), cache.expert_inference_scope():
        before = bank(x)
        bank.train()
        bank.loraA[0].weight.data.add_(2)
        bank.eval()
        assert not torch.equal(bank(x), before)


@pytest.mark.parametrize('kind', ['disco', 'hide'])
def test_adapter_switch_clears_scope_and_state_dict_stays_clean(modules, kind):
    cache, _, _, disco, hide = modules
    cls = disco.MCITDiscoLoraLinear if kind == 'disco' else hide.MCITHiDeLoraLinear
    model = layer(cls, kind=kind).eval()
    model.update_layer('other', r=4, lora_alpha=4, lora_dropout=0, init_lora_weights=True)
    if kind == 'disco':
        model.lora_AB['other'] = nn.Linear(4, 4, bias=False)
    keys = set(model.state_dict())
    count = [0]
    original = cache._RequestCache.clear

    def clear(self):
        count[0] += 1
        return original(self)

    x = torch.randn(1, 4)
    with patch.object(cache._RequestCache, 'clear', clear), torch.no_grad(), cache.expert_inference_scope():
        model.active_adapter = 'default'
        model(x)
        model(x)
        assert count[0] == 0
        model.active_adapter = 'other'
        model(x)
        assert count[0] == 1
        model.active_adapter = 'default'
        model(x)
        assert count[0] == 2
    assert set(model.state_dict()) == keys


def test_inference_tensor_without_version_never_crashes_or_reuses(modules):
    cache, _, _, disco, _ = modules
    with torch.inference_mode():
        bank = disco.MCITDiscoBankA(4, 4, 2, 0, False).eval()
        x = torch.randn(1, 4)
        with pytest.raises(RuntimeError):
            _ = bank.loraA[0].weight._version
        original = torch.cat
        with patch.object(torch, 'cat', wraps=original) as cat, cache.expert_inference_scope():
            bank(x)
            bank(x)
            assert cat.call_count == 2


def test_generate_scope_clears_on_exception_and_preserves_signature_and_state(modules):
    cache, _, _, disco, _ = modules

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.bank = disco.MCITDiscoBankA(4, 4, 2, 0, False)

        def generate(self, values, fail=False):
            with torch.no_grad():
                first = self.bank(values)
                if fail:
                    raise ValueError('expected')
                return first, self.bank(values)

    model = Model().eval()
    signature, keys = inspect.signature(model.generate), set(model.state_dict())
    cache.install_generation_scope(model)
    assert inspect.signature(model.generate) == signature
    assert cache.install_generation_scope(model) is model
    original = torch.cat
    with patch.object(torch, 'cat', wraps=original) as cat:
        with pytest.raises(ValueError, match='expected'):
            model.generate(torch.randn(1, 4), fail=True)
        model.generate(torch.randn(1, 4))
        assert cat.call_count == 2
    assert set(model.state_dict()) == keys and cache._ACTIVE.get() is None


def test_tiny_32_layer_hide_generation_keeps_last_layer_dynamic(modules):
    cache, _, old_hide, _, hide = modules

    class Toy(nn.Module):
        def __init__(self, cls):
            super().__init__()
            self.layers = nn.ModuleList([
                cls('default', 4, 4, r=4, lora_alpha=4, lora_dropout=0, expert_num=2,
                    cur_task=1, train_signal=False, layer=i, expert_weight=[0.2, 0.8], bias=False)
                for i in range(32)
            ])

        def generate(self, x, steps):
            with torch.no_grad():
                for index in range(steps):
                    self.layers[-1].expert_weight[:] = [0.2 + .1 * index, 0.8 - .1 * index]
                    for block in self.layers:
                        x = torch.tanh(block(x))
                return x

    torch.manual_seed(17)
    old = Toy(old_hide.MCITHiDeLoraLinear).eval()
    new = Toy(hide.MCITHiDeLoraLinear).eval()
    new.load_state_dict(copy.deepcopy(old.state_dict()))
    cache.install_generation_scope(new)
    x = torch.randn(1, 4)
    with patch.object(nn.Linear, 'reset_parameters', lambda _self: None):
        expected = old.generate(x, 3)
        actual = new.generate(x, 3)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('kind', ['disco', 'hide'])
def test_eval_entry_builder_scopes_real_generate(tmp_path, monkeypatch, kind):
    """Exercise the actual evaluator override, builder and returned model hook."""
    import importlib
    from types import SimpleNamespace

    import peft

    from mcitlib.backbones.llava.eval_runners import entry
    from mcitlib.backbones.llava.overrides import import_llava

    import_llava()
    import llava
    import llava.model
    import llava.model.builder as llava_builder
    import llava.model.language_model.llava_llama as llava_llama

    # The production installer updates these process-wide attributes. Restore
    # them after this test so that the other method can be tested independently.
    for module, name in ((llava, 'LlavaLlamaForCausalLM'),
                         (llava.model, 'LlavaLlamaForCausalLM'),
                         (llava_llama, 'LlavaLlamaForCausalLM'),
                         (llava_builder, 'load_pretrained_model')):
        monkeypatch.setattr(module, name, getattr(module, name))
    target = f'mcitlib.methods._{kind}.eval_runners.model_others'
    monkeypatch.setenv('MCITLIB_METHOD', kind)
    monkeypatch.setenv('MCITLIB_SELFCHECK', '1')
    monkeypatch.setattr(sys, 'argv', ['entry', target])
    entry.main()
    builder = importlib.import_module(f'mcitlib.methods._{kind}.llava_overrides.builder')
    runner = importlib.import_module(target)
    assert runner.load_pretrained_model is builder.load_pretrained_model
    assert Path(builder.__file__).resolve() == (
        Path(__file__).resolve().parents[2] / f'mcitlib/methods/_{kind}/llava_overrides/builder.py'
    )

    class Tower:
        is_loaded = True
        image_processor = object()

        def to(self, **kwargs):
            assert kwargs['device'] == 'cpu'
            return self

    class Tokenizer:
        def add_tokens(self, *args, **kwargs):
            pass

        def __len__(self):
            return 32

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == 'disco':
                self.bank = disco.MCITDiscoBankA(4, 4, 2, 0, False)
            else:
                self.bank = hide.MCITHiDeBankA(4, 4, 2, 1, False, 3, [])
            self.config = SimpleNamespace(mm_use_im_patch_token=False, max_sequence_length=64)
            self.lm_head = nn.Linear(4, 32, bias=False)
            self.vision_tower, self.text_tower = Tower(), Tower()
            self.eval()

        def set_clip_tokenizer(self, tokenizer):
            self.clip_tokenizer = tokenizer

        def set_tokenizer(self, tokenizer):
            self.tokenizer = tokenizer

        def set_eval(self, num_task):
            self.num_task = num_task

        def resize_token_embeddings(self, size):
            assert size == 32

        def get_vision_tower(self):
            return self.vision_tower

        def get_text_tower(self):
            return self.text_tower

        def generate(self, x, fail=False):
            with torch.no_grad():
                first = self.bank(x)
                if fail:
                    raise ValueError('generation failed')
                return first, self.bank(x)

    model, tokenizer = Model(), Tokenizer()
    config = SimpleNamespace(mm_vision_tower='test-vision')
    torch.save({}, tmp_path / 'non_lora_trainables.bin')
    monkeypatch.setattr(builder.AutoConfig, 'from_pretrained', lambda *args, **kwargs: config)
    monkeypatch.setattr(builder.AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(builder.LlavaLlamaForCausalLM, 'from_pretrained', lambda *args, **kwargs: model)
    monkeypatch.setattr(peft.PeftModel, 'from_pretrained',
                        lambda *args, **kwargs: SimpleNamespace(merge_and_unload=lambda: model))
    signature = inspect.signature(model.generate)
    actual_tokenizer, actual_model, image_processor, context_len = runner.load_pretrained_model(
        str(tmp_path), 'test-base', 'llava-lora', device='cpu', num_task=2, text_tower='test-text',
    )
    assert (actual_tokenizer, actual_model, image_processor, context_len) == (
        tokenizer, model, model.vision_tower.image_processor, 64,
    )
    assert model.num_task == 2 and inspect.signature(model.generate) == signature
    assert config.mm_text_tower == 'test-text'
    values = torch.randn(1, 4)
    operation = 'cat' if kind == 'disco' else 'zeros'
    original = getattr(torch, operation)
    with patch.object(torch, operation, wraps=original) as build:
        with pytest.raises(ValueError, match='generation failed'):
            model.generate(values, fail=True)
        first, second = model.generate(values)
        assert build.call_count == 2
        third, fourth = model.generate(values)
        assert build.call_count == 3
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, third)
    torch.testing.assert_close(third, fourth)
    assert cache._ACTIVE.get() is None


@pytest.mark.parametrize('kind', ['disco', 'hide'])
def test_builder_scopes_transformers_generate_on_real_tiny_llama(tmp_path, monkeypatch, kind):
    """Generate multiple tokens through actual Llama and its official expert layers."""
    import importlib
    from types import SimpleNamespace

    import peft
    import transformers

    from mcitlib.backbones.llava.overrides import import_llava

    import_llava()
    builder = importlib.import_module(f'mcitlib.methods._{kind}.llava_overrides.builder')
    variant = disco if kind == 'disco' else hide

    class Tower:
        is_loaded = True
        image_processor = object()

        def to(self, **kwargs):
            return self

    class Tokenizer:
        def __len__(self):
            return 32

    class TinyLlama(transformers.LlamaForCausalLM):
        def __init__(self, config):
            super().__init__(config)
            self.vision_tower, self.text_tower = Tower(), Tower()

        def set_clip_tokenizer(self, tokenizer):
            self.clip_tokenizer = tokenizer

        def set_tokenizer(self, tokenizer):
            self.tokenizer = tokenizer

        def set_eval(self, num_task):
            self.num_task = num_task
            self.eval()

        def get_vision_tower(self):
            return self.vision_tower

        def get_text_tower(self):
            return self.text_tower

    torch.manual_seed(888)
    config = transformers.LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32,
        bos_token_id=1, eos_token_id=None, pad_token_id=0, use_cache=True,
    )
    config.mm_use_im_patch_token = False
    model = TinyLlama(config)
    old_projection = model.model.layers[0].self_attn.q_proj
    cls = disco.MCITDiscoLoraLinear if kind == 'disco' else hide.MCITHiDeLoraLinear
    options = {'expert_num': 2, 'cur_task': 1, 'train_signal': False, 'bias': False}
    if kind == 'hide':
        options.update(layer=0, expert_weight=[0.3, 0.7])
    projection = cls('default', 16, 16, r=4, lora_alpha=4, lora_dropout=0, **options)
    with torch.no_grad():
        projection.weight.copy_(old_projection.weight)
        for parameter in projection.lora_B.parameters():
            parameter.normal_(0, 0.1)
    model.model.layers[0].self_attn.q_proj = projection
    model.eval()
    tokenizer = Tokenizer()
    torch.save({}, tmp_path / 'non_lora_trainables.bin')
    monkeypatch.setattr(builder.AutoConfig, 'from_pretrained', lambda *args, **kwargs: config)
    monkeypatch.setattr(builder.AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(builder.LlavaLlamaForCausalLM, 'from_pretrained', lambda *args, **kwargs: model)
    monkeypatch.setattr(peft.PeftModel, 'from_pretrained',
                        lambda *args, **kwargs: SimpleNamespace(merge_and_unload=lambda: model))
    signature = inspect.signature(model.generate)
    _, loaded, _, _ = builder.load_pretrained_model(
        str(tmp_path), 'tiny-base', 'llava-lora', device='cpu', num_task=2, text_tower='tiny-text',
    )
    assert loaded is model and inspect.signature(loaded.generate) == signature
    assert getattr(loaded.generate, '_mcit_expert_inference_scope', False)

    original_expert_weight = variant.expert_weight
    builds = []

    def counted(bank, parameters, x, *, extra, build):
        def build_once():
            builds.append(extra[0])
            return build()

        return original_expert_weight(bank, parameters, x, extra=extra, build=build_once)

    monkeypatch.setattr(variant, 'expert_weight', counted)
    ids = torch.tensor([[1, 5, 7]])
    with torch.no_grad():
        first = loaded.generate(ids, do_sample=False, max_new_tokens=3)
        assert len(builds) == 2
        second = loaded.generate(ids, do_sample=False, max_new_tokens=3)
    assert first.shape == (1, 6) and torch.equal(first, second)
    assert len(builds) == 4 and cache._ACTIVE.get() is None
