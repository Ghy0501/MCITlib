# SPDX-License-Identifier: Apache-2.0
"""SAME prototype lifecycle and actual PEFT outer-forward behavior on CPU."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from mcitlib.methods._same.prior import SIDECAR, PriorRuntime, PriorState, clip_snapshot_identity
from mcitlib.methods.same import SAMEMethod
from mcitlib.tuners._peft_registry import install_method_adapter
from mcitlib.tuners.variants.same import SAMELinear


def original_router_methods():
    source = os.environ.get('MCITLIB_PRISM_ROUTER_SOURCE')
    if not source:
        pytest.skip('set MCITLIB_PRISM_ROUTER_SOURCE to compare original PRISM methods')
    data = Path(source).read_bytes()
    assert hashlib.sha256(data).hexdigest() == '52b1ebfc400dbc0f0b93c7d755f01038a4bb0649e547b962a09d0fa42371d3e6'
    tree = ast.parse(data.decode(), filename=source)
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'RouterIntegration')
    wanted = {'_ensure_prototypes_on_model', '_update_running_prototypes',
              '_compute_cosine_similarities', '_similarities_to_mixture'}
    node = copy.deepcopy(cls)
    node.bases, node.keywords, node.decorator_list = [], [], []
    node.body = [x for x in node.body if isinstance(x, ast.FunctionDef) and x.name in wanted]
    ast.fix_missing_locations(node)
    ns = {'torch': torch, 'F': F}
    # Match the source's deferred annotations without importing PRISM internals.
    import __future__
    exec(compile(ast.Module(body=[node], type_ignores=[]), source, 'exec',
                 flags=__future__.annotations.compiler_flag), ns)
    return ns['RouterIntegration']


def test_source_anchor_initialization_update_and_mixture():
    original = original_router_methods()
    ref = original()
    ref.task_num = 3
    ref.feature_dim = 4
    ref.mixture_logit_scale = 28.0
    ref.image_anchors = ref.text_anchors = ref.image_boundary = ref.text_boundary = None
    model = nn.Linear(1, 1)
    torch.manual_seed(1947)
    ref._ensure_prototypes_on_model(model)
    torch.manual_seed(1947)
    state = PriorState(3, 4)
    assert torch.equal(state.image_anchors, torch.cat(list(ref.image_anchors)))
    assert torch.equal(state.text_anchors, torch.cat(list(ref.text_anchors)))
    image = torch.tensor([[1., 2., 3., 4.], [2., 1., 0., 1.]])
    text = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    ref._update_running_prototypes(image, text, 1)
    state.update(image, text, 1)
    assert torch.equal(state.image_anchors, torch.cat(list(ref.image_anchors)))
    assert torch.equal(state.text_anchors, torch.cat(list(ref.text_anchors)))
    assert torch.equal(state.image_boundary, torch.cat(list(ref.image_boundary)))
    assert torch.equal(state.text_boundary, torch.cat(list(ref.text_boundary)))
    for image_or_none in (image, None):
        expected = ref._similarities_to_mixture(ref._compute_cosine_similarities(image_or_none, text, text.device))
        torch.testing.assert_close(state.mixture(image_or_none, text), expected, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('prior', [[1., 1., 1.], [.76, .19, .05]])
def test_actual_layer_matches_source_topk_and_forward(dtype, prior):
    source = os.environ.get('MCITLIB_PRISM_ROUTER_SOURCE')
    if not source:
        pytest.skip('set MCITLIB_PRISM_ROUTER_SOURCE to compare original PRISM methods')
    path = Path(source).parents[2] / 'PEFT/tuners/custom/same.py'
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == '04d2b35fb56a43879b729543035d9a392ef466bb0c4e67719875f1f776511e1d'
    tree = ast.parse(data.decode(), filename=str(path))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'SAMELinear')
    node = copy.deepcopy(cls)
    node.bases, node.keywords, node.decorator_list = [], [], []
    node.body = [x for x in node.body if isinstance(x, ast.FunctionDef) and x.name == '_apply_topk_sparsification']
    ast.fix_missing_locations(node)
    ns = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    torch.manual_seed(56)
    layer = SAMELinear('default', 4, 5, r=6, lora_alpha=12, expert_num=3).to(dtype).eval()
    with torch.no_grad():
        for expert in layer.lora_B['default'].loraB:
            expert.weight.normal_()
    layer.set_global_prior(torch.tensor(prior))
    x = torch.randn(2, 7, 4, dtype=dtype)
    means = (layer.lora_router['default'](x.reshape(-1, 4)) / 3).softmax(-1).mean(0)
    route = ns['SAMELinear']._apply_topk_sparsification(layer, means, layer.router, k=2).to(dtype)
    a = sum(route[i] * expert(x) for i, expert in enumerate(layer.lora_A['default'].loraA))
    b = sum(route[i] * expert(a) for i, expert in enumerate(layer.lora_B['default'].loraB))
    expected = F.linear(x, layer.weight, layer.bias) + b * layer.scaling['default']
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)


class Encoding(dict):
    def to(self, device):
        return Encoding({k: v.to(device) for k, v in self.items()})


class MainTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.seen = None

    def batch_decode(self, tokens, skip_special_tokens):
        self.seen = tokens.copy()
        return ['SYSTEM\nuser question ASSISTANT hidden'] * len(tokens)


class ClipTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.seen = None

    def __call__(self, values, **kwargs):
        self.seen = (values, kwargs)
        return Encoding(input_ids=torch.tensor([[1, 2, 3, 4]] * len(values)))


class TextTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1), requires_grad=False)
        self.modes = []

    def forward(self, input_ids):
        self.modes.append(self.training)
        return SimpleNamespace(pooler_output=input_ids.to(self.weight.dtype))


class ImageTower(nn.Module):
    def __init__(self, fails=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1), requires_grad=False)
        self.modes = []
        self.fails = fails

    def forward(self, pixel_values):
        self.modes.append(self.training)
        if self.fails:
            raise RuntimeError('deliberate image extraction failure')
        return SimpleNamespace(image_embeds=pixel_values.to(self.weight.dtype))


class TinyLlama(LlamaForCausalLM):
    """PEFT still wraps this real Llama; layer checkpointing is observable."""

    def forward(self, input_ids=None, images=None, labels=None, **kwargs):
        from torch.utils.checkpoint import checkpoint

        x = self.model.embed_tokens(input_ids.clamp_min(0)).detach().requires_grad_(True)
        y = checkpoint(self.model.layers[0].mlp.gate_proj, x, use_reentrant=False)
        return SimpleNamespace(loss=y.float().square().sum())


def tiny_model():
    config = LlamaConfig(vocab_size=40, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=32)
    install_method_adapter('same')
    method = SAMEMethod({'r': 6, 'lora_alpha': 12, 'expert_num': 3, 'cur_task': 0})
    return method, method.build_trainable(TinyLlama(config), None)


def runtime_for(state, *, task=0, image_fails=False):
    return PriorRuntime(state, MainTokenizer(), ClipTokenizer(), TextTower(), ImageTower(image_fails), task)


@pytest.fixture
def clip_snapshot(tmp_path):
    path = tmp_path / 'clip'
    path.mkdir()
    for name, data in {'config.json': b'{}', 'tokenizer.json': b'{}',
                       'preprocessor_config.json': b'{}', 'pytorch_model.bin': b'clip weights'}.items():
        (path / name).write_bytes(data)
    return path


def test_real_peft_outer_hook_once_with_checkpoint_recompute():
    method, model = tiny_model()
    model.train()
    state = PriorState(3, 4)
    runtime = runtime_for(state)
    runtime.install_training_hook(model)
    tokens = torch.tensor([[1, 2, 3, 4]])
    loss = model(input_ids=tokens, images=torch.ones(1, 4), labels=tokens).loss
    loss.backward()
    assert state.text_boundary.tolist() == [2., 1., 1.]
    assert state.image_boundary.tolist() == [2., 1., 1.]
    assert runtime.tokenizer.seen.tolist() == tokens.tolist()
    assert runtime.clip_tokenizer.seen[0] == ['user question']
    assert runtime.clip_tokenizer.seen[1] == {'padding': 'longest', 'max_length': 77,
                                               'truncation': True, 'return_tensors': 'pt'}
    assert runtime.text_tower.modes == [True] and runtime.image_tower.modes == [True]
    layer = next(m for m in model.modules() if isinstance(m, SAMELinear))
    assert layer.state['step'].item() == 1
    assert layer.router.dtype == torch.float32 and layer.router.tolist() == [1., 0., 0.]


def test_generate_prior_refresh_cache_and_text_only_fallback():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = SAMELinear('default', 4, 4, r=6, lora_alpha=12, expert_num=3)
            self.seen = []

        def generate(self, input_ids, images=None):
            self.seen.append(self.layer.router.clone())
            return self.layer(torch.ones(1, 4))

    state = PriorState(3, 4)
    state.image_anchors.copy_(torch.eye(3, 4))
    state.text_anchors.copy_(torch.eye(3, 4))
    model = TinyModel().eval()
    runtime = runtime_for(state, image_fails=True)
    runtime.install_generate_hook(model)
    tokens = torch.tensor([[1, -200, 3, 4]])
    model.generate(tokens, images=torch.ones(1, 4))
    assert runtime.tokenizer.seen.tolist() == [[1, 0, 3, 4]]
    expected = state.mixture(None, torch.tensor([[1., 2., 3., 4.]]))
    torch.testing.assert_close(model.seen[0], expected)
    assert runtime.text_tower.modes == [False]
    assert state.image_boundary.tolist() == [1., 1., 1.]
    model.generate(torch.tensor([[1]]), images=torch.ones(1, 4))
    assert torch.equal(model.seen[1], model.seen[0])
    assert runtime.text_tower.modes == [False]  # no second feature extraction


def test_builder_restores_binds_and_wraps_generation(tmp_path, clip_snapshot, monkeypatch):
    import llava.model.builder as builder

    from mcitlib.methods._same import inference

    checkpoint = tmp_path / 'task0'
    checkpoint.mkdir()
    (checkpoint / 'adapter_config.json').write_text(json.dumps({'expert_num': 3, 'cur_task': 0}))
    state = PriorState(3, 4)
    state.bind_clip(clip_snapshot)
    state.update(torch.ones(1, 4), torch.ones(1, 4), 0)
    state.complete_task(0)
    state.save(checkpoint)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = SAMELinear('default', 4, 4, r=6, lora_alpha=12, expert_num=3)
            self.config = SimpleNamespace(mm_vision_tower=str(clip_snapshot))

        def generate(self, input_ids, images=None):
            return self.layer(torch.ones(1, 4))

    called = []

    def original_builder(*args, **kwargs):
        called.append(True)
        return MainTokenizer(), TinyModel(), object(), 32

    def runtime_builder(restored, tokenizer, clip_path, *, device, dtype, task_id):
        restored.bind_clip(clip_path)
        assert task_id == 0
        assert torch.equal(restored.text_anchors, state.text_anchors)
        return PriorRuntime(restored, tokenizer, ClipTokenizer(), TextTower(), ImageTower(), task_id)

    monkeypatch.setattr(builder, 'load_pretrained_model', original_builder)
    monkeypatch.setattr(inference, 'load_clip_runtime', runtime_builder)
    monkeypatch.setattr(inference, 'FEATURE_DIM', 4)
    monkeypatch.setenv('MCITLIB_SAME_CLIP_TOWER', str(clip_snapshot))
    inference.install_builder()
    _, model, _, _ = builder.load_pretrained_model(str(checkpoint), None, 'fake')
    assert model.layer.router is None
    model.generate(torch.tensor([[1, 2, 3, 4]]), images=torch.ones(1, 4))
    assert model.layer.router is not None and model.layer.router.numel() == 3
    prior_calls = len(called)
    with pytest.raises(FileNotFoundError, match='lacks same_prior_state'):
        builder.load_pretrained_model(str(tmp_path / 'legacy'), None, 'fake')
    (checkpoint / 'adapter_config.json').write_text(json.dumps({'expert_num': 3, 'cur_task': 1}))
    with pytest.raises(ValueError, match='completed task does not match'):
        builder.load_pretrained_model(str(checkpoint), None, 'fake')
    assert len(called) == prior_calls  # neither failure reached the model builder
    (checkpoint / 'adapter_config.json').write_text(json.dumps({'expert_num': 3, 'cur_task': 0}))
    different = tmp_path / 'different-clip'
    shutil.copytree(clip_snapshot, different)
    (different / 'pytorch_model.bin').write_bytes(b'different weights')
    monkeypatch.setenv('MCITLIB_SAME_CLIP_TOWER', str(different))
    with pytest.raises(ValueError, match='CLIP snapshot content does not match'):
        builder.load_pretrained_model(str(checkpoint), None, 'fake')


def test_strict_sidecar_roundtrip_and_legacy_rejection(tmp_path, clip_snapshot):
    state = PriorState(3, 4)
    state.bind_clip(clip_snapshot)
    state.update(torch.ones(1, 4), torch.ones(1, 4), 0)
    state.prior = torch.tensor([1., 0., 0.])
    state.complete_task(0)
    state.save(tmp_path)
    restored = PriorState(3, 4)
    restored.bind_clip(clip_snapshot)
    restored.load(tmp_path)
    restored.require_observed_through(0)
    for key in ('image_anchors', 'text_anchors', 'image_boundary', 'text_boundary', 'prior'):
        assert torch.equal(getattr(restored, key), getattr(state, key))
    with pytest.raises(ValueError, match='completed task does not match'):
        restored.require_observed_through(1)
    other = tmp_path / 'relocated-clip'
    shutil.copytree(clip_snapshot, other)
    restored.bind_clip(other)
    assert clip_snapshot_identity(other) == state.clip_identity
    (other / 'pytorch_model.bin').write_bytes(b'different clip weights')
    different = PriorState(3, 4)
    different.load(tmp_path)
    with pytest.raises(ValueError, match='CLIP snapshot content does not match'):
        different.bind_clip(other)
    with pytest.raises(FileNotFoundError, match='historical image/text anchors'):
        PriorState(3, 4).load(tmp_path / 'legacy')
    payload = torch.load(tmp_path / SIDECAR, weights_only=True)
    payload.pop('text_anchors')
    torch.save(payload, tmp_path / SIDECAR)
    with pytest.raises(ValueError, match='text_anchors'):
        PriorState(3, 4).load(tmp_path)


def _distributed_worker(rank, init_path, artifact_dir, clip_path):
    import torch.distributed as dist

    dist.init_process_group('gloo', init_method='file://' + init_path, rank=rank, world_size=2)
    try:
        torch.manual_seed(423 + rank)
        state = PriorState(3, 4)
        _, model = tiny_model()
        model.train()
        runtime = runtime_for(state)
        before = {}
        original_update = state.update

        def record(image, text, task_id):
            before['image'] = state.image_anchors.clone()
            before['text'] = state.text_anchors.clone()
            original_update(image, text, task_id)

        state.update = record
        runtime.install_training_hook(model)
        tokens = torch.tensor([[1, 2, 3, 4]])
        # Rank 1 has no image. Both ranks must broadcast before that branch.
        model(input_ids=tokens, images=torch.ones(1, 4) if rank == 0 else None, labels=tokens)
        first = {'before': before if rank == 0 else {'image': state.image_anchors.clone(),
                                                    'text': state.text_anchors.clone()},
                 'image_boundary': state.image_boundary.clone(), 'text_boundary': state.text_boundary.clone()}
        if rank == 0:
            state.bind_clip(clip_path)
            state.complete_task(0)
            state.save(Path(artifact_dir) / 'task0')
        dist.barrier()
        torch.manual_seed(178 + rank)
        continued = PriorState(3, 4)
        continued.load(Path(artifact_dir) / 'task0')
        continued.bind_clip(clip_path)
        continued.require_observed_through(0)
        _, next_model = tiny_model()
        next_model.train()
        next_runtime = runtime_for(continued, task=1)
        next_runtime.install_training_hook(next_model)
        next_model(input_ids=tokens, images=torch.full((1, 4), float(rank + 1)), labels=tokens)
        result = {'first': first, 'next_image_anchors': continued.image_anchors.clone(),
                  'next_text_boundary': continued.text_boundary.clone(),
                  'next_image_boundary': continued.image_boundary.clone()}
        torch.save(result, Path(artifact_dir) / f'rank{rank}.pt')
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_two_process_broadcast_asymmetric_inputs_and_continuation(tmp_path, clip_snapshot):
    import torch.multiprocessing as mp

    context = mp.spawn(_distributed_worker, args=(str(tmp_path / 'gloo-init'), str(tmp_path),
                                                  str(clip_snapshot)), nprocs=2, join=False)
    try:
        deadline = time.monotonic() + 35
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError('SAME data-parallel test did not finish')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join()
    rank0 = torch.load(tmp_path / 'rank0.pt', weights_only=True)
    rank1 = torch.load(tmp_path / 'rank1.pt', weights_only=True)
    assert torch.equal(rank0['first']['before']['image'], rank1['first']['before']['image'])
    assert torch.equal(rank0['first']['before']['text'], rank1['first']['before']['text'])
    assert rank0['first']['text_boundary'].tolist() == [2., 1., 1.]
    assert rank1['first']['text_boundary'].tolist() == [1., 1., 1.]
    assert rank0['next_text_boundary'].tolist() == rank1['next_text_boundary'].tolist() == [2., 2., 1.]
    assert rank0['next_image_boundary'].tolist() == rank1['next_image_boundary'].tolist() == [2., 2., 1.]
    assert torch.equal(rank0['next_image_anchors'][0], rank1['next_image_anchors'][0])
    assert not torch.equal(rank0['next_image_anchors'][1], rank1['next_image_anchors'][1])


def test_global_state_not_cast_with_model_and_counters_pass_256():
    state = PriorState(3, 4)
    model = nn.Linear(4, 4).bfloat16()
    for _ in range(300):
        state.update(None, torch.ones(1, 4, dtype=torch.bfloat16), 0)
    model.half()
    assert state.text_boundary.dtype == torch.float32
    assert state.text_boundary[0].item() == 301
    assert model.weight.dtype == torch.float16


def test_missing_layer_prior_raises():
    layer = SAMELinear('default', 4, 4, r=6, lora_alpha=12, expert_num=3).eval()
    with pytest.raises(RuntimeError, match='requires a restored global'):
        layer(torch.ones(1, 4))


@pytest.mark.parametrize('backbone', ['internvl', 'videollava'])
def test_prior_entry_rejects_other_backbones_before_installation(monkeypatch, backbone):
    from mcitlib.backbones.llava import train_entry
    from mcitlib.backbones.llava.eval_runners import entry as eval_entry

    monkeypatch.setenv('MCITLIB_METHOD', 'same')
    monkeypatch.setenv('MCITLIB_BACKBONE', backbone)
    monkeypatch.setattr('sys.argv', ['entry', 'placeholder'])
    for main in (train_entry.main, eval_entry.main):
        with pytest.raises(ValueError, match='supports only the LLaVA backbone'):
            main()
