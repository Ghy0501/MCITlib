# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""CPU regression for Octopus external regularization with checkpointing."""
import copy
import unittest
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
checkpoint_module = pytest.importorskip("torch.utils.checkpoint")

from mcitlib.backbones.llava.olora_checkpoint import (  # noqa: E402
    non_reentrant_checkpoint_scope,
    octopus_checkpoint_scope,
    olora_checkpoint_scope,
)
from mcitlib.methods._octopus.core import regularization  # noqa: E402


class TinyLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([[0.4, -0.7, 0.3], [0.2, 0.6, -0.5]], dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor([[0.2, -0.3], [0.8, 0.5]], dtype=torch.float64))
        self.register_buffer('weight', torch.tensor([[0.4, 0.1, 0.6], [-0.2, 0.7, 0.3]], dtype=torch.float64))

    def block(self, x):
        return torch.sin(x @ self.weight.T + (x @ self.a.T) @ self.b.T)

    def forward(self, x, checkpointed):
        if checkpointed:
            return checkpoint_module.checkpoint(self.block, x)
        return self.block(x)


class OctopusCheckpointTest(unittest.TestCase):
    def test_pinned_llama_checkpoint_with_external_parameter_penalty(self):
        # Transformers 4.31's real LlamaModel calls torch.checkpoint directly.
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(42)
        config = LlamaConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
                             num_hidden_layers=2, num_attention_heads=2, use_cache=False)
        original = LlamaForCausalLM(config).float().train()
        prompt = torch.tensor([[1, 2, 3, 4]])
        a = torch.eye(8, dtype=torch.float32)
        gpwc = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 100 + 0.01

        def run(mode):
            model = copy.deepcopy(original)
            weight = model.model.layers[0].self_attn.q_proj.weight
            hooks = []
            weight.register_hook(lambda grad: hooks.append(grad.detach().clone()))
            if mode != 'eager':
                model.gradient_checkpointing_enable()
            def backward():
                task_loss = model(input_ids=prompt, labels=prompt).loss
                penalty, _, _ = regularization([(a, weight, gpwc)], 0.2, 0.1)
                loss = task_loss / 2 + penalty
                loss.backward()
                return loss.detach()
            if mode == 'non_reentrant':
                with octopus_checkpoint_scope('octopus', '2', SimpleNamespace(gradient_checkpointing=True)) as obs:
                    loss = backward()
                    self.assertEqual(obs.calls, 2)
            else:
                loss = backward()
            return loss, weight.grad, len(hooks)

        eager, scoped, reentrant = run('eager'), run('non_reentrant'), run('reentrant')
        torch.testing.assert_close(scoped[0], eager[0], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(scoped[1], eager[1], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(reentrant[1], eager[1], rtol=1e-6, atol=1e-7)
        self.assertEqual((eager[2], scoped[2], reentrant[2]), (1, 1, 2))

    def test_external_regularizer_gradient_matches_eager_and_hook_fires_once(self):
        base = TinyLoRA()
        x = torch.tensor([[0.3, 0.5, -0.1], [0.6, -0.2, 0.4]], dtype=torch.float64)
        gradient = torch.tensor([[0.5, 0.7, -0.9], [-0.1, 0.3, 0.8]], dtype=torch.float64)

        def run(mode):
            model = copy.deepcopy(base)
            values = x.detach().clone().requires_grad_(True)
            hook_calls = []
            model.b.register_hook(lambda grad: hook_calls.append(grad.detach().clone()))
            args = SimpleNamespace(gradient_checkpointing=True)

            def backward():
                result = model(values, checkpointed=(mode != 'eager'))
                task_loss = (result ** 2).mean()
                penalty, _, _ = regularization([(model.a, model.b, gradient)], 0.2, 0.1)
                # Released placement: task loss divided by accumulation, plus penalty.
                loss = task_loss / 4 + penalty
                loss.backward()
                return loss.detach()

            if mode == 'non_reentrant':
                with non_reentrant_checkpoint_scope(args, label='Octopus') as observation:
                    loss = backward()
                    self.assertEqual(observation.calls, 1)
            else:
                loss = backward()
            return loss, model.a.grad, model.b.grad, values.grad, len(hook_calls)

        eager, scoped, reentrant = run('eager'), run('non_reentrant'), run('reentrant')
        for index in range(4):
            torch.testing.assert_close(scoped[index], eager[index], rtol=1e-11, atol=1e-11)
            torch.testing.assert_close(reentrant[index], eager[index], rtol=1e-11, atol=1e-11)
        # ZeRO-2 rejects the two hooks fired by the old reentrant computation.
        self.assertEqual(eager[4], 1)
        self.assertEqual(scoped[4], 1)
        self.assertEqual(reentrant[4], 2)
        model = copy.deepcopy(base)
        penalty, _, _ = regularization([(model.a, model.b, gradient)], 0.2, 0.1)
        a_grad, b_grad = torch.autograd.grad(penalty, (model.a, model.b), allow_unused=True)
        self.assertIsNone(a_grad)
        self.assertGreater(b_grad.abs().sum().item(), 0)

    def test_scope_guards_restoration_and_olora_behavior(self):
        original = checkpoint_module.checkpoint
        args = SimpleNamespace(gradient_checkpointing=True)
        with self.assertRaisesRegex(ValueError, 'intentional'):
            with non_reentrant_checkpoint_scope(args, label='Octopus'):
                raise ValueError('intentional')
        self.assertIs(checkpoint_module.checkpoint, original)

        self.assertEqual(args.gradient_checkpointing_kwargs, {'use_reentrant': False})
        with self.assertRaisesRegex(RuntimeError, 'conflicting'):
            with non_reentrant_checkpoint_scope(
                SimpleNamespace(gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': True}),
                label='Octopus',
            ):
                pass
        unrelated = SimpleNamespace(id='other')
        with olora_checkpoint_scope(unrelated, SimpleNamespace(gradient_checkpointing=True)) as obs:
            self.assertIsNone(obs)
        olora = SimpleNamespace(id='olora', training_overrides=lambda: {'gradient_checkpointing_kwargs': {'use_reentrant': False}})
        with olora_checkpoint_scope(olora, SimpleNamespace(gradient_checkpointing=True)) as obs:
            value = torch.tensor([2.0], requires_grad=True)
            checkpoint_module.checkpoint(lambda x: x.sin(), value).sum().backward()
            self.assertEqual(obs.calls, 1)
        self.assertIs(checkpoint_module.checkpoint, original)

    def test_octopus_stage_dispatch_and_other_methods(self):
        original = checkpoint_module.checkpoint
        for method, stage, active in [('octopus', '1', True), ('octopus', '2', True),
                                      ('octopus', '3', False), ('olora', '2', False),
                                      ('dgg', '2', False), ('pclr', '2', False), (None, None, False)]:
            args = SimpleNamespace(gradient_checkpointing=True)
            with self.subTest(method=method, stage=stage):
                with octopus_checkpoint_scope(method, stage, args) as observation:
                    self.assertEqual(observation is not None, active)
                    value = torch.tensor([1.0], requires_grad=True)
                    checkpoint_module.checkpoint(lambda x: x.sin(), value).sum().backward()
                    if active:
                        self.assertEqual(observation.calls, 1)
                self.assertIs(checkpoint_module.checkpoint, original)


if __name__ == '__main__':
    unittest.main()
