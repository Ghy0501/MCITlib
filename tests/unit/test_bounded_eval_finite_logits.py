# SPDX-License-Identifier: Apache-2.0
"""The bounded generation hook observes real logits and always removes itself."""
from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tools import bounded_eval_entry  # noqa: E402


def tiny_model():
    config = transformers.LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4)
    return transformers.LlamaForCausalLM(config).eval()


def configure(monkeypatch: pytest.MonkeyPatch, original_generate, invoke) -> None:
    from transformers.generation.utils import GenerationMixin

    monkeypatch.setenv("MCITLIB_SMOKE_EVAL_TARGET", "mcitlib.methods._disco.eval_runners.model_others")
    monkeypatch.setenv("MCITLIB_SMOKE_MAX_NEW_TOKENS", "16")
    monkeypatch.setattr(sys, "argv", ["test"])
    monkeypatch.setattr(GenerationMixin, "generate", original_generate)
    monkeypatch.setattr(bounded_eval_entry.runpy, "run_module", invoke)


def test_finite_logits_preserve_return_arguments_and_other_hooks(monkeypatch, capsys) -> None:
    from transformers.generation.utils import GenerationMixin

    model = tiny_model()
    ids = torch.tensor([[1, 2, 3]])
    expected = torch.tensor([[1, 2, 3, 4]])
    other = model.register_forward_hook(lambda *_: None)
    prior_hooks = tuple(model._forward_hooks)
    observed = []

    def original_generate(received, *args, **kwargs):
        assert received is model and args[0] is ids
        assert kwargs == {"max_new_tokens": 16, "temperature": 0.25}
        assert torch.isfinite(received(input_ids=ids).logits).all()
        assert torch.isfinite(received(input_ids=ids).logits).all()
        return expected

    def invoke(target, *, run_name):
        assert target == "mcitlib.methods._disco.eval_runners.model_others" and run_name == "__main__"
        observed.append(GenerationMixin.generate(model, ids, max_new_tokens=256, temperature=0.25))

    configure(monkeypatch, original_generate, invoke)
    try:
        bounded_eval_entry.main()
        assert observed[0] is expected
        assert tuple(model._forward_hooks) == prior_hooks
        assert "checked finite logits for 2 generation step(s)" in capsys.readouterr().out
    finally:
        other.remove()


def test_nan_logits_fail_and_hook_is_removed(monkeypatch) -> None:
    from transformers.generation.utils import GenerationMixin

    model = tiny_model()
    ids = torch.tensor([[1, 2, 3]])
    prior_hooks = tuple(model._forward_hooks)

    def original_generate(received, *args, **kwargs):
        assert kwargs["max_new_tokens"] == 16
        received(input_ids=ids)
        with torch.no_grad():
            received.lm_head.weight.fill_(float("nan"))
        received(input_ids=ids)
        pytest.fail("non-finite logits did not stop generation")

    def invoke(_target, *, run_name):
        GenerationMixin.generate(model, ids, max_new_tokens=16)

    configure(monkeypatch, original_generate, invoke)
    with pytest.raises(RuntimeError, match="non-finite logits"):
        bounded_eval_entry.main()
    assert tuple(model._forward_hooks) == prior_hooks
    assert torch.isnan(model(input_ids=ids).logits).all()
