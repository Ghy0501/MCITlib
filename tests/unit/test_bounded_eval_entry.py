# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""The smoke wrapper must bound the real runner's generation call."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers.generation.utils import GenerationMixin  # noqa: E402

from tools import bounded_eval_entry, bounded_validation_smoke  # noqa: E402


def test_generation_limit_wraps_the_selected_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    values = []
    monkeypatch.setenv("MCITLIB_SMOKE_EVAL_TARGET", "mcitlib.methods._sefe.eval_runners.model_others")
    monkeypatch.setenv("MCITLIB_SMOKE_MAX_NEW_TOKENS", "16")
    class Dummy(torch.nn.Module):
        def forward(self):
            return SimpleNamespace(logits=torch.ones(1, 1, 2))

    def generate(model, **kwargs):
        model()  # The bounded runner also verifies finite model logits.
        values.append(kwargs["max_new_tokens"])

    monkeypatch.setattr(GenerationMixin, "generate", generate)

    def run_module(target, *, run_name):
        assert target == "mcitlib.methods._sefe.eval_runners.model_others" and run_name == "__main__"
        GenerationMixin.generate(Dummy(), max_new_tokens=1024)
        GenerationMixin.generate(Dummy(), max_new_tokens=8)

    monkeypatch.setattr(bounded_eval_entry.runpy, "run_module", run_module)
    bounded_eval_entry.main()
    assert values == [16, 8]


def test_checkpoint_requires_finite_loss_positive_lr_and_changed_lora_b(tmp_path) -> None:
    output = tmp_path / "checkpoint"
    output.mkdir()
    for name in ("config.json", "adapter_config.json", "non_lora_trainables.bin"):
        (output / name).write_text("{}", encoding="utf-8")
    (output / "trainer_state.json").write_text(
        '{"global_step":2,"log_history":[{"loss":1.0,"learning_rate":0.001}]}', encoding="utf-8"
    )
    train = {"kind": "train", "argv": ["deepspeed", "mcitlib.backbones.llava.train_entry", "--output_dir",
                                        str(output), "--lora_enable", "True"]}
    stage = {"commands": [train], "checkpoint": str(output)}
    adapter = output / "adapter_model.bin"
    torch.save({"layer.lora_B.weight": torch.ones(2, 2)}, adapter)
    result = bounded_validation_smoke._check_checkpoint(stage, 2)
    assert "nonzero LoRA B" in result["update_evidence"]
    torch.save({"layer.lora_B.weight": torch.zeros(2, 2)}, adapter)
    with pytest.raises(RuntimeError, match="unchanged"):
        bounded_validation_smoke._check_checkpoint(stage, 2)
    (output / "trainer_state.json").write_text(
        '{"global_step":2,"log_history":[{"loss":NaN,"learning_rate":0.001}]}', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        bounded_validation_smoke._check_checkpoint(stage, 2)


def test_prompt_resume_compares_saved_prompt_embedding_rows(tmp_path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    weights = torch.zeros(12, 3)
    torch.save({"model.embed_tokens.weight": weights}, first / "non_lora_trainables.bin")
    torch.save({"model.embed_tokens.weight": weights.clone()}, second / "non_lora_trainables.bin")
    previous = {"checkpoint": str(first)}
    current = {"checkpoint": str(second), "commands": [{"kind": "train", "argv": [
        "--prefix_len", "2", "--num_tasks", "2"]}]}
    with pytest.raises(RuntimeError, match="did not change"):
        bounded_validation_smoke._check_prompt_resume(previous, current)
    weights[-1, 0] = 1
    torch.save({"model.embed_tokens.weight": weights}, second / "non_lora_trainables.bin")
    assert "changed" in bounded_validation_smoke._check_prompt_resume(previous, current)
