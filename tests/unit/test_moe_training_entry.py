# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Exercise the real shared training branch and adapters on a tiny CPU model."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("deepspeed")

from mcitlib.backbones.llava import moe_train  # noqa: E402
from mcitlib.methods.clmoe import CLMoEMethod  # noqa: E402
from mcitlib.methods.disco import DISCOMethod  # noqa: E402
from mcitlib.methods.hide import HiDeMethod  # noqa: E402
from mcitlib.methods.moelora import MoELoRAMethod  # noqa: E402
from mcitlib.methods.olora import OLoRAMethod  # noqa: E402
from mcitlib.methods.smolora import SMoLoRAMethod  # noqa: E402


class StopAfterAdapter(Exception):
    pass


@pytest.mark.parametrize(
    ("method", "method_class", "cur_task", "task_label", "expected_context", "expert_num"),
    [
        ("moelora", MoELoRAMethod, 0, "", 1, 6),
        ("moelora", MoELoRAMethod, 1, "", 2, 6),
        ("olora", OLoRAMethod, 0, "", 1, 6),
        ("olora", OLoRAMethod, 5, "", 6, 6),
        ("olora", OLoRAMethod, 4, "", 5, 5),
        ("disco", DISCOMethod, 0, "", 1, 6),
        ("disco", DISCOMethod, 1, "", 2, 6),
        ("hide", HiDeMethod, 0, "", 1, 6),
        ("hide", HiDeMethod, 5, "", 6, 6),
        ("hide", HiDeMethod, 4, "", 5, 5),
        ("smolora", SMoLoRAMethod, 0, "", 1, 6),
        ("smolora", SMoLoRAMethod, 1, "", 2, 6),
        ("clmoe", CLMoEMethod, 0, "UCIT_ImgNet", 1, 6),
        ("clmoe", CLMoEMethod, 0, "DCL_RS", 1, 5),
        ("clmoe", CLMoEMethod, 0, "DCL_Med", 2, 5),
    ],
)
def test_real_training_branch_builds_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, method_class: type,
    cur_task: int, task_label: str, expected_context: int, expert_num: int,
) -> None:
    config = transformers.LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32,
    )
    if method == "hide":
        from mcitlib.methods._hide.llava_overrides import LlavaLlamaForCausalLM

        model = LlavaLlamaForCausalLM(config)
    else:
        model = transformers.LlamaForCausalLM(config)
    model_args = moe_train.ModelArguments(
        model_name_or_path="tiny-local", vision_tower="tiny-vision" if method == "hide" else None,
        cur_task=cur_task,
        task=task_label, expert_num=expert_num, task_embedding_dim=12, ins_type=cur_task,
        ins_emb=str(Path(__file__).resolve().parents[2] / "assets/smolora/ins_emb_multi_ucit.pkl"),
    )
    training_args = SimpleNamespace(
        local_rank=0, fp16=False, bf16=False, bits=16, gradient_checkpointing=False,
        lora_enable=True, lora_r=3 * expert_num, lora_alpha=6 * expert_num,
        lora_dropout=0.0, lora_bias="none",
        output_dir=str(tmp_path / "checkpoint"), cache_dir=str(tmp_path), model_max_length=32,
    )
    parser = SimpleNamespace(parse_args_into_dataclasses=lambda: (model_args, SimpleNamespace(), training_args))
    monkeypatch.setattr(moe_train.transformers, "HfArgumentParser", lambda *_: parser)
    model_class = moe_train.LlavaLlamaForCausalLM if method == "hide" else moe_train.transformers.LlamaForCausalLM
    monkeypatch.setattr(model_class, "from_pretrained", lambda *_, **__: model)
    monkeypatch.setenv("MCITLIB_METHOD", method)
    monkeypatch.delenv("MCITLIB_MODEL_OVERRIDE", raising=False)

    observed = {}
    original_before_task = method_class.before_task
    original_build = method_class.build_trainable

    def before_task(instance, ctx):
        observed["context"] = ctx
        observed["method"] = instance
        return original_before_task(instance, ctx)

    def build_trainable(instance, base_model, backbone):
        adapted = original_build(instance, base_model, backbone)
        observed["model"] = adapted
        return adapted

    monkeypatch.setattr(method_class, "before_task", before_task)
    monkeypatch.setattr(method_class, "build_trainable", build_trainable)

    def stop_at_tokenizer(*_, **__):
        raise StopAfterAdapter

    monkeypatch.setattr(moe_train.transformers.AutoTokenizer, "from_pretrained", stop_at_tokenizer)
    with pytest.raises(StopAfterAdapter):
        moe_train.train()
    assert observed["context"].task_id == expected_context
    assert observed["context"].output_dir == tmp_path / "checkpoint"
    assert observed["method"].cfg["cur_task"] == cur_task
    assert model_args.task == task_label
    assert any("lora_" in name for name, _ in observed["model"].named_parameters())
    adapter_config = observed["model"].peft_config["default"]
    assert adapter_config.expert_num == expert_num
    if method != "smolora":
        assert adapter_config.task_embedding_dim == 12
    if method in {"olora", "hide", "disco"}:
        assert adapter_config.cur_task == cur_task
    observed["model"].train()
    trainable = [(name, parameter) for name, parameter in observed["model"].named_parameters()
                 if parameter.requires_grad]
    initial = {name: parameter.detach().clone() for name, parameter in trainable}
    optimizer = torch.optim.SGD([parameter for _, parameter in trainable], lr=0.01)
    ids = torch.tensor([[1, 2, 3]])
    output = observed["model"](input_ids=ids, labels=ids)
    assert output.logits.shape == (1, 3, 32) and torch.isfinite(output.loss)
    output.loss.backward()
    optimizer.step()
    assert any(not torch.equal(initial[name], parameter.detach()) for name, parameter in trainable)
    if method == "clmoe":
        assert observed["method"].router_stats.task_id == expected_context
    if method == "hide":
        tuner = observed["model"].base_model
        assert tuner.expert_weight == [0.0] * 8
        layers = [layer for layer in tuner.modules() if type(layer).__name__ == "MCITHiDeLoraLinear"]
        assert layers and all(layer.expert_weight is model.expert_weight for layer in layers)
