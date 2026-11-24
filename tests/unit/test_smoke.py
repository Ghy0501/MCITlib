# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CPU-only tests of the smoke guardrails; these are not GPU run evidence."""
from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mcitlib.backbones.llava.smoke_infer import generate_one
from mcitlib.cli import main
from mcitlib.doctor import FAIL, OK, Report
from mcitlib.smoke import _first_records, _run, build_plan, execute_plan


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch, repo_root: Path) -> Path:
    project = tmp_path / "project"
    (project / "logs").mkdir(parents=True)
    (project / "configs").symlink_to(repo_root / "configs", target_is_directory=True)
    monkeypatch.setattr("mcitlib.smoke.repo_root", lambda: project)
    config = tmp_path / "paths.yaml"
    config.write_text(yaml.safe_dump({"model_root": str(tmp_path / "models"), "data_root": str(tmp_path / "data"),
                                      "ckpt_root": str(tmp_path / "official-ckpts"),
                                      "results_root": str(tmp_path / "official-results")}))
    data = tmp_path / "data/UCIT/ImageNet-R"
    images = tmp_path / "data/UCIT"
    data.mkdir(parents=True)
    for i in range(5):
        (images / f"{i}.jpg").write_bytes(b"image")
    records = [{"id": str(i), "image": f"{i}.jpg", "conversations": [
        {"from": "human", "value": "<image>\nWhat is this?"}, {"from": "gpt", "value": "answer"}]} for i in range(5)]
    (data / "train.json").write_text(json.dumps(records))
    dcl = tmp_path / "data/MLLM-CL/RS"
    dcl.mkdir(parents=True)
    for i in range(5):
        (dcl / f"{i}.jpg").write_bytes(b"image")
    (dcl / "train.json").write_text(json.dumps(records))
    return config


def flag(argv: list[str], name: str) -> str:
    assert argv.count(name) == 1
    return argv[argv.index(name) + 1]


@pytest.mark.parametrize("benchmark", ["ucit", "mllm_dcl"])
def test_preview_is_read_only_and_uses_only_bounded_task1(inputs: Path, tmp_path: Path, monkeypatch, capsys,
                                                           benchmark: str) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("preview must not run doctor or processes")

    monkeypatch.setattr("mcitlib.doctor.run_doctor", unexpected)
    monkeypatch.setattr("mcitlib.smoke.subprocess.run", unexpected)
    run_dir = tmp_path / "project/logs/preview"
    assert main(["smoke", "--benchmark", benchmark, "--paths", str(inputs), "--run-dir", str(run_dir)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "preview" and not run_dir.exists()
    assert plan["benchmark"] == benchmark
    expected = "data/UCIT/ImageNet-R/train.json" if benchmark == "ucit" else "data/MLLM-CL/RS/train.json"
    assert plan["source_data"] == str(tmp_path / expected)
    assert plan["task"] == 1 and plan["limits"] == {"samples": 4, "max_steps": 1, "batch_per_device": 1,
        "gradient_accumulation_steps": 1, "workers": 0, "inference_samples": 1, "max_new_tokens": 8}
    train, infer = plan["training_command"], plan["inference_command"]
    argv = train["argv"]
    assert train["kind"] == "train" and train["task_id"] == 1
    assert flag(argv, "--module") == "mcitlib.backbones.llava.train_entry"
    assert flag(argv, "--data_path") == str(run_dir / "train-subset.json")
    assert flag(argv, "--output_dir") == str(run_dir / "Task1_llava_lora")
    for name, value in (("--max_steps", "1"), ("--per_device_train_batch_size", "1"),
                        ("--gradient_accumulation_steps", "1"), ("--dataloader_num_workers", "0"),
                        ("--save_strategy", "no")):
        assert flag(argv, name) == value
    assert infer["argv"][3] == "mcitlib.backbones.llava.smoke_infer"
    assert flag(infer["argv"], "--benchmark") == benchmark
    assert flag(infer["argv"], "--adapter") == flag(argv, "--output_dir")
    assert flag(infer["argv"], "--subset") == flag(argv, "--data_path")
    assert not any("Task2" in token or "eval_math" in token for token in argv + infer["argv"])


@pytest.mark.parametrize("samples,steps", [(0, 1), (9, 1), (1, 0), (1, 4)])
def test_bounds_rejected_before_execution(inputs: Path, tmp_path: Path, samples: int, steps: int) -> None:
    with pytest.raises(ValueError, match="samples must be"):
        build_plan(inputs, tmp_path / "run", samples, steps)
    assert not (tmp_path / "run").exists()


def test_run_directory_must_be_new_and_outside_inputs_and_formal_outputs(inputs: Path, tmp_path: Path) -> None:
    for directory in (tmp_path / "official-ckpts/run", tmp_path / "official-results/run",
                      tmp_path / "data/UCIT/run", tmp_path / "models/llava-v1.5-7b/run"):
        with pytest.raises(ValueError, match="inside"):
            build_plan(inputs, directory, 1, 1)
    existing = tmp_path / "project/logs/existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        build_plan(inputs, existing, 1, 1)
    link = tmp_path / "project/logs/dangling"
    link.symlink_to(tmp_path / "missing-target")
    with pytest.raises(ValueError, match="already exists"):
        build_plan(inputs, link, 1, 1)
    (tmp_path / "project/logs/outside").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="inside"):
        build_plan(inputs, tmp_path / "project/logs/outside/new-run", 1, 1)


def test_external_adapter_parent_keeps_logs_and_loader_name_isolated(inputs: Path, tmp_path: Path,
                                                                    monkeypatch) -> None:
    weight_parent = tmp_path / "weights"
    weight_parent.mkdir()
    run_dir = tmp_path / "project/logs/smoke"
    adapter = weight_parent / "Task1_llava_lora"
    plan = build_plan(inputs, run_dir, 2, 1, adapter_dir=weight_parent)
    assert plan["adapter_path"] == str(adapter)
    assert flag(plan["training_command"]["argv"], "--output_dir") == str(adapter)
    assert flag(plan["training_command"]["argv"], "--logging_dir") == str(run_dir / "logs")
    assert flag(plan["inference_command"]["argv"], "--adapter") == str(adapter)
    assert not adapter.exists() and not run_dir.exists()
    monkeypatch.setattr("mcitlib.doctor.run_doctor", lambda *args, **kwargs: Report())

    def fake_run(command, root, log):
        assert run_dir in log.parents
        if command["kind"] == "train":
            adapter.mkdir()
            for name in ("config.json", "adapter_config.json", "adapter_model.bin", "non_lora_trainables.bin"):
                (adapter / name).write_text("mock")
            (adapter / "trainer_state.json").write_text(json.dumps({"global_step": 1}))
        else:
            (run_dir / "inference.json").write_text(json.dumps({"benchmark": "ucit", "loaded": True,
                "adapter_path": str(adapter), "sample_id": "0", "new_tokens": 1, "text": "answer"}))

    monkeypatch.setattr("mcitlib.smoke._run", fake_run)
    completed = execute_plan(plan, inputs)
    assert completed["status"] == "completed"
    assert (run_dir / "train-subset.json").is_file() and (run_dir / "smoke-result.json").is_file()
    assert {p.name for p in weight_parent.iterdir()} == {"Task1_llava_lora"}
    with pytest.raises(ValueError, match="already exists"):
        build_plan(inputs, tmp_path / "project/logs/another", 2, 1, adapter_dir=weight_parent)


@pytest.mark.parametrize("benchmark", ["ucit", "mllm_dcl"])
def test_doctor_fail_blocks_directory_and_processes(inputs: Path, tmp_path: Path, monkeypatch, benchmark: str) -> None:
    report = Report()
    report.add(FAIL, "CUDA", "not available")
    calls = []
    def doctor(*args, **kwargs):
        calls.append((args, kwargs))
        return report
    monkeypatch.setattr("mcitlib.doctor.run_doctor", doctor)
    monkeypatch.setattr("mcitlib.smoke._run", lambda *args: pytest.fail("doctor failure must stop execution"))
    run_dir = tmp_path / "project/logs/failed-doctor"
    plan = execute_plan(build_plan(inputs, run_dir, 2, 1, benchmark=benchmark), inputs)
    assert plan["status"] == "doctor_failed" and plan["doctor"]["failed"]
    assert calls[0][0][:2] == ("lora_ft", benchmark)
    assert calls[0][1] == {"skip_gpu": False, "skip_entrypoints": False, "sequence": "train"}
    assert not run_dir.exists()


@pytest.mark.parametrize("benchmark", ["ucit", "mllm_dcl"])
def test_execution_uses_first_records_and_loads_saved_adapter(inputs: Path, tmp_path: Path, monkeypatch,
                                                               benchmark: str) -> None:
    report = Report()
    report.add(OK, "all checks")
    doctor_calls = []
    def doctor(*args, **kwargs):
        doctor_calls.append((args, kwargs))
        return report
    monkeypatch.setattr("mcitlib.doctor.run_doctor", doctor)
    calls = []
    run_dir = tmp_path / "project/logs/run"
    def fake_run(command, root, log):
        calls.append(command["kind"])
        subset = json.loads((run_dir / "train-subset.json").read_text())
        assert [row["id"] for row in subset] == ["0", "1", "2"]
        adapter = run_dir / "Task1_llava_lora"
        if command["kind"] == "train":
            adapter.mkdir()
            for name in ("config.json", "adapter_config.json", "adapter_model.bin", "non_lora_trainables.bin"):
                (adapter / name).write_text("mock")
            (adapter / "trainer_state.json").write_text(json.dumps({"global_step": 2}))
        else:
            assert flag(command["argv"], "--adapter") == str(adapter)
            (run_dir / "inference.json").write_text(json.dumps({"benchmark": benchmark, "loaded": True, "adapter_path": str(adapter),
                                                                "sample_id": subset[0]["id"],
                                                                "new_tokens": 8, "text": "mock result"}))
    monkeypatch.setattr("mcitlib.smoke._run", fake_run)
    plan = execute_plan(build_plan(inputs, run_dir, 3, 2, benchmark=benchmark), inputs)
    assert doctor_calls[0][0][:2] == ("lora_ft", benchmark)
    assert plan["benchmark"] == benchmark
    assert calls == ["train", "smoke-inference"]
    assert plan["status"] == "completed" and plan["actual_global_steps"] == 2
    assert plan["inference"]["sample_id"] == "0" and plan["inference"]["new_tokens"] == 8
    assert json.loads((run_dir / "smoke-result.json").read_text())["status"] == "completed"
    assert not (tmp_path / "official-ckpts").exists() and not (tmp_path / "official-results").exists()


def test_too_many_steps_prevents_inference(inputs: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("mcitlib.doctor.run_doctor", lambda *a, **k: Report())
    run_dir = tmp_path / "project/logs/run"
    def fake_run(command, root, log):
        assert command["kind"] == "train"
        adapter = run_dir / "Task1_llava_lora"
        adapter.mkdir()
        (adapter / "trainer_state.json").write_text(json.dumps({"global_step": 4}))
    monkeypatch.setattr("mcitlib.smoke._run", fake_run)
    result = execute_plan(build_plan(inputs, run_dir, 1, 1), inputs)
    assert result["status"] == "failed" and result["actual_global_steps"] == 4
    assert not (run_dir / "inference.json").exists()


def test_streaming_subset_never_skips_records_or_reads_more(tmp_path: Path) -> None:
    data = tmp_path / "large.json"
    rows = [{"text": "a,b" * 40000}, {"id": 2}, {"id": 3}]
    data.write_text(json.dumps(rows))
    assert _first_records(data, 2) == rows[:2]
    data.write_text('[{"id": 1}, 7, {"id": 3}]')
    with pytest.raises(ValueError, match="not an object"):
        _first_records(data, 2)
    data.write_text('[{"id": 1}]')
    with pytest.raises(ValueError, match="fewer"):
        _first_records(data, 2)


def test_generation_is_one_call_with_hard_token_cap() -> None:
    class Tensor:
        def unsqueeze(self, value):
            assert value == 0
            return self
        def half(self):
            return self
        def cuda(self):
            return self
    class Model:
        def __init__(self):
            self.calls = []
        def generate(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "mock tokens"
    class Torch:
        inference_mode = staticmethod(nullcontext)
    model = Model()
    assert generate_one(Torch, model, "input", Tensor()) == "mock tokens"
    assert len(model.calls) == 1 and model.calls[0][1]["max_new_tokens"] == 8


@pytest.mark.parametrize("benchmark", ["ucit", "mllm_dcl"])
def test_inference_main_reloads_with_shared_name_and_only_first_sample(tmp_path: Path, monkeypatch,
                                                                        benchmark: str) -> None:
    """Run the actual inference entry with CPU fakes, retaining the real shared name helper."""
    import sys
    from types import ModuleType, SimpleNamespace

    from PIL import Image

    from mcitlib.backbones.llava import smoke_infer

    adapter = tmp_path / "Task1_llava_lora"
    adapter.mkdir()
    subset = tmp_path / "subset.json"
    subset.write_text(json.dumps([
        {"id": "first", "image": "one.png", "conversations": [{"from": "human", "value": "<image>\nhello"}]},
        {"id": "second", "image": "missing.png", "conversations": [{"from": "human", "value": "ignored"}]},
    ]))
    Image.new("RGB", (2, 2)).save(tmp_path / "one.png")
    calls = []

    class Tensor:
        def __init__(self, length=3):
            self.shape = (1, length)
        def unsqueeze(self, dim):
            assert dim == 0
            return self
        def cuda(self):
            return self
        def half(self):
            return self
        def __getitem__(self, key):
            return Tensor(self.shape[1] - key[1].start)
    class Tokenizer:
        def batch_decode(self, ids, skip_special_tokens):
            assert ids.shape == (1, 2) and skip_special_tokens
            return ["answer"]
    class Model:
        config = SimpleNamespace(mm_use_im_start_end=False)
        def generate(self, *args, **kwargs):
            calls.append(("generate", kwargs))
            return Tensor(5)
    class Processor:
        def preprocess(self, image, return_tensors):
            assert Path(image.filename).name == "one.png" and return_tensors == "pt"
            return {"pixel_values": [Tensor()]}
    class Conversation:
        roles = ("human", "assistant")
        sep_style, sep, sep2 = "single", "</s>", "</s>"
        def copy(self):
            return self
        def append_message(self, role, value):
            calls.append(("prompt", role, value))
        def get_prompt(self):
            return "the prompt"

    def module(name, **attributes):
        result = ModuleType(name)
        for key, value in attributes.items():
            setattr(result, key, value)
        monkeypatch.setitem(sys.modules, name, result)
        return result
    module("torch", inference_mode=nullcontext)
    module("llava", __path__=[])
    module("llava.constants", DEFAULT_IMAGE_TOKEN="<image>", DEFAULT_IM_START_TOKEN="<start>",
           DEFAULT_IM_END_TOKEN="<end>", IMAGE_TOKEN_INDEX=-200)
    module("llava.conversation", SeparatorStyle=SimpleNamespace(TWO="two"), conv_templates={"vicuna_v1": Conversation()})
    module("llava.model", __path__=[])
    def load_model(path, base, name):
        calls.append(("load", path, base, name))
        return Tokenizer(), Model(), Processor(), 2048
    module("llava.model.builder", load_pretrained_model=load_model)
    def process_images(images, processor, config):
        assert len(images) == 1 and images[0].mode == "RGB"
        calls.append(("process_images", processor, config))
        return [Tensor()]
    def stopping_criteria(keywords, tokenizer, input_ids):
        assert keywords == ["</s>"]
        return "stop-rule"
    module("llava.mm_utils", tokenizer_image_token=lambda *a, **k: Tensor(),
           process_images=process_images, KeywordsStoppingCriteria=stopping_criteria)
    module("llava.utils", disable_torch_init=lambda: None)
    output = tmp_path / "inference.json"
    monkeypatch.setattr(sys, "argv", ["smoke_infer", "--benchmark", benchmark, "--adapter", str(adapter), "--model-base", "base-model",
                                     "--subset", str(subset), "--image-folder", str(tmp_path), "--output", str(output)])
    smoke_infer.main()
    result = json.loads(output.read_text())
    assert ("load", str(adapter), "base-model", "Task1_llava_lora") in calls
    assert len([call for call in calls if call[0] == "generate"]) == 1
    options = next(call[1] for call in calls if call[0] == "generate")
    assert options["max_new_tokens"] == 8
    assert bool([call for call in calls if call[0] == "process_images"]) == (benchmark == "mllm_dcl")
    assert options.get("stopping_criteria") == (["stop-rule"] if benchmark == "mllm_dcl" else None)
    assert result["benchmark"] == benchmark
    assert result["loaded"] and result["adapter_path"] == str(adapter)
    assert result["sample_id"] == "first" and result["image"] == "one.png"
    assert result["text"] == "answer" and result["new_tokens"] == 2


@pytest.mark.parametrize("option", ["--skip-gpu", "--skip-entrypoints"])
def test_doctor_bypass_flags_are_not_available(inputs: Path, tmp_path: Path, monkeypatch, option: str) -> None:
    monkeypatch.setattr("mcitlib.smoke._run", lambda *args: pytest.fail("unexpected process"))
    with pytest.raises(SystemExit):
        main(["smoke", "--paths", str(inputs), "--run-dir", str(tmp_path / "run"), "--execute", option])
    assert not (tmp_path / "run").exists()


def test_dcl_mapping_uses_dcl_scope_and_other_inputs_stay_protected(inputs: Path, tmp_path: Path) -> None:
    values = yaml.safe_load(inputs.read_text())
    values["path_mappings"] = {"mllm_dcl": [{"from": "${data_root}/MLLM-CL", "to": str(tmp_path / "domain-data")}]}
    inputs.write_text(yaml.safe_dump(values))
    dcl = build_plan(inputs, tmp_path / "project/logs/dcl-run", 1, 1, benchmark="mllm_dcl")
    assert dcl["source_data"] == str(tmp_path / "domain-data/RS/train.json")
    assert dcl["image_folder"] == str(tmp_path / "domain-data/RS")
    ucit = build_plan(inputs, tmp_path / "project/logs/ucit-run", 1, 1)
    assert ucit["benchmark"] == "ucit" and ucit["source_data"] == str(tmp_path / "data/UCIT/ImageNet-R/train.json")
    with pytest.raises(ValueError, match="inside"):
        build_plan(inputs, tmp_path / "domain-data/run", 1, 1)


def test_acl_smoke_is_rejected(inputs: Path, tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="smoke benchmark"):
        build_plan(inputs, tmp_path / "run", 1, 1, benchmark="mllm_acl")
    monkeypatch.setattr("mcitlib.doctor.run_doctor", lambda *a, **k: pytest.fail("unsupported benchmark"))
    plan = build_plan(inputs, tmp_path / "project/logs/run", 1, 1)
    plan["benchmark"] = "mllm_acl"
    with pytest.raises(ValueError, match="smoke benchmark"):
        execute_plan(plan, inputs)
    with pytest.raises(SystemExit):
        main(["smoke", "--paths", str(inputs), "--run-dir", str(tmp_path / "run"), "--benchmark", "mllm_acl"])
    assert not (tmp_path / "run").exists()


def test_training_and_inference_use_isolated_cache_without_mutating_parent(inputs: Path, tmp_path: Path,
                                                                            monkeypatch) -> None:
    monkeypatch.setenv("TRITON_CACHE_DIR", "/parent/triton")
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", "/parent/torch")
    run_dir = tmp_path / "project/logs/run"
    plan = build_plan(inputs, run_dir, 1, 1)
    expected = {"TRITON_CACHE_DIR": str(run_dir / "cache/triton"),
                "TORCH_EXTENSIONS_DIR": str(run_dir / "cache/torch_extensions")}
    for command in (plan["training_command"], plan["inference_command"]):
        assert all(command["env"][key] == value for key, value in expected.items())
    assert not run_dir.exists()
    environments = []
    def fake_subprocess(argv, *, cwd, env, stdout, stderr, check):
        environments.append(env)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr("mcitlib.smoke.subprocess.run", fake_subprocess)
    run_dir.mkdir()
    for index, command in enumerate((plan["training_command"], plan["inference_command"])):
        _run(command, tmp_path / "project", run_dir / f"{index}.log")
    assert len(environments) == 2
    assert all(all(env[key] == value for key, value in expected.items()) for env in environments)
    assert os.environ["TRITON_CACHE_DIR"] == "/parent/triton"
    assert os.environ["TORCH_EXTENSIONS_DIR"] == "/parent/torch"
    assert not (run_dir / "cache").exists()
