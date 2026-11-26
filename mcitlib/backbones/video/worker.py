# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Native video worker launched with the selected backbone's Python and PEFT."""
import argparse
import importlib
import json
import os
import sys
from pathlib import Path


def report(values):
    print(json.dumps(values, default=str), flush=True)


def media_file(root, stem, extensions):
    base = Path(root) / stem
    return next(p for ext in extensions if (p := Path(str(base) + ext)).is_file())


def native_train_module(backbone):
    name = "videollava.train.train" if backbone == "videollava" else "videollama2.train"
    return importlib.import_module(name)


def doctor(args):
    import deepspeed
    import peft
    import torch
    import transformers
    from decord import VideoReader, cpu
    from transformers import AutoTokenizer

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    module = native_train_module(args.backbone)
    native = Path(os.environ["PYTHONPATH"]).resolve()
    if native not in Path(module.__file__).resolve().parents:
        raise RuntimeError(f"Native training package escaped its runtime: {module.__file__}")
    if Path(sys.prefix) not in Path(peft.__file__).resolve().parents:
        raise RuntimeError(f"Native PEFT escaped the selected environment: {peft.__file__}")
    config = json.loads((Path(args.base) / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=False, local_files_only=True)
    raw = VideoReader(args.video, ctx=cpu(0), num_threads=1)
    indices = [i * (len(raw) - 1) // 7 for i in range(8)]
    if args.backbone == "videollava":
        from videollava.model.multimodal_encoder.languagebind import (
            LanguageBindVideoConfig,
            LanguageBindVideoProcessor,
        )
        importlib.import_module("videollava.train.llama_flash_attn_monkey_patch")

        tower = LanguageBindVideoConfig.from_pretrained(config["mm_video_tower"], local_files_only=True)
        processor = LanguageBindVideoProcessor(tower)
        pixels = processor(args.video, return_tensors="pt")["pixel_values"][0]
        if pixels.ndim != 4 or tuple(pixels.shape[:2]) != (3, 8):
            raise ValueError(f"Expected LanguageBind [C,T,H,W], got {pixels.shape}")
    else:
        from transformers import CLIPImageProcessor
        from videollama2.mm_utils import frame_sample, process_video

        indices = frame_sample(len(raw), mode="uniform", num_frames=8).tolist()
        processor = CLIPImageProcessor.from_pretrained(config["mm_vision_tower"], local_files_only=True)
        pixels = process_video(args.video, processor, aspect_ratio="pad", num_frames=8)
        if pixels.ndim != 4 or tuple(pixels.shape[:2]) != (8, 3):
            raise ValueError(f"Expected VideoLLaMA2 [T,C,H,W], got {pixels.shape}")
    if len(set(indices)) < 2 or tensor_frame_difference(pixels, args.backbone) == 0:
        raise ValueError("The video probe lacks temporal variation")
    if args.backbone == "videollava":
        from deepspeed.ops.op_builder import CPUAdamBuilder

        optimizer = CPUAdamBuilder().load()
        report({"event": "cpu_adam_ready", "extension": optimizer.__file__,
                "cc": os.environ.get("CC"), "cxx": os.environ.get("CXX")})
    report({
        "event": "doctor", "argv": sys.argv, "python": sys.executable,
        "native_train": module.__file__, "pythonpath": os.environ["PYTHONPATH"],
        "peft": peft.__file__, "peft_version": peft.__version__, "transformers": transformers.__version__,
        "deepspeed": deepspeed.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
        "physical_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"), "logical_device": torch.cuda.current_device(),
        "gpu": torch.cuda.get_device_name(), "rank": os.environ.get("LOCAL_RANK"),
        "tokens": len(tokenizer), "native_sampled_indices": indices, "input_tensor": list(pixels.shape),
        "processor": type(processor).__name__, "projector": config["mm_projector_type"],
    })


def tensor_frame_difference(pixels, backbone):
    frames = pixels.transpose(0, 1) if backbone == "videollava" else pixels
    return float((frames - frames[0]).abs().max())


def install_training_hooks(module, backbone, method_hooks=None, observe=False):
    # The native trainer still owns its optimizer, loss forward and lifecycle.
    from contextlib import nullcontext

    import torch
    from transformers import TrainerCallback

    def is_adapter(name):
        return ".lora_" in name

    def is_projector(name):
        return ".mm_projector." in name

    def freeze_sample(param):
        return param.detach().reshape(-1)[::max(1, param.numel() // 64)].cpu().clone()

    def train_snapshot(model):
        return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}

    class Observe(TrainerCallback):
        def on_train_begin(self, args, state, control, model, **kwargs):
            self.before = train_snapshot(model)
            target_names = [n for n in self.before if is_adapter(n)]
            self.frozen, self.method_state = {}, {}
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    continue
                if getattr(method_hooks, "method", None) == "same" and ".lora_state." in name:
                    self.method_state[name] = freeze_sample(parameter)
                else:
                    self.frozen[name] = freeze_sample(parameter)
            if not target_names or not any(is_projector(n) for n in self.before) or any(
                "tower" in n or not (is_adapter(n) or is_projector(n)) for n in self.before
            ):
                raise RuntimeError(f"Unexpected trainable targets: {list(self.before)}")
            target_layers = {n.rsplit(".lora_", 1)[0] for n in target_names}
            expected_targets = 32 * (3 if getattr(method_hooks, "method", None) == "same" else 7)
            if len(target_layers) != expected_targets:
                raise RuntimeError(f"LoRA targets do not match the expected {expected_targets} language linears")
            media_shapes = []
            core = model.get_model()
            tower = core.get_video_tower() if backbone == "videollava" else core.get_vision_tower()

            def observe_media(_tower, inputs, output):
                if media_shapes:
                    return
                media_shapes.append(tuple(inputs[0].shape))
                report({"event": "vision_forward", "shape": media_shapes[0],
                        "features": tuple(output.shape), "dtype": str(inputs[0].dtype),
                        "feature_dtype": str(output.dtype), "tower_class": type(_tower).__name__})

            self.hook = tower.register_forward_hook(observe_media)
            self.media_shapes = media_shapes
            report({
                "event": "training_start", "previous": _option("--previous_task_model_path"),
                "dtype": str(next(model.parameters()).dtype), "trainable": len(self.before),
                "lora_targets": len(target_layers), "sampled_frozen_tensors": len(self.frozen),
                "projector_class": type(core.mm_projector).__name__, "gpu": torch.cuda.get_device_name(),
                "logical_device": torch.cuda.current_device(), "world_size": args.world_size, "argv": sys.argv,
            })

        def on_train_end(self, args, state, control, model, **kwargs):
            self.hook.remove()
            after = train_snapshot(model)
            changed = [n for n, previous in self.before.items() if not torch.equal(previous, after[n])]
            drift = [n for n, p in model.named_parameters() if n in self.frozen and (
                p.requires_grad or not torch.equal(self.frozen[n], freeze_sample(p))
            )]
            if drift:
                raise RuntimeError(f"Sampled frozen parameters changed: {drift[:8]}")
            changed_state = sum(not torch.equal(self.method_state[n], freeze_sample(p))
                                for n, p in model.named_parameters() if n in self.method_state)
            if self.method_state and not changed_state:
                raise RuntimeError("SAME did not update its non-gradient spectral or routing state")
            loss = [entry["loss"] for entry in state.log_history if "loss" in entry]
            if not self.media_shapes or not loss or not all(0 < value < float("inf") for value in loss):
                raise RuntimeError(f"No valid real video update: loss={loss} media={self.media_shapes}")
            if not (any(is_adapter(n) for n in changed) and any(is_projector(n) for n in changed)):
                raise RuntimeError(f"Missing LoRA/projector update: {len(changed)} changed tensors")
            router_updates = sum(".lora_router." in n for n in changed)
            if method_hooks is not None and method_hooks.method in {"moelora", "clmoe"} and not router_updates:
                raise RuntimeError("The trainable expert router did not update")
            report({"event": "parameters_updated", "router_tensors_changed": router_updates, "step": state.global_step, "loss": loss,
                    "lora_tensors_changed": sum(map(is_adapter, changed)),
                    "projector_tensors_changed": sum(map(is_projector, changed)),
                    "method_state_tensors_changed": changed_state,
                    "sampled_frozen_tensors_equal": len(self.frozen),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()})

    trainer = "LLaVATrainer" if backbone == "videollava" else "VideoLLaMA2Trainer"
    original = getattr(module, trainer)

    class ObservedTrainer(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if observe:
                self.add_callback(Observe())

        def create_optimizer(self):
            optimizer = super().create_optimizer()
            if getattr(method_hooks, "method", None) == "dgg":
                self.optimizer = method_hooks.prepare_optimizer(self, optimizer)
                return self.optimizer
            return optimizer

        def compute_loss(self, model, inputs, return_outputs=False):
            if method_hooks is None:
                return super().compute_loss(model, inputs, return_outputs=return_outputs)
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            extra = method_hooks.loss(model, inputs, outputs, loss)
            if extra is not None:
                loss = loss + extra
                if not getattr(self, "_reported_video_extra", False) and extra.detach().abs().item() > 0:
                    report({"event": "auxiliary_loss", "method": method_hooks.method, "loss": extra.detach().item()})
                    self._reported_video_extra = True
            return (loss, outputs) if return_outputs else loss

        def train(self, *args, **kwargs):
            scope = method_hooks.checkpoint_scope(self.args) if method_hooks is not None else nullcontext(None)
            train_scope = getattr(method_hooks, "train_scope", None)
            lifecycle = train_scope(self) if train_scope is not None else nullcontext()
            if method_hooks is not None and method_hooks.method == "dgg":
                kwargs["resume_from_checkpoint"] = method_hooks.resume_checkpoint
            with scope as checkpoint, lifecycle:
                outcome = super().train(*args, **kwargs)
            if checkpoint is not None:
                report({"event": "non_reentrant_checkpoint", "calls": checkpoint.calls})
            return outcome

    setattr(module, trainer, ObservedTrainer)


def _option(flag):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None


def method_bridge(method):
    # Select the environment's PEFT before MCITlib bootstraps its image vendor paths.
    import peft

    if Path(sys.prefix) not in Path(peft.__file__).resolve().parents:
        raise RuntimeError(f"Video worker imported PEFT outside the pinned environment: {peft.__file__}")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from mcitlib.backbones.video import expert_hooks
    return expert_hooks


def train(args, remaining):
    import torch

    sys.argv = [sys.argv[0], *remaining]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if args.backbone == "videollava":
        from videollava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
    hooks = None
    if args.method_options:
        bridge = method_bridge(args.method)
        options = json.loads(args.method_options)
        if args.method in bridge.METHODS:
            hooks = bridge.ExpertHooks(args.method, options)
        elif args.method == "same":
            from mcitlib.backbones.video.same import SameHooks

            hooks = SameHooks(args.backbone, options)
        elif args.method in {"disco", "hide"}:
            from mcitlib.backbones.video.video_priors import VideoPriorHooks

            if not bridge.install_method_adapter(args.method):
                raise RuntimeError(f"Missing route tuner for {args.method}")
            hooks = VideoPriorHooks(args.method, args.backbone, options)
        elif args.method == "pclr":
            from mcitlib.backbones.video.pclr import PCLRHooks

            hooks = PCLRHooks(args.backbone, options)
        elif args.method == "modalprompt":
            from mcitlib.backbones.video.modalprompt import ModalPromptHooks

            hooks = ModalPromptHooks(args.backbone, options)
        else:
            from mcitlib.backbones.video.auxiliary_hooks import (
                DGGHooks,
                KeepLoRAHooks,
                OctopusHooks,
                SEFEHooks,
            )

            hooks = {"dgg": DGGHooks, "keeplora": KeepLoRAHooks, "sefe": SEFEHooks, "octopus": OctopusHooks}[args.method](options)
    # Legacy native train modules bind PEFT's set_state_dict at import time.
    # Register and install their existing MCITlib state codec before that import.
    module = native_train_module(args.backbone)
    if args.method not in {"pclr", "modalprompt"} and (args.observe or hooks is not None):
        install_training_hooks(module, args.backbone, hooks, args.observe)
    if args.backbone == "videollava":
        module.train(method_hooks=hooks)
    else:
        module.train("eager", method_hooks=hooks)
    if int(os.environ.get("RANK", 0)) == 0 and hasattr(hooks, "after_checkpoint"):
        hooks.after_checkpoint(_option("--output_dir"))


def generate(args):
    from contextlib import nullcontext

    import torch

    questions = json.loads(Path(args.questions).read_text())
    answers = json.loads(Path(args.answers).read_text())
    answer_map = {(type(row["question_id"]), row["question_id"]): row["answer"] for row in answers}
    question_keys = [(type(row["question_id"]), row["question_id"]) for row in questions]
    if len(question_keys) != len(questions) or len(set(question_keys)) != len(question_keys) or set(question_keys) != answer_map.keys() or len(answers) != len(answer_map):
        raise ValueError("Question/answer IDs do not pair uniquely")
    checkpoint = Path(args.checkpoint)
    standalone = args.method in {"pclr", "modalprompt"}
    if not standalone and not args.merged and (not (checkpoint / "non_lora_trainables.bin").is_file() or not (checkpoint / "adapter_config.json").is_file()):
        raise FileNotFoundError(f"Missing LoRA or projector state in {checkpoint}")
    if args.method in {"moelora", "clmoe", "olora", "smolora", "sigma", "disco", "hide", "same"}:
        bridge = method_bridge(args.method)
        if not bridge.install_method_adapter(args.method):
            raise RuntimeError(f"Missing expert tuner for {args.method}")
    llava = args.backbone == "videollava"
    if standalone:
        import importlib

        method_bridge(args.method)
        module = importlib.import_module(f"mcitlib.backbones.video.{args.method}")
        tokenizer, routed, routed_processor = module.load_for_inference(args.backbone, args.base, args.checkpoint)
    if llava:
        if args.router:
            from videollava.eval.run_inference_video_qa_router import get_model_output
        else:
            from videollava.eval.run_inference_video_qa import get_model_output
        from videollava.model.builder import load_pretrained_model

        if standalone:
            model, processor = routed, routed_processor
        else:
            tokenizer, model, processor, _ = load_pretrained_model(
                args.checkpoint, None if args.merged else args.base,
                "videollava" if args.merged else "videollava-lora", device_map={"": "cuda:0"},
                merge_adapter=args.method != "octopus",
            )
        args.device = "cuda:0"
        extensions = (".mp4", ".avi", ".mov", ".mkv")
    else:
        from videollama2 import mm_infer, model_init
        from videollama2.utils import disable_torch_init

        if standalone:
            model, processor = routed, routed_processor
        else:
            disable_torch_init()
            model, processor, tokenizer = model_init(
                args.checkpoint, None if args.merged else args.base, device_map={"": "cuda:0"},
                merge_adapter=args.method != "octopus",
            )
        extensions = (".mp4", ".webm", ".avi", ".mov", ".mkv")
    model.eval()
    if args.method == "octopus":
        method_bridge("octopus")
        from mcitlib.methods._octopus.runtime import attach_validated_history
        from mcitlib.methods._octopus.state import load_manifest

        history = load_manifest(checkpoint)
        if history:
            attach_validated_history(model, history)
        # Native video helpers pass input_ids positionally. The injected
        # LoRA and history layers are on the underlying native model.
        model = model.get_base_model()
        report({"event": "octopus_history_reloaded", "tasks": len(history)})
    if args.method in {"disco", "hide"}:
        from mcitlib.backbones.video.video_priors import install_inference_prior

        install_inference_prior(model, tokenizer, args.backbone, args.method, checkpoint)
    if args.method == "same":
        from mcitlib.backbones.video.same import install_same_inference

        install_same_inference(model, tokenizer, args.backbone, checkpoint)
    report({"event": "reloaded_for_inference", "python": sys.executable, "checkpoint": str(checkpoint),
            "model_class": type(model).__name__, "dtype": str(model.dtype), "gpu": torch.cuda.get_device_name()})
    output = Path(args.predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.jsonl")
    with partial.open("x") as handle:
        for sample, qid in zip(questions, question_keys, strict=True):
            video = str(media_file(args.media, sample["video_name"], extensions))
            question = sample["question"]
            if hasattr(model, "bind_question"):
                context = model.bind_question(question)
            elif hasattr(model, "bind_clip_video"):
                context = model.bind_clip_video(video)
            else:
                context = nullcontext()
            with context:
                if llava:
                    pred = get_model_output(model, processor["video"], tokenizer, video, question, args)
                else:
                    clip = processor["video"](video)
                    if clip.ndim != 4 or clip.shape[0] != model.config.num_frames:
                        raise ValueError(f"Unexpected inference frame tensor: {clip.shape}")
                    report({"event": "eval_video", "tensor_shape": list(clip.shape), "path": video})
                    prompt = question
                    if args.router:
                        from videollama2.eval.inference_video_qa_router import (
                            ROUTER_PROMPT_PREFIX,
                            ROUTER_PROMPT_SUFFIX,
                        )
                        prompt = ROUTER_PROMPT_PREFIX + question + ROUTER_PROMPT_SUFFIX
                    pred = mm_infer(clip, prompt, model=model, tokenizer=tokenizer, modal="video", do_sample=False)
            handle.write(json.dumps({"id": qid[1], "question_id": qid[1], "question": question,
                                     "answer": answer_map[qid], "pred": pred}, ensure_ascii=False) + "\n")
            handle.flush()
    partial.rename(output)
    report({"event": "generation_complete", "count": len(questions), "predictions": str(output),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=("videollava", "videollama2"))
    parser.add_argument("--method", default="lora_ft")
    parser.add_argument("--method-options")
    parser.add_argument("--worker-seed", type=int, help="Seed before native model initialization or sampling")
    sub = parser.add_subparsers(dest="stage", required=True)
    preflight = sub.add_parser("doctor")
    preflight.add_argument("--base", required=True)
    preflight.add_argument("--video", required=True)
    trainer = sub.add_parser("train")
    trainer.add_argument("--observe", action="store_true")
    infer = sub.add_parser("generate")
    infer.add_argument("--router", action="store_true")
    infer.add_argument("--merged", action="store_true")
    for name in ("base", "checkpoint", "media", "questions", "answers", "predictions"):
        infer.add_argument(f"--{name}", required=True)
    replay = sub.add_parser("dgg_replay")
    replay.add_argument("--media", required=True)
    replay.add_argument("--output", required=True)
    replay.add_argument("--sources", required=True)
    args, remainder = parser.parse_known_args()
    if args.worker_seed is not None:
        from transformers import set_seed
        set_seed(args.worker_seed)
    if args.stage == "train":
        train(args, remainder)
    elif remainder and not all(part.startswith("--local_rank=") for part in remainder):
        parser.error(f"unrecognized arguments: {remainder}")
    elif args.stage == "doctor":
        doctor(args)
    elif args.stage == "dgg_replay":
        method_bridge("dgg")
        from mcitlib.backbones.video.auxiliary_hooks import prepare_dgg_replay

        report(prepare_dgg_replay(json.loads(args.sources), args.media, args.output))
    else:
        generate(args)


if __name__ == "__main__":
    main()
