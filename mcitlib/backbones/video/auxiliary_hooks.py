# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Method-specific continuations shared by both native video trainers."""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch


class PlainAdapterHooks:
    def build(self, model, config):
        from peft import get_peft_model
        return get_peft_model(model, config)

    def loss(self, model, inputs, outputs, base_loss):
        return None

    def checkpoint_scope(self, args):
        return nullcontext(None)


class SEFEHooks(PlainAdapterHooks):
    method = "sefe"

    def __init__(self, options):
        self.previous_keys = options.get("previous_keys")
        self.modules = []

    def build(self, model, config):
        wrapped = super().build(model, config)
        if self.previous_keys is None:
            return wrapped
        serialized = torch.load(self.previous_keys, map_location="cpu")
        keys = {name.removesuffix(".weight"): record for name, record in serialized.items()}
        target_names = {name.rsplit(".lora_A.", 1)[0] for name, _ in wrapped.named_parameters()
                        if ".lora_A.default.weight" in name}
        if keys.keys() != target_names or len(keys) != 224:
            raise ValueError(f"SEFE requires key elements for each of the 224 language LoRA targets ({len(keys)} keys)")
        for name, record in sorted(keys.items()):
            module = wrapped.get_submodule(name)
            coordinates = record["top2"]
            if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) == 0:
                raise ValueError(f"Invalid SEFE coordinates for {name}")
            self.modules.append((module, coordinates))
        return wrapped

    def loss(self, model, inputs, outputs, base_loss):
        if not self.modules:
            return None
        penalty = base_loss.new_zeros(())
        for module, indices in self.modules:
            factor_a = module.lora_A["default"].weight
            factor_b = module.lora_B["default"].weight
            rows, cols = indices.to(factor_a.device).unbind(dim=1)
            penalty = penalty + (factor_b @ factor_a)[rows, cols].abs().mean()
        return penalty * (2500.0 / len(self.modules))

    def checkpoint_scope(self, args):
        from .checkpointing import video_non_reentrant_scope

        return video_non_reentrant_scope(args, "SEFE video") if self.previous_keys else nullcontext()


class KeepLoRAHooks(PlainAdapterHooks):
    method = "keeplora"

    def __init__(self, options):
        self.gradient_subspace = Path(options["gradient_subspace"])

    def build(self, model, config):
        from ...methods._keeplora.lora import freeze_lora_a_matrices, initialize_lora_weights_custom

        wrapped = super().build(model, config)
        subspace = torch.load(self.gradient_subspace, map_location="cpu")["gradients"]
        if len(subspace) != 224 or any(not value.isfinite().all() for value in subspace.values()):
            raise ValueError("KeepLoRA requires 224 finite language subspaces")
        initialize_lora_weights_custom(wrapped, subspace, config.r)
        freeze_lora_a_matrices(wrapped)
        return wrapped


class DGGHooks(PlainAdapterHooks):
    method = "dgg"

    def __init__(self, options):
        self.options = options
        self.resume_checkpoint = options.get("resume_from_checkpoint")

    @staticmethod
    def prepare_optimizer(trainer, optimizer):
        plugin = trainer.accelerator.state.deepspeed_plugin
        config = plugin.deepspeed_config
        if config["zero_optimization"].get("offload_optimizer", {}).get("device", "none") != "cpu":
            return optimizer
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        if isinstance(optimizer, DeepSpeedCPUAdam):
            return optimizer
        # Create the offloaded optimizer before HF attaches LambdaLR. This
        # prevents Accelerate 0.21 rebuilding it with a distinct group table.
        config["zero_force_ds_cpu_optimizer"] = False
        groups = [group for group in optimizer.param_groups if group["params"]]
        return DeepSpeedCPUAdam(groups, adamw_mode=True)

    @contextmanager
    def train_scope(self, trainer):
        from transformers import TrainerCallback

        from ...methods._dgg.runtime import DGGCallback, guard_reference, load_guidance
        from ...methods._dgg.scheduler_checkpoint import DGGSchedulerCheckpoint
        from .dgg_snapshot import recovery_scope

        args = trainer.args
        reference = self.options.get("reference_adapter")
        probability = self.options["guidance_probability"]
        guard_reference(
            args.output_dir, self.options["data_path"], reference, probability, self.resume_checkpoint,
            self.options.get("dataset_sha256"), self.options.get("replay_manifest"),
        )
        callback = None

        trainer.add_callback(DGGSchedulerCheckpoint(trainer, self.resume_checkpoint))
        if self.options.get("stop_after_save") is not None:
            stop_step = int(self.options["stop_after_save"])

            class PauseAfterSave(TrainerCallback):
                def on_save(self, args, state, control, **kwargs):
                    if state.global_step == stop_step:
                        control.should_training_stop = True
                    return control

            trainer.add_callback(PauseAfterSave())
        if reference is not None:
            guidance = load_guidance(trainer.model, reference, probability)
            callback = DGGCallback(trainer, guidance)
            trainer.add_callback(callback)
        try:
            with recovery_scope(self.options.get("compact_recovery_files")):
                yield
        finally:
            if callback is not None:
                callback.close()
            if args.process_index == 0:
                evidence = callback.evidence() if callback else {"stage": "initial_task_no_history"}
                evidence["resume_from_checkpoint"] = self.resume_checkpoint
                path = Path(args.output_dir) / "dgg_guidance.json"
                path.write_text(json.dumps(evidence, indent=2) + "\n")


def prepare_dgg_replay(paths, media_root, output, *, samples_per_task=2000, seed=1993):
    from ...methods._dgg.core import replay_rows
    from .protocol import hash_video_qa_inputs

    datasets, hashes = hash_video_qa_inputs(paths, media_root)
    selected = replay_rows(datasets[0], datasets[1:], samples_per_task=samples_per_task, seed=seed)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(selected, handle, ensure_ascii=False)
    evidence = {
        "version": 1, "current": len(datasets[0]), "seed": seed, "samples_per_task": samples_per_task,
        "replay_by_task": [min(samples_per_task, len(source)) for source in datasets[1:]],
        "input_sha256": hashes, "output": str(output.resolve()),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    with output.with_suffix(".manifest.json").open("x") as handle:
        json.dump(evidence, handle, indent=2)
    return evidence


class OctopusHooks(PlainAdapterHooks):
    method = "octopus"

    def __init__(self, options):
        self.options = options
        self.stage = options["stage"]
        if self.stage not in {"stage1", "stage2"}:
            raise ValueError("Octopus video training requires an explicit stage")
        self.regularizer = None
        self.reported_terms = False

    @contextmanager
    def train_scope(self, trainer):
        from .checkpointing import video_non_reentrant_scope

        if self.stage == "stage2":
            import os

            from ...methods._octopus.runtime import setup_stage_two

            os.environ.update(
                MCITLIB_OCTOPUS_HISTORY=json.dumps(self.options["history"]),
                MCITLIB_OCTOPUS_GPWC=self.options["gpwc_path"],
                MCITLIB_OCTOPUS_LAMBDA1=str(self.options["lambda_1"]),
                MCITLIB_OCTOPUS_LAMBDA2=str(self.options["lambda_2"]),
            )
            self.regularizer = setup_stage_two(
                trainer.model, trainer.args.gradient_accumulation_steps, self.options.get("gpwc_audit"),
            )
        with video_non_reentrant_scope(trainer.args, f"Octopus video {self.stage}"):
            yield

    def loss(self, model, inputs, outputs, base_loss):
        if self.regularizer is None:
            return None
        loss = self.regularizer.extra_loss(model, inputs, outputs, base_loss)
        if not self.reported_terms:
            orthogonality, magnitude = [term.item() for term in self.regularizer.last_terms]
            if orthogonality > 0 and magnitude > 0:
                print(json.dumps({"event": "octopus_regularizers", "orthogonality": orthogonality,
                                  "magnitude": magnitude}), flush=True)
                self.reported_terms = True
        return loss

    def after_checkpoint(self, output_directory):
        if self.stage == "stage2":
            from ...methods._octopus.state import write_manifest

            write_manifest(Path(output_directory), self.options["history"], self.regularizer.provenance)
