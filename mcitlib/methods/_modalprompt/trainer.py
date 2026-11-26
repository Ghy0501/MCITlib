# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""ModalPrompt loss, optimizer grouping, and embedding-master restoration."""
from __future__ import annotations

import torch

from mcitlib.backbones.llava.trainer import (
    ALL_LAYERNORM_LAYERS,
    LLaVATrainer,
    ShardedDDPOption,
    get_parameter_names,
    is_sagemaker_mp_enabled,
    logger,
)
from mcitlib.methods._modalprompt import observe


def capture_origin(model):
    """Call v3's snapshot method after loading the previous checkpoint."""
    weight = model.get_input_embeddings().weight
    if weight.ndim != 2:
        if not hasattr(weight, "ds_id"):
            raise RuntimeError("ModalPrompt embedding is neither full nor a ZeRO parameter")
        import deepspeed

        # Compatibility case: the checkpoint load left a ZeRO placeholder.
        # The v3 detach then captures the gathered storage; no copy or dtype
        # conversion is applied. Exiting the context re-partitions weight.
        with deepspeed.zero.GatheredParameters([weight]):
            if weight.ndim != 2:
                raise RuntimeError("DeepSpeed did not gather the ModalPrompt embedding")
            model.set_origin_embed_tokens()
    else:
        model.set_origin_embed_tokens()
    return model.origin_embed_tokens_weights


class ModalPromptTrainer(LLaVATrainer):
    def __init__(self, *args, origin, allow_cpu_standin=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._modal_origin = origin
        self._allow_cpu_standin = allow_cpu_standin

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        image = self.model.image_cos_sim_loss
        text = self.model.text_cos_sim_loss
        combined = loss + image + text
        if observe.enabled():
            observe.write_event(int(self.model.cur_task), int(self.state.global_step), "loss",
                                language=float(loss.detach().float().item()),
                                image=float(image.detach().float().item()),
                                text=float(text.detach().float().item()),
                                total=float(combined.detach().float().item()))
        loss = combined
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        if ShardedDDPOption is not None and self.sharded_ddp == ShardedDDPOption.SIMPLE:
            return super().create_optimizer()
        if self.optimizer is not None:
            return self.optimizer
        if self.args.mm_projector_lr is None:
            return super().create_optimizer()
        decay = {name for name in get_parameter_names(self.model, ALL_LAYERNORM_LAYERS) if "bias" not in name}
        grouped = []
        for special in (False, True):
            for use_decay in (True, False):
                params = [p for name, p in self.model.named_parameters()
                          if p.requires_grad and (name in decay) == use_decay
                          and any(key in name for key in ("prompt_transform", "embed_tokens", "mm_projector"))
                          == special]
                group = {"params": params, "weight_decay": self.args.weight_decay if use_decay else 0.0}
                if special:
                    group["lr"] = self.args.mm_projector_lr
                grouped.append(group)
        opt_cls, opt_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = opt_cls(grouped, **opt_kwargs)
        if opt_cls.__name__ == "Adam8bit":
            import bitsandbytes

            manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
            skipped = 0
            for module in self.model.modules():
                if isinstance(module, torch.nn.Embedding):
                    skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                    logger.info(f"skipped {module}: {skipped / 2**20}M params")
                    manager.register_module_override(module, "weight", {"optim_bits": 32})
                    logger.debug(f"skipped: {skipped / 2**20}M params")
        return self.optimizer

    def training_step(self, model, inputs):
        # Match v3/Transformers 4.31 order: prepare inputs, restore the master,
        # then perform forward and backward. No low-precision sync is added.
        model.train()
        inputs = self._prepare_inputs(inputs)
        self.restore_embedding_master(model)
        if is_sagemaker_mp_enabled():
            from transformers.trainer import smp_forward_backward

            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            from apex import amp

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps

    def restore_embedding_master(self, model):
        mask = self.model.embed_tokens_mask_1d
        for name, weight in model.named_parameters():
            if "embed_tokens.weight" not in name:
                continue
            if hasattr(weight, "ds_id") or hasattr(weight, "_hp_mapping"):
                from deepspeed.utils import (
                    safe_get_full_fp32_param,
                    safe_set_full_fp32_param,
                )

                current = safe_get_full_fp32_param(weight)
                if current is None:
                    raise RuntimeError("DeepSpeed did not expose the embedding FP32 master weights")
                old = self._modal_origin.to(current.device, current.dtype)
                expanded = mask.to(current.device, current.dtype).unsqueeze(1)
                restored = old * (1 - expanded) + current * expanded
                safe_set_full_fp32_param(weight, restored)
                if observe.enabled():
                    master_min, master_max = torch.aminmax(current)
                    restore_min, restore_max = torch.aminmax(restored)
                    observe.write_event(int(self.model.cur_task), int(self.state.global_step), "master_restore",
                                        parameter=name, master_shape=list(current.shape),
                                        origin_shape=list(self._modal_origin.shape),
                                        active_rows=int(mask.count_nonzero().item()),
                                        master_range=[float(master_min.item()), float(master_max.item())],
                                        restored_range=[float(restore_min.item()), float(restore_max.item())],
                                        safe_get="deepspeed.utils.safe_get_full_fp32_param",
                                        safe_set="deepspeed.utils.safe_set_full_fp32_param",
                                        groups=observe.optimizer_groups(model, weight._z3_optimizer))
            elif self._allow_cpu_standin:
                # CPU test stand-in only, never the configured GPU training.
                with torch.no_grad():
                    old = self._modal_origin.to(weight.device, weight.dtype)
                    expanded = mask.to(weight.device, weight.dtype).unsqueeze(1)
                    weight.copy_(old * (1 - expanded) + weight * expanded)
            else:
                raise RuntimeError("ModalPrompt requires a DeepSpeed FP32 master parameter")
