# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Native CL-VISTA ModalPrompt with real video and text guidance."""
import inspect
import json
from contextlib import nullcontext
from pathlib import Path
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, CLIPTextModel

STATE = "modal_prompt.pt"


def gather(parameter, *, modifying=False):
    if not hasattr(parameter, "ds_id"):
        return nullcontext()
    import deepspeed

    return deepspeed.zero.GatheredParameters([parameter], modifier_rank=0 if modifying else None)


def setup(model, tokenizer, backbone, task, tasks, length, clip_path, previous=None, *, training):
    model.requires_grad_(False)
    embedding = model.get_input_embeddings()
    vocab = embedding.num_embeddings
    hidden = embedding.embedding_dim
    tokens = [f"[PRE{index}_{offset}]" for index in range(1, tasks + 1) for offset in range(1, length + 1)]
    if tokenizer.add_special_tokens({"additional_special_tokens": tokens}) != len(tokens):
        raise ValueError("ModalPrompt needs an unextended base tokenizer")
    ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).reshape(tasks, length)
    if not torch.equal(ids.flatten(), torch.arange(vocab, vocab + len(tokens))):
        raise ValueError("ModalPrompt tokens are not contiguous after the base vocabulary")
    with gather(embedding.weight):
        weights = embedding.weight.detach()
        replacement = nn.Embedding(vocab + len(tokens), hidden, padding_idx=embedding.padding_idx,
                                   dtype=weights.dtype, device=weights.device)
        with torch.no_grad():
            replacement.weight[:vocab].copy_(weights)
            random_source = torch.randint(vocab, (len(tokens),), generator=torch.Generator().manual_seed(42))
            replacement.weight[vocab:].copy_(weights[random_source])
    model.set_input_embeddings(replacement)
    # Output vocabulary remains the original language vocabulary. Prompt
    # embeddings can enter the prefix but cannot become target tokens.
    model.get_model().vocab_size = vocab + len(tokens)
    model.cur_task = task
    model.prompt_transform = nn.ModuleList([
        nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 768))
        for _ in range(tasks)
    ]).to(replacement.weight)
    projector = model.get_model().mm_projector
    projector.requires_grad_(training)
    if previous:
        boundary = torch.load(Path(previous) / STATE, map_location="cpu")
        expected = task - 1 if training else task
        if (boundary["backbone"], boundary["task"], boundary["tasks"], boundary["length"], boundary["vocab"],
                boundary["text_tower"]) != (backbone, expected, tasks, length, vocab, clip_path):
            raise ValueError("ModalPrompt checkpoint identity differs from the requested video boundary")
        if boundary["prefix_rows"].shape != (expected * length, hidden) or len(boundary["transforms"]) != expected:
            raise ValueError("Missing ModalPrompt embedding rows or trained transforms")
        for transform, saved in zip(model.prompt_transform[:expected], boundary["transforms"], strict=True):
            transform.load_state_dict(saved)
        for name, parameter in projector.named_parameters():
            with gather(parameter, modifying=True), torch.no_grad():
                parameter.copy_(boundary["projector"][name])
        for name, buffer in projector.named_buffers():
            buffer.copy_(boundary["projector"][name])
        with torch.no_grad():
            replacement.weight[vocab:vocab + expected * length].copy_(boundary["prefix_rows"])
    model.prompt_transform.requires_grad_(False)
    model.prompt_transform[task - 1].requires_grad_(training)
    replacement.requires_grad_(training)
    mask = torch.zeros(replacement.num_embeddings, dtype=torch.bool)
    mask[ids[task - 1]] = True
    model.embed_tokens_mask_1d = mask
    install_route(model, tokenizer, backbone, ids[:task], clip_path)
    return vocab


def install_route(model, tokenizer, backbone, prompt_ids, clip_path):
    llava = backbone == "videollava"
    tower = model.get_video_tower() if llava else model.get_vision_tower()
    tower_core = tower.video_tower if llava else tower.vision_tower
    text_tokenizer = AutoTokenizer.from_pretrained(clip_path, local_files_only=True)
    text_tower = CLIPTextModel.from_pretrained(clip_path, local_files_only=True).to(
        device=tower.device, dtype=tower.dtype,
    ).requires_grad_(False).eval()
    # ZeroInit partitions from_pretrained parameters even for the frozen
    # auxiliary CLIP. Register the module so engine hooks gather its weights.
    model.modal_text_tower = text_tower
    prepare = model.prepare_inputs_labels_for_multimodal
    signature = inspect.signature(prepare)
    encode_name = "encode_videos" if llava else "encode_images_or_videos"
    encode = getattr(model, encode_name)

    def routed(base, *args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        values = bound.arguments
        tokens, media = values["input_ids"], values["images"]
        if media is None or tokens.shape[1] == 1:
            return prepare(*args, **kwargs)
        batch = tokens.shape[0]
        video = torch.stack(list(media)) if llava else media
        pooled = []

        def capture(_tower, _inputs, output):
            pooled.append(output.pooler_output.detach())

        with tower_core.register_forward_hook(capture):
            features = encode(video)
        if len(pooled) != 1:
            raise RuntimeError("ModalPrompt expected one native vision pass")
        vision = pooled[0]
        if not llava:
            vision = vision.reshape(batch, base.config.num_frames, -1).mean(dim=1)
        if vision.shape != (batch, 1024):
            raise ValueError(f"Unexpected ModalPrompt video feature shape {vision.shape}")
        vision = F.interpolate(vision.float().unsqueeze(0), size=768, mode="linear", align_corners=True).squeeze(0)
        clean = tokens.masked_fill(tokens.lt(0), tokenizer.pad_token_id or 0).cpu()
        decoded = tokenizer.batch_decode(clean, skip_special_tokens=True)
        questions = ["\n".join(value.split("\n")[1:]) for value in decoded]
        if llava:
            questions = [q.split(" ASSISTANT")[0] for q in questions]
        else:
            questions = [q.split("<</SYS>>")[-1].split("[/INST]")[0].strip() for q in questions]
        text_batch = text_tokenizer(questions, padding=True, max_length=77, truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_tower.eval()
            text = text_tower(**text_batch.to(vision.device)).pooler_output
        ids = prompt_ids.to(tokens.device)
        prototypes = torch.stack([
            transform(base.get_input_embeddings()(row)).mean(dim=0)
            for row, transform in zip(ids, base.prompt_transform[:base.cur_task], strict=True)
        ])
        visual_scores = F.cosine_similarity(vision[:, None], prototypes.float()[None], dim=-1)
        text_scores = F.cosine_similarity(text.float()[:, None], prototypes.float()[None], dim=-1)
        if base.training:
            selected = ids[-1:].expand(batch, -1)
            base.image_cos_sim_loss = 1 - visual_scores[:, -1].mean()
            base.text_cos_sim_loss = 1 - text_scores[:, -1].mean()
        else:
            route = (visual_scores + text_scores).argmax(dim=1)
            selected = ids[route]
            print(json.dumps({"event": "modalprompt_routed", "task": base.cur_task,
                              "selected_task": (route + 1).tolist()}), flush=True)
        values["input_ids"] = torch.cat((tokens[:, :1], selected, tokens[:, 1:]), dim=1)
        attention = values["attention_mask"]
        if attention is None:
            attention = torch.ones_like(tokens, dtype=torch.bool)
        values["attention_mask"] = torch.cat((attention[:, :1], torch.ones_like(selected).to(attention),
                                             attention[:, 1:]), dim=1)
        labels = values["labels"]
        if labels is not None:
            values["labels"] = torch.cat((labels[:, :1], torch.full_like(selected, -100), labels[:, 1:]), dim=1)
        if "position_ids" in values:
            values["position_ids"] = None
        # Use the projected tokens from the same native forward that supplied
        # routing features. The normal backbone owns multimodal token insertion.
        setattr(base, encode_name, lambda _media: features)
        try:
            return prepare(*bound.args, **bound.kwargs)
        finally:
            delattr(base, encode_name)

    model.prepare_inputs_labels_for_multimodal = MethodType(routed, model)


class ModalPromptHooks:
    method = "modalprompt"

    def __init__(self, backbone, options):
        self.backbone = backbone
        self.options = options

    def run(self, model, tokenizer, data_args, args, previous):
        from deepspeed.utils import safe_get_full_fp32_param

        from ...methods._modalprompt.trainer import ModalPromptTrainer
        from .checkpointing import video_non_reentrant_scope
        from .worker import native_train_module

        task, tasks, length, clip = (self.options[name] for name in ("task", "tasks", "prefix_length", "text_tower"))
        if (task > 1) != bool(previous):
            raise ValueError("ModalPrompt continuation requires the preceding boundary")
        vocab = setup(model, tokenizer, self.backbone, task, tasks, length, clip, previous, training=True)
        origin = model.get_input_embeddings().weight.detach().cpu().clone()
        native = native_train_module(self.backbone)
        trainer = native.LLaVATrainer if self.backbone == "videollava" else native.VideoLLaMA2Trainer

        class VideoPromptTrainer(trainer):
            restore_embedding_master = ModalPromptTrainer.restore_embedding_master
            _modal_origin = origin
            _allow_cpu_standin = False

            def training_step(self, current, inputs):
                self.restore_embedding_master(current)
                return super().training_step(current, inputs)

            def compute_loss(self, current, inputs, return_outputs=False):
                language, outputs = super().compute_loss(current, inputs, return_outputs=True)
                image, text = model.image_cos_sim_loss, model.text_cos_sim_loss
                if self.is_world_process_zero() and self.state.global_step == 0:
                    print(json.dumps({"event": "modalprompt_guidance", "image_loss": image.item(),
                                      "text_loss": text.item(), "language_loss": language.item()}), flush=True)
                result = language + image + text
                return (result, outputs) if return_outputs else result

        data = native.make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        runner = VideoPromptTrainer(model=model, tokenizer=tokenizer, args=args, **data)
        projector = model.get_model().mm_projector
        before = {}
        for name, parameter in projector.named_parameters():
            with gather(parameter):
                before[name] = parameter.detach().cpu().clone()
        with video_non_reentrant_scope(args, "ModalPrompt"):
            runner.train()
        if args.process_index == 0:
            print(json.dumps({"event": "modalprompt_optimizer", "zero_stage": runner.model_wrapped.zero_optimization_stage(),
                              "scheduler": type(runner.lr_scheduler).__name__,
                              "learning_rates": runner.lr_scheduler.get_last_lr()}), flush=True)
        runner.save_state()
        embedding = model.get_input_embeddings().weight
        active = model.embed_tokens_mask_1d
        # Training restores master rows immediately before each forward.
        # Export one more restored master boundary so the last backward
        # cannot leak its inactive-row update into the next task.
        runner.restore_embedding_master(runner.model_wrapped)
        master = safe_get_full_fp32_param(embedding)
        if master is None:
            raise RuntimeError("ModalPrompt cannot export a boundary without the ZeRO-3 FP32 master")
        after = master.detach().cpu()
        unchanged = torch.equal(after[~active], origin[~active].to(after))
        changed = not torch.equal(after[active], origin[active].to(after))
        if not unchanged or not changed:
            raise RuntimeError(f"Invalid ModalPrompt master update: active_changed={changed} inactive_equal={unchanged}")
        rows = after[vocab:vocab + task * length].to(embedding.dtype).clone()
        states = []
        for transform in model.prompt_transform[:task]:
            state = {}
            for name, parameter in transform.named_parameters():
                with gather(parameter):
                    state[name] = parameter.detach().cpu().clone()
            states.append(state)
        projector_state = {}
        for name, parameter in projector.named_parameters():
            with gather(parameter):
                projector_state[name] = parameter.detach().cpu().clone()
        updated = sum(not torch.equal(before[name].to(value), value) for name, value in projector_state.items())
        if not updated:
            raise RuntimeError("ModalPrompt did not update the native projector")
        projector_state.update({name: value.detach().cpu().clone() for name, value in projector.named_buffers()})
        if args.process_index == 0:
            output = Path(args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            with (output / STATE).open("xb") as handle:
                torch.save({"backbone": self.backbone, "task": task, "tasks": tasks, "length": length, "vocab": vocab,
                            "text_tower": clip, "prefix_rows": rows, "transforms": states,
                            "projector": projector_state}, handle)
            print(json.dumps({"event": "modalprompt_embedding_update", "active_rows": length,
                              "inactive_rows_equal": int((~active).sum()), "projector_tensors_changed": updated,
                              "task": task}), flush=True)


def load_for_inference(backbone, base, checkpoint):
    if backbone == "videollava":
        from videollava.model.builder import load_pretrained_model

        tokenizer, model, processor, _ = load_pretrained_model(base, None, "videollava", device_map={"": "cuda:0"})
    else:
        from videollama2 import model_init

        model, processor, tokenizer = model_init(base, None, device_map={"": "cuda:0"})
        model.config.num_frames = 8
    state = torch.load(Path(checkpoint) / STATE, map_location="cpu")
    setup(model, tokenizer, backbone, state["task"], state["tasks"], state["length"], state["text_tower"],
          previous=checkpoint, training=False)
    return tokenizer, model, processor
