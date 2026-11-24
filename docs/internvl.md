# InternVL implementation

The `internvl` backbone supports the LLaVA-formatted `InternVL-Chat-ViT-6B-Vicuna-7B` checkpoint on UCIT and MLLM-DCL. Select it explicitly in the unified CLI. The compatibility matrix describes implemented entry points; GPU and sample completion are recorded separately in experiment reports.

## Model and environment

Place these directories under the configured `model_root`:

```text
InternVL-Chat-ViT-6B-Vicuna-7B/
InternViT-6B-224px/
clip-vit-large-patch14-336/
```

The decoder has 32 layers and hidden size 4096. The InternViT tower has 48 layers and hidden size 3200. It uses 336-pixel inputs, interpolates the pretrained positional grid, and selects hidden-state layer `-4`; patch features pass through the 3200-to-4096 projector. DISCO, HiDe, and ModalPrompt use a separate CLIP tower for 768-dimensional guidance. Model paths are configured in memory without rewriting the supplied weight configuration.

Use the method-specific requirements listed in the [README](../README.md), plus [InternVL vision dependencies](../envs/internvl-extra.txt). The backbone-aware doctor verifies the additional dependency pin, local model geometry, generated inputs and dependencies, and actual training/evaluation imports. Vendored LLaVA and PEFT remain the shared runtime; the InternViT factory is installed by the project package.

## Methods

| Method | Training and state contract |
| --- | --- |
| LoRA-FT | Continue the preceding task's adapter and additional trainable state. |
| Replay | Train on current-task QA plus the configured memory of original previous-task QA. |
| MoELoRA | Preserve configured experts, task state, and adapter save/reload behavior. |
| MR-LoRA | Train independent task experts and a separate router; construct expert predictions before joining router-selected outputs. |
| OLoRA | Preserve frozen trained experts, orthogonal loss, and verified non-reentrant checkpointing. |
| CL-MoE | Resolve the benchmark's task names to their task indices and preserve method-specific routing state. |
| DISCO | Use separate CLIP guidance, checkpointed feature/prototype state, and request-scoped inference caches. |
| HiDe | Preserve hierarchical routing, separate A/B expert sums and their cross terms, and request-scoped inference caches. |
| SMoLoRA | Load the benchmark's instruction embeddings, preserve task gating and adapter state, and use InternViT patch features. |
| ModalPrompt | Train prompts with the separate CLIP text/vision guidance; restore the preceding task's prompt and trainable projection state. |
| SEFE | Train ordinary LoRA on the seven decoder projections, accumulate the top 2% element indices selected from each `B @ A` update, apply the `2.5e3` regularization weight, and merge each task for continuation. |
| KeepLoRA | Produce the configured gradient/SVD initialization before training and preserve its merge and subspace dependencies. |

MR-LoRA's fixed expert pool is a retrospective evaluation protocol. Its routed matrices use `UCIT_router` and `DCL_router`; raw expert predictions and routing diagnostics remain separate. Router letter labels are never used as Replay answers.

The [shared method contract](llava_implementation.md) also applies to InternVL. Inference caches must clear at the end of each request, stay out of checkpoint state, and invalidate when adapter state, mode, parameters, device or dtype change. Input-dependent routing remains dynamic.

## Configuration and execution

```bash
mcitlib config expand --backbone internvl --method lora_ft --benchmark ucit --task 1
mcitlib doctor --backbone internvl --method lora_ft --benchmark ucit \
  --paths configs/paths.yaml --validation-fixes
mcitlib train --backbone internvl --method lora_ft --benchmark ucit \
  --paths configs/paths.yaml --validation-fixes --dry-run
```

Full-data configurations retain each InternVL method's rank, expert settings, learning rate, batch and task epochs. The explicit four-GPU **sample** profile uses global batch 64, learning rate `2e-4`, and one main epoch for all methods, including ModalPrompt, with a separately preserved MR router budget. Use it with the sample paths; it is not a full-data training profile.

The [sample queue](internvl_sample_validation.md) checks data hashes, obtains exclusive GPU 0–3 leases, runs complete doctors, and isolates each method's failures. Checkpoints, logs, results and matrices are scoped to the backbone and run. The source and input binding protects a running queue against subsequent edits.

## Hardware and validation boundaries

KeepLoRA gradient extraction keeps the dense model, accumulated gradients and current gradients on each preparation device. The doctor computes a lower memory bound from the actual configuration and target modules; the four-GPU sample plan does not shard those tensors across devices. A failed memory check blocks preparation rather than changing the algorithm.

SEFE key indices and merged models require additional disk reserves. The queue protects its free-space floor and removes an intermediate merged model only after its verified final consumer, while retaining adapters, logs, results and the final merged model.

CPU integration checks cover real tiny InternViT/decoder forwards, finite gradients, adapter and prompt lifecycle, private merge/gradient builders, model dispatch, routing matrices and method state. A doctor using `--skip-gpu` establishes CPU prerequisites only. Actual GPU checks and complete sample matrices must be established by their own run records.
