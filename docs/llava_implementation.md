# LLaVA implementation contract

| Method | Default behavior | Source |
| --- | --- | --- |
| OLoRA | Non-reentrant checkpointing during training | [Training entry](../mcitlib/backbones/llava/moe_train.py), [checkpoint scope](../mcitlib/backbones/llava/olora_checkpoint.py) |
| DISCO | Direct linear operations and per-request constant expert caches in gradient-free inference | [Expert layers](../mcitlib/tuners/variants/disco.py), [builder](../mcitlib/methods/_disco/llava_overrides/builder.py) |
| HiDe | Per-request constant expert caches with input-dependent routing | [Expert layers](../mcitlib/tuners/variants/hide.py), [builder](../mcitlib/methods/_hide/llava_overrides/builder.py) |

## OLoRA

The OLoRA `trainer.train()` scope uses `use_reentrant=False` and restores the torch checkpoint function on success or failure. It verifies actual checkpoint calls and rejects conflicting settings, unsupported APIs, replaced wrappers, and a checkpoint-enabled run with no calls.

Gradient checkpointing, the 0.05 orthogonal loss weight, frozen trained experts, optimizer settings, and training budget are preserved. Other methods and explicitly disabled checkpointing use their normal path. Run one Trainer at a time in a process.

## DISCO and HiDe

Builders automatically scope `model.generate` through the [shared cache](../mcitlib/tuners/inference_cache.py). Caches apply only in evaluation without gradients and clear after a successful or failed request. Adapter changes, state loading, device or dtype changes, parameter versions, and training or gradient mode changes invalidate them. They are excluded from `state_dict`. Explicitly clear a cache after modifying parameters through `.data`; concurrent weight changes during generation are unsupported.

DISCO preserves its `lora_AB` mask, zero flag, and SSA. HiDe ordinary layers sum A experts and B experts separately before applying the linear operations, preserving cross terms. Special layers retain dynamic, input-dependent expert weights. Only constant weights are cached.

Preserve floating point casts, accumulation order, tokenizer, generation settings, masks, and training gradients. Save boundary, feature, prototype, and anchor statistics with the model; checkpoint values take precedence over initialization.

## Verification

```bash
python -m pytest -q \
  tests/integration/test_llava_mainline_olora.py \
  tests/integration/test_llava_mainline_inference.py
```

The tests cover real tiny-model backward passes, frozen experts, exception restoration, generation entry points, cache invalidation, and state lifecycle. Bounded 7B generation checks cover two real examples per inference method with matching tokens and step logits. They do not establish throughput or scientific quality for every input.

For another backbone, verify its checkpoint API, dimensions, special layers, and state lifecycle. Determine layer locations from that model; do not copy a fixed LLaVA layer index. HiDe's separate A/B sums must retain their cross terms.
