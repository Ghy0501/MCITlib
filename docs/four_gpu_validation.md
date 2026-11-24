# Four-GPU validation execution profile

The `four_gpu_validation` profile is an explicit option for UCIT and MLLM-DCL sample validation. Without it, existing two-GPU plans and the normal GPU-count check are unchanged. The profile uses the four device IDs `0,1,2,3` and rejects other GPU selections. ACL and general evaluation are not supported by this profile; they remain available with the default configuration.

```bash
# Build a plan only; this does not execute training or inference.
python -m mcitlib.cli train --method lora_ft --benchmark ucit --task 1 \
  --execution-profile four_gpu_validation --gpus 0,1,2,3 --dry-run

# Check the same four-GPU plan and its prerequisites before executing it.
python -m mcitlib.cli doctor --method lora_ft --benchmark ucit \
  --execution-profile four_gpu_validation --gpus 0,1,2,3 --master-port 29873
```

For API users, pass `execution_profile="four_gpu_validation"`, `gpus="0,1,2,3"` and the same `master_port` to `build_sequence_commands` and `run_doctor`. If GPU IDs are omitted with this profile, `0,1,2,3` is used. On the CLI, `MCITLIB_GPUS` and `MCITLIB_MASTER_PORT` supply omitted values; an incompatible `MCITLIB_GPUS` is rejected. The profile is declared in `configs/execution/four_gpu_validation.yaml`; it does not edit the method configurations or the machine's paths configuration.

For each training command the planner preserves `world_size × per_device_train_batch_size × gradient_accumulation_steps`. It first reduces accumulation when that preserves the value exactly; otherwise it reduces per-device batch size while keeping accumulation. A plan that cannot preserve the effective batch with positive integers fails. Learning rate, epochs, maximum steps, and all other training arguments are unchanged. This preserves the intended effective batch and algorithm; a change in process count and data partition does not promise bitwise identical training results.

Regular inference runs four chunk commands, one on each GPU, producing `4_0.jsonl` through `4_3.jsonl`. The merge step consumes those four files in index order before scoring. Use `--validation-fixes` consistently for complete MR-LoRA MLLM-DCL plans: each of the 25 expert/data combinations runs inference before merging and scoring. A discovery merge accepts only a complete four-chunk set.

KeepLoRA's gradient preparation is an intentional exception to four-way auxiliary processing. It retains the original two logical extraction chunks and the original two-chunk merge on GPUs 0 and 1. This preserves the existing equal-weight gradient initialization procedure. Its training command and ordinary inference use all four GPUs. Such preparation commands carry `gradient_preparation_shards: 2` in their metadata, and profile commands carry `execution_profile: four_gpu_validation`.

The recorded KeepLoRA MLLM-DCL stage 2 plan evaluates RS but omits Med; with `--validation-fixes` the planner adds the missing Med inference, merge and original scorer. The MR-LoRA repair runs all five experts on all five domains, then routes each previously seen domain through a fixed five-expert pool and the domain's original scorer. It keeps the standalone implementation's A–F priority and random fallback (F aliases RS), records the raw selection, chosen expert and fallback for each sample, and does not fix the random seed. It joins predictions by the **evaluated domain and question ID**, fails on missing or duplicate IDs, and does not silently discard unmatched samples. This is a retrospective fixed-five-expert evaluation: earlier stages can select experts trained later, so it is not a strict online continual-learning protocol.

Raw router predictions remain under `DCL/each_dataset`; selected expert predictions, `routing_stats.json` and scores are isolated under `DCL_router/each_dataset`. The queue selects this router directory for MR-LoRA DCL metrics. For the 3k/1k sample profile it verifies all five effective test mappings against the sample manifest and file hashes, and passes the verified sample counts and hashes to both doctor and worker; each routing join rechecks them. A successfully constructed plan does not mean any job or metric has completed.

The CPU-only tests validate all 195 recorded training and router tasks, exact effective-batch preservation, inference pinning, chunk merge inputs, KeepLoRA's two-shard exception, doctor plan consistency, and discovery failure behavior. `python tools/audit_four_gpu_validation.py` builds all 24 UCIT/DCL method plans and both MR-LoRA router plans using the installed method dependencies without executing any planned command. It reports discovery dependencies separately. Building all methods may require the full model environment even though the audit does not use a GPU.

