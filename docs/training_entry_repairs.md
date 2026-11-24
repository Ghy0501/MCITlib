# LLaVA training behavior

The [implementation contract](llava_implementation.md) defines OLoRA checkpointing and DISCO/HiDe inference. The shared entry also enforces the following behavior:

| Scope | Contract |
| --- | --- |
| Task identity | Parse registered task labels and preserve each method's algorithm task index. |
| Expert construction | Propagate expert count, task embedding dimensions, and recorded method settings. |
| PEFT registration | Install method tuner and state-dict support before constructing or loading a model in every process. |
| Sampling | Preserve modality-length sampling and the non-CLI `is_SAT=False` default. |
| SEFE vision initialization | Supply `mm_patch_merge_type=flat` and retain the method parameters. |
| ZeRO-3 loading | Use the [partition-aware loader](../mcitlib/backbones/llava/zero3_loading.py) for partitioned parameters. |
| DISCO/HiDe state | Save frozen statistics and boundary updates; loaded checkpoint values take precedence. |

A checkpoint without required prototype or anchor state cannot be repaired by assigning fresh defaults. Validate training, saving, loading, and task continuation in separate processes.

ModalPrompt uses `envs/llava-modalprompt-ds126.txt`. Its Trainer maintains image/text cosine loss and restores inactive FP32 prompt rows at the start of each step. Inspect the effective ZeRO optimizer and scheduler rather than inferring parameter-group learning rates from unused Trainer settings.

Use `--validation-fixes` consistently for the complete ModalPrompt continuation and MR-LoRA evaluation plans. KeepLoRA gradient preparation uses two extraction shards. Its full-rank gradient preparation may exceed a 40 GiB GPU; a passing CPU test does not establish GPU capacity.

Sample runs check execution completeness. Full DCL task epochs compensate for different dataset sizes. ModalPrompt main training and MR-LoRA router training have separate budgets; record them explicitly.
