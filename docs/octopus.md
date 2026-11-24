# Octopus on LLaVA

Octopus is available for the UCIT and MLLM-DCL task sequences with the LLaVA backbone. Use a paths configuration that points to the desired model, image and benchmark files, then run, for example:

```sh
python -m mcitlib.cli doctor --method octopus --benchmark ucit --paths configs/paths.yaml --gpus 0,1 --validation-fixes
python -m mcitlib.cli train --method octopus --benchmark ucit --paths configs/paths.yaml --gpus 0,1 --validation-fixes --dry-run
python -m mcitlib.cli train --method octopus --benchmark ucit --paths configs/paths.yaml --gpus 0,1 --validation-fixes
python -m mcitlib.cli eval --method octopus --benchmark ucit --paths configs/paths.yaml --gpus 0,1 --after-task 2 --validation-fixes
```

Use `mllm_dcl` as the benchmark for the other supported sequence. Each stage-two checkpoint is an incremental adapter with a manifest of its ordered predecessor checkpoints and their hashes. Keep the entire Stage2 sequence together: generation reconstructs each cumulative model from its base and ordered adapters. Stage1 checkpoints are also required to continue training later tasks. The manifest rejects missing, changed or out-of-order predecessors. Octopus does not support MLLM-ACL or InternVL in this implementation.

For task 1, Stage1 is copied to Stage2. For each later task, Stage1 begins from the previous Stage1 adapter. For GPWC, the implementation reconstructs each previous cumulative Stage2 model in order and measures its full-model gradients on current-task examples. It computes the released per-logical-batch running means and sums historical matrices using a stacked reduction in their original dtype. Stage2 begins from the current Stage1 adapter and adds frozen forward branches for all earlier Stage2 adapters. The projection through the current A matrix is detached, so the orthogonality term regularizes B; the regularizer is placed after task-loss normalization for gradient accumulation.

Stage2 retains a temporary GPWC sum for the LoRA target matrices. Per-history full-model mean tensor hashes, sample settings and an aggregate hash are recorded in an audit, and successful evaluation and checkpoint reload precede cleanup of the temporary sum. Failed or interrupted work retains it. Historical data is not needed. Sampling repeats a small dataset to meet the configured sample count.

By default, Stage2 sample data, the GPWC sum and their audit files live next to
the Stage2 checkpoint. To place these temporary files under a separate project
logs directory, set `octopus_aux_root` in the paths configuration; it must be
inside `logs_root`. For example, `octopus_aux_root: ${logs_root}/octopus-aux`
keeps the Stage1 and Stage2 checkpoints in `ckpt_root`. This option does not
change the training, sampling or GPWC budgets.

The full UCIT configuration follows the released Stage1 settings: BF16, LoRA rank 48, alpha 96, learning rate `1e-4`, global batch 64, one epoch, maximum length 4096 and warmup ratio 0.05. The GPWC schedule uses 256 examples with logical batches of 16. Stage2 uses one epoch at `1e-4`, with 9,600 samples for tasks 2–4, 16,000 for task 5 and 3,200 for task 6; its orthogonality and magnitude coefficients are `0.02` and `0.01`, except that task 6 uses magnitude `0.05`. The MLLM-DCL sequence is an adaptation: Stage1 uses the benchmark's full-data epoch schedule of 1/3/1/2/1; tasks 2–5 each use 9,600 Stage2 samples, one Stage2 epoch and coefficients `0.02` and `0.01`. The validation-sample queue separately changes Stage1 to `2e-4`, global batch 64 and one epoch, while retaining these explicit GPWC and Stage2 schedules.

The released Swift four-rank DDP recipe enables activation checkpointing with non-reentrant recomputation; its separate DeepSpeed path selects reentrant recomputation. V4 follows the DDP recipe's non-reentrant choice in both Octopus stages when checkpointing is enabled, including when V4 uses DeepSpeed. The loss, GPWC calculation and stage budgets are unchanged.

The port uses the V4 LLaVA data pipeline and runtime in place of the release's Hugging Face LLaVA, Swift and vLLM stack. The released GPWC script leaves Swift's attention implementation unset, selecting between SDPA and eager through the underlying runtime. This port uses the existing V4 FlashAttention patch for GPWC as well as training to reduce attention memory. GPWC uses BF16 and can split each logical batch into token-weighted microbatches; changing the attention kernel and finite-precision accumulation order may change rounding relative to the release. Evaluation loads an FP32 base, merges each adapter sequentially in chronological order, then converts to FP16 for V4 generation. This chooses a reproducible order where the release discovers evaluation directories through an unsorted `os.listdir` and avoids storing full merged copies.

The implementation corresponds to [Octopus: History-Free Gradient Orthogonalization for Continual Learning in Multimodal Large Language Models](https://arxiv.org/abs/2605.14938) and the [official code](https://github.com/Fxmangd/Octopus) at commit `ea6c2a3ed3cf72ce570d9286405f047a45e19a92`. The release's README and setup metadata declare Apache-2.0; the checked revision does not contain a standalone root license file.
