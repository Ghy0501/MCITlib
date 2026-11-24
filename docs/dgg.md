# Dynamic Gradient Guidance for LLaVA

DGG approximates the missing gradient of earlier tasks with the displacement
from the preceding task's parameters. At each backward pass it draws once from
a Bernoulli distribution. On a successful draw it adds each parameter's
anchor displacement to its gradient, limiting the displacement norm to the
current gradient norm. The current task begins with a fresh LoRA; the preceding
checkpoint supplies the guidance anchor.

The implementation follows [DGG's paper](https://arxiv.org/abs/2511.15164)
and [official source](https://github.com/lisongze/DGG), commit
`f0c7d20552c00e33b05f15ff638894c0c34440bb` (Apache-2.0). Its pinned
Transformers and DeepSpeed submodules are respectively
`b73b600919dde4a48533f5b375c14c13fb91ce12` and
`11d998afb3922b571b72413f12d7d247eb1ecf24`.

DGG supports the LLaVA backbone for UCIT and MLLM-DCL. For each preceding task,
the replay stage samples up to 2,000 original question-and-answer rows with an
independently initialized Python seed of 1993. It combines the current task
with those samples and records source and output hashes. Distinct image roots
are resolved before combining data. Training verifies the replay manifest and
loads the previous task's adapter and non-LoRA parameters as an anchor.

The released UCIT configuration uses rank 48, alpha 96, bfloat16, one GPU,
learning rate 2e-4, batch size 48, accumulation 1, and one epoch. Its Bernoulli
probabilities by task are 0.1, 0.1, 0.1, 0.05, 0.05, and 0.1. The MLLM-DCL
configuration is an adaptation: it uses the same replay budget, probability
0.1 for each task, and the V4 DCL full-data compensation schedule of
1, 3, 1, 2, and 1 epochs. This is not a published DGG result. Guidance runs in
a method-scoped callback after the DeepSpeed optimizer backward and before the
engine's reduction epilogue. The method requires the pinned single-GPU bf16
ZeRO-2 path; the four-GPU execution profile is not supported.

The optional `--dgg-sample-profile` is separate from the full-data budget. It
verifies the recorded 3,000-example sample hashes, uses learning rate 2e-4,
microbatch 8, accumulation 8, and one epoch, and keeps the 2,000-per-old-task
replay policy. It checks input hashes again when preparing replay. Use a paths
file mapping each benchmark to the reviewed sample inputs; see
[configuration](configuration.md) for path resolution.

Train and evaluate tasks in order. For example, with `PATHS` set to that paths
file and `GPU` set to the selected physical GPU ID:

```bash
python -m mcitlib.cli doctor --method dgg --benchmark ucit --paths "$PATHS" --gpus "$GPU" --dgg-sample-profile
python -m mcitlib.cli train --method dgg --benchmark ucit --task 1 --paths "$PATHS" --gpus "$GPU" --dgg-sample-profile
python -m mcitlib.cli eval --method dgg --benchmark ucit --after-task 1 --paths "$PATHS" --gpus "$GPU"
python -m mcitlib.cli train --method dgg --benchmark ucit --task 2 --paths "$PATHS" --gpus "$GPU" --dgg-sample-profile
python -m mcitlib.cli eval --method dgg --benchmark ucit --after-task 2 --paths "$PATHS" --gpus "$GPU"
```

Use `mllm_dcl` as the benchmark for the corresponding domain sequence. On one
GPU, the evaluation chunks run serially. The UCIT generator follows the DGG
release's RGB and image-processing path and uses a 128-token maximum; it
handles the prompt-prefix convention of the active LLaVA generation API.

A task output contains the adapter and non-LoRA weights, `dgg_reference.json`,
and `dgg_guidance.json`. The reference manifest binds input bytes, guidance
probability, and the preceding checkpoint. The guidance file records backward
calls, Bernoulli successes, and gradient tensors seen and changed. Task 1 has
no prior-task guidance. Trainer checkpoints are retained for continuation:
pass `--dgg-resume-checkpoint` with a checkpoint directory under that task's
output to resume with the same data and reference. The native checkpoint
includes the optimizer, master weights, and RNG state. Each new DGG checkpoint
also contains `dgg_scheduler_v1.json`, which records the Trainer's separate
scheduler and its actual optimizer-group identity. Resume checks this sidecar
and fails clearly if it is missing or incompatible; older checkpoints without
it cannot be resumed as complete training state.
