# SAME and SiGMA for LLaVA

This document records the two LLaVA implementations and the limits of the
current evidence. UCIT and MLLM-DCL are configured; MLLM-ACL and non-LLaVA
backbones are explicitly unsupported. The MLLM-DCL choices below are V4
adaptations, not published benchmark results.

## Sources and licenses

- SAME: [paper, arXiv:2602.01990v3](https://arxiv.org/abs/2602.01990), validated
  PDF SHA-256 `24a1627fb3143c5a4802fa10b8f8d9a80ddad6b63ab7ef7189ed391f33e7b170`;
  [official PRISM repository](https://github.com/LAMDA-CL/Prism), commit
  `fce5be61279a34db3a3f6e47ad027718342800d7`. PRISM is MIT-licensed; its
  [license](SAME-UPSTREAM-LICENSE.txt) is included. The V4 SAME tuner adapts
  the behavior in `PEFT/tuners/custom/same.py`, its task-boundary state in
  `method/custom/same/integration.py`, and the image/text prototype routing in
  `method/custom/specialized_integration.py`.
- SiGMA: [paper, arXiv:2607.20511v1](https://arxiv.org/abs/2607.20511),
  validated PDF SHA-256
  `fce4dae2c3c93b6973405c973fd06d4ea7b70c550cc3d783a13cd3eb038cc35f`;
  [official repository](https://github.com/pgh2874/SiGMA-Multimodal-Continaul-Instruction-Tuning),
  commit `e288d434a1f538f3f0fc1d5806d91a0fae18dc96`. The repository has no
  declared top-level license. No SiGMA source is vendored; the V4 algorithm
  code is newly written from the paper and its release behavior. The reference
  repository is not distributed as part of this implementation.

The SiGMA README advertises UCIT; its directory also contains DCL scripts, but
this implementation does not claim those scripts establish a released DCL protocol.

## Algorithm correspondence and implementation choices

SAME is injected into LLaVA FFN `gate_proj`, `up_proj`, and `down_proj` only.
It freezes the multimodal projector, matching the PRISM training flags.
It maintains a routed bank of low-rank A and B experts. At each training
forward the current router probabilities are averaged, masked with the
adaptive expert activation mask, and used to weight the A and B banks
separately. This deliberately retains cross-expert A/B terms. A separate,
frozen CLIP image encoder and text encoder provide the features for running
image and text prototypes for each task. For a generation request with a full
token input, SAME computes cosine similarities to all task prototypes,
combines image and text similarities with weights 0.2 and 0.8, and applies a
scale-28 softmax over all experts. If image feature extraction raises a
runtime error, the global prior uses text similarities alone. Short or missing
token input reuses a cached prior when one is available. During training, the
current task's prototypes
are updated on eligible multimodal outer forwards; the layer's training route
continues to use the adaptive mask.

For evaluation, each layer divides its router logits by the expert count,
applies softmax per token and averages the probabilities over tokens. It
multiplies that local mean by the FP32 image/text prior, normalizes, retains
the two largest weights, normalizes again and converts to the layer's expert
dtype. All expert slots participate in the global prior, including slots for
tasks not yet seen. The general LLaVA builder calls `merge_and_unload`; SAME
remains an unmerged dynamic layer and is explicitly placed in evaluation mode.

Router-input centered minibatch factors are recomputed by randomized QR every
20 forwards; between recomputations the latest factors are reused. After the
first 10 forwards the router gradient is split into the estimated principal
subspace and its complement; the in-subspace terms use smoothed reciprocal
singular-value scaling clamped to 0.8–1.2. On later tasks the A-expert gradient
is preconditioned by the previous task's stored factors and damping 0.9.
Utilization and input-energy-weighted importance are tracked with 0.95/0.05
exponential updates; their normalized difference controls the adaptive mask
at threshold 0.1, with the current task's expert retained. Covariance factors,
previous snapshot, validity, running metrics, mask and counters are stored
with the adapter parameters. A task boundary commits the factors; task 2
requires a valid previous snapshot. All buffers are included in the PEFT
checkpoint, including when PEFT receives an explicit parameter-only state
mapping. Non-reentrant activation checkpointing reuses the original mask and
does not count a recomputation as a second observation.

The observation and sample counters are stored as non-trainable `int64`
parameters so mixed-precision conversion cannot saturate their increments.
SAME checkpoints must contain the complete per-layer state and `int64-v1`
counters. Older floating-point or incomplete counter states are rejected:
their values can already have saturated and cannot be safely recovered.

The released PRISM code and the paper are not identical: the paper describes
an uncentered cumulative covariance over all seen tasks and a sum of
individual routed expert outputs. The release uses centered periodic
minibatch QR factors and separately weighted A and B banks. This implementation
follows the released implementation for those choices; it does not claim to
reproduce the paper's cumulative covariance formula. PRISM's state is
collected per process, and the task checkpoint carries rank 0's statistics;
no cross-rank covariance reduction is added.

The prototype anchors, their sample-count boundaries and cached prior are kept
in FP32, including after model dtype casts. This avoids low-precision boundary
counter saturation. PRISM initially creates its prototype parameters in FP32,
but a later cast of the containing model can also cast those parameters. This
is therefore a numerical adaptation when such a cast occurs. With data parallel
training, initial prototype state is broadcast from rank 0 once; subsequent
prototype updates are local to each process, and the task-boundary checkpoint
stores rank 0's state.

For factor estimation, MCITlib centers and projects in float32 before storing
the factors in the state dtype. The PRISM release centers in the input dtype,
computes QR in float32, and converts the basis back to the input dtype for
subsequent projection. Mixed-precision results can therefore differ; bitwise
equivalence and reproduction of the paper's accuracy are not claimed.

SiGMA leaves the multimodal projector trainable, matching its released training
scripts. It stores a separate **full rank-16** A/B pair for each task. During task t,
the current pair trains while previous pairs are frozen. Except for LLaVA
layer 31, it forms `(sum(previous B)) @ (sum(previous A))`, including the
cross-task terms, then adds only entries whose signs match the current
`B_t @ A_t`. The prior contribution is unscaled, matching the released
forward; the current pair uses alpha/r. Layer 31 applies the current pair only.

At inference the first task's scaled delta is the anchor. For each later task,
same-sign entries are added, while conflicting entries survive only when
`abs(delta) > 0.8 * abs(anchor)` and are scaled by one minus the cosine
similarity of the selected conflict vectors. Each same/conflict term is added
in task order to the destination base-weight dtype; the anchor is added last,
as in the release. This preserves per-add FP16/BF16 rounding. The unmerged
PEFT evaluation path makes the same sequence on a temporary weight. A merged
weight cannot be reversed exactly without preserving a full base-weight copy;
reload the base to undo it.

Both methods reject 4-bit/8-bit model loading instead of silently using a
plain LoRA layer. CPU tests exercise small Llama checkpoints; full-model execution requires the
LLaVA runtime and compatible hardware.

## SAME checkpoints and use

Set the paths in `configs/paths.yaml` for the LLaVA base model, data and
checkpoints. SAME uses the LLaVA `model_config.vision_tower` path for both
dedicated routing encoders. The configured default is
`${model_root}/clip-vit-large-patch14-336`; it must be a complete local CLIP
snapshot with configuration, tokenizer and preprocessor files, and the model
weights or all indexed weight shards. If the files are elsewhere, use the
paths configuration and mappings described in [Configuration](configuration.md).
The standard SAME train and evaluation plans pass this resolved path to the
routing runtime. Identity checks read the selected weight files, which adds I/O
when the snapshot is first bound in a process.

Each completed task saves `same_prior_state.pt` alongside its PEFT adapter
and model configuration. The sidecar contains the FP32 image/text anchors,
boundaries, cached prior when present, completed task and a content identity
for the CLIP snapshot. Keep the sidecar with its adapter. Inference requires
the sidecar task to match `adapter_config.json`; training the next task also
requires the immediately preceding task's adapter and prior. The loader checks
observed text boundaries for completed tasks and requires both image and text
boundaries to remain at their initial values for unseen tasks. A completed
task's image boundary can remain at its initial value if image feature
extraction fell back to text alone. The loader rejects missing, malformed,
out-of-order or mismatched state. CLIP identity is based on the selected
configuration, tokenizer/preprocessor files and weight contents, so the same
snapshot can be moved to another path. A different snapshot is rejected even
if its feature dimensions match.

Older SAME adapters without the image/text prior sidecar cannot be restored by
this implementation: their historical prototypes cannot be recovered from the
adapter weights. Use the matching earlier source and configuration when
reproducing outputs from those checkpoints. The sidecar is written at task
boundaries; it does not provide a mid-task optimizer or engine restart.

Keep the configured LLaVA checkpoint names, for example
`${ckpt_root}/MCITlib/UCIT/LLaVA/SAME/Task1_llava_lora` and
`Task2_llava_lora`. The LLaVA evaluation builder selects its model-loading
branch using the checkpoint-derived model name. When using a custom checkpoint
location, preserve a basename that identifies both `llava` and `lora`, and
provide the corresponding LLaVA base model; a name such as `task1` alone can
select a different loading branch.

The standard training configuration uses BF16. The routing CLIP towers use
BF16 when BF16 training is enabled and FP16 otherwise. At evaluation, the
routing towers follow the loaded model's dtype; the standard LLaVA builder
loads the base in FP16. The PRISM default inference precision is BF16, so
these settings and the FP32 prototype adaptation do not imply bitwise or
end-to-end numerical equivalence with the original release.

After configuring paths, inspect the SAME plan before running it:

```bash
mcitlib doctor --method same --benchmark ucit --paths configs/paths.yaml --validation-fixes
mcitlib train --method same --benchmark ucit --paths configs/paths.yaml --validation-fixes --dry-run
mcitlib eval --method same --benchmark ucit --paths configs/paths.yaml --after-task 1 --validation-fixes --dry-run
```

Use `mllm_dcl` for the other supported benchmark. The configured task sequence
and its completed checkpoints determine continuation and evaluation paths.

## Budgets

The normal method configurations are full-data plans. SAME UCIT follows the
PRISM release's rank 48 total (six rank-8 experts), alpha 96, 2e-4 learning
rate, one epoch, two GPUs, accumulation 1, and per-device batch 8 for task 1
and 6 thereafter. SiGMA UCIT uses rank 16 **per task**, alpha 32, 2e-4, one
epoch and effective batch 64; V4 uses two GPUs, microbatch 8 and accumulation
4 to match the release's four-GPU 8 × 2 effective batch. MLLM-DCL uses five
experts, SAME rank 40/alpha 80 or SiGMA rank 16/alpha 32, effective batch 64,
learning rate 2e-4, and task epochs 1/3/1/2/1. This DCL schedule is an
explicit V4 adaptation rather than a claimed upstream schedule.

The `validation_batch` worker is a separate 3k/1k sample workflow for these
methods. It explicitly sets learning rate 2e-4, one epoch, microbatch 4 and
accumulation 8 on two GPUs (or 4 on four GPUs), for global batch 64. It does
not alter the full-data YAMLs or method-specific adapter hyperparameters.
The bounded integration smoke helper uses a third, much smaller budget: four
training examples and two optimizer steps per task. It is execution evidence,
not a quality evaluation.

