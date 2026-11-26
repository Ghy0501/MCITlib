# Porting a continual-learning method to a backbone

Use `mcitlib list` for the supported method, backbone and benchmark combinations in your checkout. The listing describes what the CLI enables, not measured accuracy. To inspect how an existing combination is wired, print its plan before running it:

```bash
mcitlib list
mcitlib train --method replay --backbone llava --benchmark ucit --task 2 --dry-run
```

Start with the [method configurations](../configs/methods) and [method implementations](../mcitlib/methods). Use the [configuration guide](configuration.md) to trace the effective arguments and [benchmark inputs](benchmark-inputs.md) to bind annotations and media.

## What a port must preserve

- **Model inputs:** use the target backbone's tokenizer, processor, projector and actual adapter target layers. For video, preserve frame order, temporal sampling and media tokens in both training and generation.
- **Task transitions:** carry forward the state required by the method, not just the adapter weights. Check which parameters should update and which should stay frozen after switching tasks.
- **Evaluation:** use the benchmark's held-out questions and native scoring procedure. Join routed predictions by question ID; verify coverage and the denominator before interpreting a score.

| Methods (CLI IDs) | Method-specific check |
|---|---|
| `lora_ft`, `olora`, `sefe`, `keeplora` | Reload the adapter and any trainable projector. On a later task, check the orthogonality term for OLoRA, key-element regularization and merge for SEFE, or subspace initialization and frozen LoRA A for KeepLoRA. |
| `replay`, `mrlora` | Replay mixes original QA examples. MR-LoRA trains experts and a separate router with expert-selection labels. Do not substitute either training file for the other. |
| `moelora`, `clmoe`, `smolora`, `sigma` | Verify expert ordering, routing and parameter freezing on a task change. Preserve SMoLoRA's instruction embeddings and SiGMA's task-ordered merge. |
| `disco`, `hide`, `same`, `modalprompt` | Reload and exercise the state used at inference: DISCO prototypes, HiDe routing anchors, SAME's multimodal prior, or ModalPrompt's learned prompts. An adapter-only reload does not check these paths. |
| `pclr`, `dgg`, `octopus` | Trigger the additional stage: PCLR integration, DGG guidance from the prior task, or Octopus gradient collection and second-stage optimization. Preserve each stage's data and budget. |

Test the smallest task transition that activates the changed behavior. Compare parameters before and after an update, save, reload in a fresh process, and run held-out generation through the target backbone. A dry run checks arguments; it does not test training, checkpoint contents or scoring.
