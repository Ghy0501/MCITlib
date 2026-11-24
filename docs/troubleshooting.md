# Runtime checks and troubleshooting

Run a complete `mcitlib doctor` in the intended method environment before GPU execution. Stop on failed checks, fix that environment or input, and rerun doctor.

| Symptom | Check and action |
| --- | --- |
| Missing package or version mismatch | Use the matching pinned requirements from `envs/`; keep ModalPrompt and Transformers 4.37 runtimes separate. |
| flash-attention cannot load CUDA libraries | Check the installed wheel, CUDA runtime, and library search path in the actual training environment. |
| Data loading raises `AF_UNIX path too long` | Keep `TMPDIR` short enough for multiprocessing's additional socket names. Use a private short directory under `/tmp`, or a short private alias to project-owned temporary storage. |
| A host `nvidia-smi` query times out | Record the query and exit status separately from model execution. Keep GPU leases while retrying the idle check; require a conclusive idle result before starting work or releasing ownership. |
| A GPU observer reports a foreign PID during process exit | Correlate PID, kernel start time, pidfd state and observation time. NVML and `/proc` snapshots are not atomic; retain leases while resolving an exited owned process, and reject identities whose ownership cannot be proven. |
| A long queue reports `owned PID was reused while still tracked` | Retire the old identity only after its original pidfd proves exit; verify a new descendant independently. Reclaim exited pidfds during long runs while keeping visible zombies pinned for reaping. Before recovery, verify GPU ownership, completion markers and saved artifacts, then resume unfinished steps. |
| Evaluation cannot import `openai` | Install the scoring dependency in the interpreter used by that evaluation command. |
| DeepSpeed reports repeated gradient reduction in OLoRA | Use the default non-reentrant checkpoint scope and verify actual checkpoint calls. Keep the intended global batch and orthogonal loss. |
| Parameter or projection shape mismatch | Match backbone, expert count, prompt dimensions, and checkpoint configuration; use partition-aware loading under ZeRO-3. |
| A task label cannot be parsed | Use the registered task order and labels; do not interpret a task name as an integer. |
| A tuner is missing in a fresh process | Register its tuner and state-dict hooks before importing and loading model entries. |
| SEFE merge fails generation configuration validation | Greedy merges set the ignored `temperature` and `top_p` values to `1.0` before saving. Check the remaining generation settings if validation still fails; sampling-enabled settings are preserved. |
| SEFE InternVL evaluation fails to load an automatically mapped vision tower | In the pinned Transformers 4.37 environment, SEFE maps complete InternViT encoder layers and loads the vision tower in FP16. Check the resolved device map and available GPU memory. |
| DISCO/HiDe statistics are empty after reload | Verify prototype, anchor, feature, and boundary state was saved; evaluate a checkpoint containing that state. |
| Generation repeatedly rebuilds expert weights | Use the default builder and generation scope; check evaluation and gradient mode. |
| GPU memory is insufficient | Reduce microbatch and increase accumulation to preserve effective batch. Inspect auxiliary preparation separately from main training. |
| KeepLoRA preparation exceeds device capacity | Use hardware that can hold its gradient preparation workload. This remains a hardware prerequisite. |
| Replay or router input is missing | Generate the auxiliary data and check its mappings. Replay answers must be task QA, not router expert letters. |
| MR-LoRA scores look like individual experts | Check the explicit `UCIT_router` or `DCL_router` source and question-ID joins. |
| A result matrix has missing cells | Inspect job status, per-step logs, merge inputs, scorer output, and completion markers before aggregating. |
| A running aggregate log appears unchanged | Follow the active step log recorded in `state.json`; aggregate logs update after steps finish. |
| Disk protection blocks a step | Free space outside required adapters and results, or select another checkpoint root. Do not bypass the capacity check. |

Use [training behavior](training_entry_repairs.md), [implementation contracts](llava_implementation.md), and [MR-LoRA evaluation](mrlora_ucit_evaluation.md) for method-specific checks.

Record model, method, benchmark, task order, sample manifest, training budget, interpreter, and exact failing command with every issue. Keep detailed run evidence in `reports/`, with predictions and logs in their configured locations. A complete sample execution verifies the pipeline; it does not by itself establish paper-level quality.
