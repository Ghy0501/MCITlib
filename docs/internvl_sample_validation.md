# InternVL sample validation

The validation queue runs the UCIT and MLLM-DCL samples serially on GPUs 0–3.
Each task uses 3,000 primary training examples and 1,000 evaluation examples.
The four-device profile records a global batch of 64, a main learning rate of
`2e-4` and one main epoch for all methods, including ModalPrompt. MR-LoRA
router training keeps its separate recorded budget and all expert/router
evaluation steps. These are sample checks, not full benchmark reproductions.

Prepare the standard sample paths and the Replay/router auxiliary paths using
the project data preparation tools. Select Python interpreters that satisfy
the doctor checks for the normal, SEFE/KeepLoRA and ModalPrompt environments.
An individual unsupported method, failed doctor or failed sequence is
recorded and the queue proceeds to the next job.

Freeze the source, configuration and sample inputs without using a GPU:

```bash
python -m mcitlib.validation_batch bind-internvl \
  --paths configs/paths.validation.yaml \
  --auxiliary-paths data/validation_aux/paths.yaml \
  --output logs/validation_bindings/internvl-sample.json
```

Use the returned SHA-256 when launching a fresh run:

```bash
python -m mcitlib.validation_batch execute \
  --backbone internvl --run-id internvl-sample-01 \
  --paths configs/paths.validation.yaml \
  --auxiliary-paths data/validation_aux/paths.yaml \
  --source-binding logs/validation_bindings/internvl-sample.json \
  --source-binding-sha256 SHA256_FROM_BIND_COMMAND \
  --python-internvl /path/to/main/bin/python \
  --python-t437 /path/to/sefe-keep/bin/python \
  --python-modalprompt /path/to/modalprompt/bin/python \
  --gpus 0,1,2,3 --execution-profile four_gpu_validation --min-free-gib 32
```

The queue obtains exclusive project locks under `logs/.gpu-leases` and checks
the four devices are idle before starting. All launchers sharing these GPUs
must use the same lock directory; `--gpu-lease-root` selects a different
project-owned shared directory. A supervisor that already owns the locks can
pass their open file descriptors using `MCITLIB_INTERNVL_LEASE_FDS` and the
same directory using `MCITLIB_INTERNVL_LEASE_ROOT`. The queue checks each
descriptor's inode and exclusive lock before running and retains it while
children exist. GPUs outside 0–3 are rejected for this sample queue.

Checkpoints, results and logs are isolated under their configured root,
`internvl/<run-id>`, and existing output paths are never overwritten. Each job
has a fresh-interpreter import receipt, doctor reports, exact command list,
per-step events and markers. Source/configuration and input hashes are checked
throughout the run. A minimum 32 GiB free-space floor is enforced, with
additional reserves before large SEFE/KeepLoRA merge and SEFE top-P writes;
large methods may therefore be blocked while smaller jobs can finish.

Inspect or stop the specific queue with:

```bash
python -m mcitlib.validation_batch status --run-dir /path/to/logs/internvl/internvl-sample-01
python -m mcitlib.validation_batch stop --run-dir /path/to/logs/internvl/internvl-sample-01
python -m mcitlib.reporting --run-dir /path/to/logs/internvl/internvl-sample-01 --output-dir /path/to/report
```

The reporting command reads completed result cells and distinguishes missing
results from zero scores. Re-run it for a fresh snapshot, or use a site
supervisor to refresh reports while the queue is active.
