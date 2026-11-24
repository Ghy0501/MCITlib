# Experiment summaries

The summary reads one validation queue and writes a snapshot without starting training, evaluation, doctor or model code. It uses the existing reporting and continual-metrics implementations. From the project root:

```bash
# All methods and benchmarks recorded in this one queue.
python -m mcitlib.summary \
  --run-dir "logs/validation_3k_1k/<run-id>" \
  --output-dir "reports/<run-id>-summary"

# Select one method and benchmark, or specify either filter by itself.
python -m mcitlib.summary \
  --run-dir "logs/validation_3k_1k/<run-id>" \
  --method mrlora --benchmark mllm_dcl \
  --output-dir "reports/<run-id>-mrlora-dcl"
```

The output directory must be outside the input run's logs, results and checkpoint roots. Files of the same names in the output directory are replaced individually; use a new directory for a separate durable snapshot. The command prints absolute output paths as JSON.

- `report.md` shows job status, the lower-triangular matrix, missing cells, metric values and recorded training parameters.
- `report.json` retains the full per-job matrix, cell sources, warnings, selection and provenance. `report.csv` has one row per selected job; `cells.csv` has one row per required matrix cell.
- `training_profiles.csv` has a separate row for each recorded main-training, router-training or gradient-preparation command. `manifest.json` records source metadata and fingerprints.

Unavailable values are JSON `null` and Markdown/CSV `N/A`; a measured zero remains zero. A complete, consistent matrix from a completed job uses the existing `mft`, `mfn`, `maa` and `bwt` formulas. Partial, failed, blocked or conflicting results do not acquire aggregate metrics. A complete matrix may still be shown for a failed or running job, with its actual status and unavailable aggregates.

For MR-LoRA UCIT and MLLM-DCL, the queue's state or recorded commands must identify the matching run-local `UCIT_router` or `DCL_router` directory. Missing, inconsistent or escaping declarations produce N/A; expert scores are never substituted. Router score filenames are checked as router outputs. The fixed pools include experts from later stages, so the routing protocol is retrospective, not strictly online. UCIT expert-only matrices do not establish a routed metric.

The table and `training_profiles.csv` show the recorded command's world size, per-device batch, accumulation, effective batch, LR and epochs when available. The manifest sample column comes from the existing reporter's matched sample-manifest mapping. If a job supplies `uniform-profile.json`, its planned dataset examples and optimizer steps are shown only when its command-list hash and main-training entries match the recorded commands; these remain producer-planned values, not independently recounted data or completed steps. A sidecar marked `example_only` is a preview, not a run record. Missing or inconsistent planned values are N/A. Main training and router training retain separate sequence labels; KeepLoRA gradient preparation is shown separately. Epochs may intentionally differ between methods or between main and router training; the report does not assume one budget for every method. Hardware `execution_profile` is not the same as a scientific training profile. A declared scientific profile ID is shown only when the recorded project-relative file exists and its collection-time SHA-256 matches; it does not prove launch-time identity. A missing or conflicting profile is N/A or a warning, not an inferred match. If a DeepSpeed JSON is readable, its recorded optimizer and scheduler types are shown alongside the CLI scheduler; these observations do not reconstruct the optimizer's effective behavior. The current tool does not assert that two runs used an identical protocol.

## An MR-LoRA UCIT evaluation-only run

An evaluation-only plan for existing checkpoints is a separate result source; it need not have a queue `state.json` in its output directory. To summarize one such plan, pass the exact plan and, when available, the controller state that records its original plan SHA-256:

```bash
python -m mcitlib.summary \
  --evaluation-plan "reports/mrlora-ucit-eval/plan.json" \
  --execution-state "path/to/that-evaluation-controller-state.json" \
  --method mrlora --benchmark ucit \
  --output-dir "reports/mrlora-ucit-eval/summary"
```

The plan must name its own `UCIT_router` output, all 21 lower-triangular score destinations and the original final metrics command. The execution state must bind the same plan hash. Without a bound state the status is unknown and aggregates are N/A; a completed controller additionally requires a matching successful final-command marker and a metrics file whose source is that router directory. The related training queue is consulted only for recorded training provenance, after its state and commands match the plan hashes; its matrices and metrics are not imported.

Live files are read sequentially and may change during collection. The report records snapshot time and source hashes where available, not a transaction or launch-time attestation. Sample validation and successful command completion do not establish formal reproduction or scientific quality. See [validation queue reports](reports.md) and [MR-LoRA UCIT evaluation](mrlora_ucit_evaluation.md) for the underlying data and protocol rules.
