# Validation queue reports

The reporting tool reads a `validation_batch` queue and exports a timestamped observation without launching training, evaluation, doctor, model code, or a GPU task. It reads the queue state, job events, existing doctor and command files, and the queue's result files. Reports belong in a separate directory, outside the queue's logs, results and checkpoints.

From the project root, replace `<run-id>` with the queue's run identifier:

```bash
python -m mcitlib.cli report \
  --run-dir "logs/validation_3k_1k/<run-id>" \
  --output-dir "reports/<run-id>"
# Equivalent standalone entry point:
python -m mcitlib.reporting \
  --run-dir "logs/validation_3k_1k/<run-id>" \
  --output-dir "reports/<run-id>"
```

The command prints absolute artifact paths as JSON. Reusing an output directory replaces these report artifacts; use a new directory to keep an older snapshot. Files are replaced individually, so do not treat the five artifacts as a transactional set while another process is writing the same directory.

* `report.json` contains the full observation, all lower-triangular cells, sample provenance, events, warnings and job records.
* `report.csv` contains one row per job with status, phase, task, elapsed time, matrix completeness, metrics, errors and major logs.
* `cells.csv` contains one row per lower-triangular cell: method, benchmark, training stage, evaluation task, score, observed sample count and availability. `observed` means a score was read from a result or matrix file; `unavailable` means none could be safely read; `suppressed_by_status` means a pending, not-run or blocked job had no usable result for this report.
* `report.md` is a human-readable summary, sample-count table and matrix for every job, with available failure and log evidence.
* `manifest.json` is the provenance portion of the JSON report.

JSON uses `null` for unavailable values; CSV and Markdown show `N/A`. A real measured zero remains zero. Cells outside the lower triangle are not required and do not appear in `cells.csv`. Observed sample counts come only from a `Samples:` line in a result file; an absent count is `N/A`, even when the sample manifest states a planned count.

The state-level `execution_profile` is recorded as supplied; older states without it show `N/A`. For each job with `commands.json`, resource evidence comes from the recorded commands: DeepSpeed `--include` gives training world size, and the explicit per-device batch and gradient accumulation values give effective batch as `world size × per-device batch × accumulation`. Missing, ambiguous or contradictory values stay `N/A`. Evaluation records contain each explicit chunk count, chunk index and recorded `CUDA_VISIBLE_DEVICES` assignment. The JSON and manifest retain per-command details; Markdown and CSV summarize the observed values. These are recorded command plans and do not prove that a command was started or completed.

The `four_gpu_validation` profile is designed for four-way main training and evaluation while KeepLoRA gradient preparation retains two shards, preserving its existing gradient merge behavior. Each report shows only what the job's recorded commands actually specify, including those separate gradient-preparation commands; it does not infer four-GPU execution from the profile name. Missing command files remain `N/A`.

When supplied by the queue, `previous_run`, `related_runs` and `transition_receipt` are linked in the report and manifest. The receipt is read as metadata and fingerprinted if available. These links do not import another run's job statuses, matrices or metrics: each report remains a snapshot of its own queue.

The queue's recorded job statuses are authoritative. Its summary is recomputed from those job records, with a warning if the recorded summary disagrees. Pending, `not_run`, `blocked`, and `blocked_disk` jobs do not acquire scores from potentially stale files. `running`, `failed` and `stopped` jobs can display available partial results but do not receive aggregate metrics. A result from a stage beyond the recorded current task is suppressed with a warning. A completed job receives MFT, MFN, MAA and BWT only when its full benchmark matrix is available and consistent and the recorded doctor/events do not contradict completion. The reporter calls `mcitlib.eval.metrics.continual_metrics` for that full matrix; it does not change or approximate those formulas. A complete matrix for a failed or running job remains visible with its status, but aggregate metrics remain `N/A`.

For an MR-LoRA DCL job, a `metrics_result_dir` recorded in the job state or `commands.json` explicitly identifies the separate `DCL_router` result directory. The declaration must be the absolute `<run results_root>/mrlora/mllm_dcl/mrlora/DCL_router` path, and declarations in both files must agree. The report reads its matrix only from that router directory and accepts `metrics.json` only when its recorded `source` is the same directory. Missing or conflicting router data is not replaced by expert cross-task scores. If the declaration is absent, the report uses the `DCL` lookup and does not label the results as a router matrix. The validation uses a retrospective five-expert routing pool, which can include experts from later stages; it should not be described as a strictly online protocol.

MR-LoRA UCIT follows the same explicit-source rule for `UCIT_router`. A job without that declaration can display its available `UCIT` expert scores, but its aggregate metrics are suppressed because an expert matrix is not evidence of routed evaluation. The UCIT protocol selects among all six experts for the same evaluation domain, including later experts at earlier stages; this is retrospective rather than strictly online. The separately planned evaluation of existing checkpoints and its strict source checks are described in [MR-LoRA UCIT evaluation](mrlora_ucit_evaluation.md).

When state or commands record `expected_eval_samples` and per-domain `expected_eval_hashes`, both declarations are retained separately in the job's manifest provenance. A disagreement suppresses aggregate metrics with a warning. The report records these declarations; the validation worker checks the actual annotations against them when it performs the routing join.

Malformed, nonfinite, changing or conflicting cells are not replaced with zero. Conflicts between result files and a recorded matrix, an incomplete matrix for a completed job, partial event lines, mismatching status records and other inconsistencies appear in warnings. A queue can advance as files are being read. `snapshot_at` is the initial state-read time, `collected_at` is the end of collection, and the manifest contains the initial state SHA-256 and whether the state changed by the end. The files are read sequentially, not as a filesystem transaction; even an unchanged state does not guarantee the events and results were frozen. Re-run the command for a newer snapshot.

The reporter associates a sample manifest only when explicitly recorded in state or when exactly one sample data directory in the recorded paths profile has a corresponding `configs/<name>/manifest.json`. Per-job planned split counts additionally require the observed effective paths to contain an exact source-to-output mapping from the manifest in the applicable benchmark or common scope. Unmatched counts are `N/A`. The run-level sample table displays the manifest's dataset counts, not proof that any particular job consumed or evaluated them. Caption images and caption references are separate counts. The manifest retains existing source/output hashes, sampling rules, seed and auxiliary-manifest association; it does not rehash datasets or check image decoding.

Provenance separates the experiment project recorded in the queue state from the checkout and Python running the reporter. Small selected files, including recorded configuration, commands, doctor reports, and selected experiment and reporter source files, are hashed at collection time. The experiment Git HEAD is also observed if available, but neither HEAD nor these selected hashes describes all dirty files or proves what code, environment, configuration or samples were used at launch. Recorded environment names and interpreter paths, plus Python details already present in doctor reports, are evidence rather than an independently reconstructed environment. No weight directories are scanned or hashed. A sample validation report is not a claim of formal reproduction.

Python callers can use `build_report(run_dir)` to obtain the JSON-serializable observation, `write_report(report, output_dir)` to export it, or `generate_report(run_dir, output_dir)` to do both. The last two return a dictionary of absolute artifact paths, including `cells_csv`.
