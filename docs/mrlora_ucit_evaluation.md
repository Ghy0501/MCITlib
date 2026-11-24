# MR-LoRA UCIT evaluation

MR-LoRA UCIT evaluates a router checkpoint, selects predictions from six independent experts for the evaluated domain, and scores the selected answers separately under `UCIT_router`. Raw router output is stored in `UCIT_router_raw`. All six experts are available at every router stage, so this is a retrospective protocol rather than strictly online continual learning. Letter priority and random fallback for invalid router answers are recorded for every question. Expert-only scores and router scores are separate result sources.

The normal validation repair computes all 36 expert/domain combinations before routing. For an existing completed sample run, an evaluation-only plan can reuse verified predictions and compute only the missing combinations, without modifying its checkpoints, results or logs. A cell is reusable only if the completed recorded inference commands, model paths, four source chunks, merged prediction order, question IDs, sample manifest hash, scorer denominator and selected runtime source files agree. Checkpoint configuration files are hashed; large weights are checked for presence, size and modification time only. The selected source checks do not fingerprint the entire environment and cannot prove launch-time identity. Caption reference order and image IDs are checked because the original caption scorer assigns images by result order. Every new router join rechecks IDs, sample counts and manifest hashes, then passes the chosen predictions to the original scorer.

From the checkout root, set `SOURCE_RUN` to the absolute path of the completed source queue, and choose fresh absolute `RESULTS` and `LOGS` paths under this checkout's `reports/` and `logs/` respectively. Use the Python interpreter recorded for the source MR-LoRA job as `JOB_PYTHON`:

```bash
JOB_PYTHON=/absolute/path/to/recorded/bin/python
SOURCE_RUN=/absolute/path/to/completed/queue
RESULTS="$PWD/reports/mrlora-ucit-eval/results"
LOGS="$PWD/logs/mrlora-ucit-eval"
PLAN="$PWD/reports/mrlora-ucit-eval/plan.json"
DOCTOR="$PWD/reports/mrlora-ucit-eval/doctor.json"

"$JOB_PYTHON" -m mcitlib.methods._mrlora.ucit_eval plan \
  --old-run-dir "$SOURCE_RUN" --results-root "$RESULTS" \
  --logs-dir "$LOGS" --plan-output "$PLAN"
```

Planning only reads the source run and writes the JSON plan; it does not run inference or create the results and logs directories. Review the listed checkpoint and sample evidence, reused and missing cells, exact commands, and fresh destinations. A plan is not evidence that any new prediction or metric exists. Before executing, run the plan-specific doctor in the intended four-GPU environment:

```bash
"$JOB_PYTHON" -m mcitlib.methods._mrlora.ucit_eval doctor \
  --plan-file "$PLAN" --report-output "$DOCTOR"
```

The doctor rebuilds the plan and checks the recorded interpreter, packages, four visible GPUs, vendored imports, distinct evaluation entry points and representative input images without loading model weights or running inference. Execution requires a passing doctor report bound to the exact plan bytes and rechecks the source evidence and fresh destinations. Once the operator has reviewed the plan and doctor report, the authorized execution is:

```bash
"$JOB_PYTHON" -m mcitlib.methods._mrlora.ucit_eval execute \
  --plan-file "$PLAN" --doctor-report "$DOCTOR"
```

The executor copies verified source predictions with exclusive destination creation, computes missing expert predictions and router outputs, and stops on a failing step. It writes `metrics.json` under the new logs directory only when all 21 lower-triangular router results have the expected denominator, question IDs and routing evidence; it calls the existing `mcitlib.eval.metrics` formulas. The new evaluation does not change the source queue's status or merge its matrix with another run. A sample evaluation is not a formal reproduction claim.
