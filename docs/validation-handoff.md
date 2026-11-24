# Validation queue handoff

A handoff finishes the current job before starting the remaining jobs with an explicitly reviewed source snapshot and execution profile. Each queue keeps its own run ID, outputs, source hashes, and results.

Use `tools.transition_validation_queue preview` to record the runtime project, source snapshot, queue PID and start ticks, finish-job boundary, target run ID, profile, and fresh control paths. Inspect the resulting manifest and exact plan. Preview does not launch training.

```bash
python -m tools.transition_validation_queue --help
python -m tools.transition_validation_queue preview --help
python -m tools.transition_validation_queue go --help
```

Execute `go --manifest <review.json> --sha256 <reviewed-sha256>` only after checking the manifest and prerequisites. The monitor verifies process identity, source hashes, locks, and the job boundary before synchronizing files and launching the target queue. A training or evaluation worker is allowed to finish; the handoff uses a verified doctor boundary.

Use a fresh control directory and target run ID. Keep model/data mappings and output roots explicit. Runtime data, weights, logs, results, and local paths are excluded from source synchronization. Inspect the handoff receipt and the target queue's real state and events; preview success alone is not startup evidence.
