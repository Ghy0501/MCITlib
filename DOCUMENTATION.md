# Documentation

| Topic | Guide |
| --- | --- |
| Installation, paths, and basic commands | [README](README.md) |
| New machine: pinned weights, data sources, and first score | [Public quick start](docs/public-start.md) |
| CL-VISTA: Video-LLaVA and VideoLLaMA2 installation, binding, training, inference, and scoring | [Video runtime](docs/video-runtime.md) |
| InternVL model and method behavior | [InternVL implementation](docs/internvl.md) |
| InternVL four-GPU sample queue | [InternVL sample validation](docs/internvl_sample_validation.md) |
| Configuration templates and precedence | [Configuration](docs/configuration.md) |
| Porting a method to a backbone | [Method porting guide](docs/backbone-baseline-porting.md) |
| Sample data and validation queues | [Sample validation](configs/validation_3k_1k/README.md) |
| Four-GPU execution | [Execution profile](docs/four_gpu_validation.md) |
| Queue handoff | [Handoff](docs/validation-handoff.md) |
| Matrices and continual learning metrics | [Summaries](docs/summary.md) |
| Report fields and result sources | [Reports](docs/reports.md) |
| OLoRA, DISCO, and HiDe | [Implementation contract](docs/llava_implementation.md) |
| Training and state handling | [Training behavior](docs/training_entry_repairs.md) |
| PCLR rank pools and consolidation | [PCLR](docs/pclr.md) |
| SAME routing and SiGMA task adapters | [SAME and SiGMA](docs/same_sigma_provenance.md) |
| Octopus history and gradient orthogonalization | [Octopus](docs/octopus.md) |
| DGG gradient guidance and replay | [DGG](docs/dgg.md) |
| MR-LoRA routed evaluation | [Evaluation protocol](docs/mrlora_ucit_evaluation.md) |
| Runtime checks and common failures | [Troubleshooting](docs/troubleshooting.md) |
| CPU and model integration checks | [CI](docs/ci.md) |
| Video data and additional configuration preparation | [Offline preparation](preparations/other_backbones/README.md) |

`docs/` is for reusable method descriptions and instructions. Keep individual runs private: reports in `reports/`, logs in `logs/`, predictions and scores in `results/`, and weights in the configured checkpoint root. Do not publish machine paths, internal handoffs, per-run metrics or checkpoint fingerprints as examples. Public dependency revisions and source checksums are part of the installation instructions.
