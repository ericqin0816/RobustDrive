# First personal recording: frozen numerical evidence

These are measured outputs from `play_438c62ffaa42`, trained using source commit
`8566dbe5ef3950ee8a30d445bb9713fe1b64c97e` (clean tree), seed 42, and CPU PyTorch.
The raw recording and trained checkpoints remain local. This snapshot contains
aggregate JSON/CSV reports, configurations, learning curves, provenance, and an
overview figure; it contains no camera dataset or model weights.

- [Research interpretation](../../research_report.md)
- [Dataset, split counts, checkpoint hashes, and original artifact hashes](manifest.json)
- [Three-model comparison](comparison.csv)
- [Post-hoc label-imbalance diagnostics](diagnostics.json)
- [Unannotated live-drive observation](drive_observation.json)
- [Overview figure](overview.png)

Each model folder preserves its original configuration, training metadata,
training metrics/history, clean evaluation, and per-condition robustness values.
The uncertainty folder also contains detector and selective-prediction results.
The 55 nonzero corruptions reuse 1,404 test images; they do not enlarge the number
of independent observations. The dropout model's deterministic evaluation differs
from its 20-pass MC mean evaluation.

Source artifacts remain under `experiments/results/play_438c62ffaa42_<variant>/`.
The manifest hashes the copied source files. Overview and diagnostic summaries
were derived from these outputs and their local prediction records. Reproducing
the exact experiment requires the preserved local dataset; a fresh clone alone
cannot regenerate personal-data measurements.

The first detector result is negative. No lap-success percentage, verified absence
of failure, or advance-warning claim can be inferred from this snapshot.
