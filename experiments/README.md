# Run artifacts

**Real experiments: PENDING REAL DATA.** All generated artifacts remain ignored.

Each named training folder in `experiments/results/` contains:

| File | Meaning |
|---|---|
| config.yaml | Resolved YAML, with inherited values expanded |
| metadata.json | Seed, versions, device, parameters, Git state, provenance |
| inspection.json | Excluded records/images and usable camera counts |
| splits.json | Original row mapping, chronological indices, discarded boundary rows |
| best.pt / latest.pt | Model, optimizer, scheduler, history, split and config state |
| metrics.json | Completion status, best validation loss, measured training duration |
| history.csv | Per-epoch train/validation regression metrics and learning rate |
| evaluation.json / predictions.csv | Clean held-out metrics and individual predictions |
| robustness.json / robustness.csv | Clean-relative and per-condition corruption measurements |
| robustness_predictions.csv | Reproducible corruption predictions and seeds |
| uncertainty.json | Frozen validation cutoff and clean/pooled test metrics |
| uncertainty_validation.csv | Validation predictions used to choose the cutoff |
| uncertainty_predictions.csv | MC mean/std/variance and errors under each test condition |
| uncertainty_by_condition.csv | Detector metrics separated by condition |
| selective_prediction.csv | Requested/actual coverage and accepted MAE/RMSE |
| failure_cases_*.csv | Selected examples, including empty categories |

Figures are generated under `reports/figures/<run>/`, not stored in Git. Every
figure is labeled with the data status and exported to PNG/PDF. Tables are in
`reports/tables/ablation.{csv,md}` for real data and
`synthetic_ablation.{csv,md}` for smoke data. Empty real results stay pending.

`python scripts/run_ablation.py --train` trains the five real-data configurations
only when absent and performs clean evaluation. `--train --smoke` uses two epochs
on procedural images. A second invocation reuses matching completed runs and
verifies their dataset fingerprints. Run robustness/uncertainty separately on
the selected checkpoints before aggregating those fields. Dropout-free models
do not produce MC uncertainty measurements.

Do not rename a synthetic run to imply real data. Do not alter a completed run's
config in place; use `--name` for a new experiment. Resume `latest.pt` with the
original config and a larger epoch limit. Re-evaluate after a changed best
checkpoint; aggregation ignores measurements for a different checkpoint hash.

Drive directories are separate. They contain telemetry, manual events, metadata,
an end timestamp, optional camera frames, and `autonomy.json`. They are not offline
training folders and are excluded from ablation aggregation.
