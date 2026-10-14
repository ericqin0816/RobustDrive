# Resume wording and evidence requirements

## Supported by the first personal recording

- Implemented a PyTorch behavioral-cloning pipeline with a NVIDIA-inspired steering
  CNN, temporal data splits, recovery-camera augmentation, and a simulator adapter;
  trained on 6,606 personal simulator frames and evaluated on 1,404 chronological
  test frames, achieving normalized steering RMSE 0.1252.
- Developed reproducible image-corruption and Monte Carlo dropout tools to study
  steering errors across 11 corruptions and five nonzero severities; documented
  a negative uncertainty result (clean high-error ROC-AUC 0.2837) rather than
  claiming reliable failure prediction.

Use wording you can personally explain and demonstrate. The first recording has
now been collected and used. The [research report](research_report.md) contains
its evidence and limitations. Do not describe live control or protocol tests as
verified autonomous lap success.

## Templates after verified real experiments

- Trained a vision-based steering CNN on **[N training frames / N camera samples]**
  from personal simulator recordings, achieving **[MAE]** on **[held-out scope]**
  and completing **[X of Y]** autonomous laps without intervention at **[speed]**.
- Benchmarked **[model]** across **[K corruption types]** and **[S nonzero severity
  levels]**, measuring **[absolute / relative change]** in held-out steering MAE
  relative to clean images under **[defined conditions]**.
- Evaluated MC dropout as a high-error detector, measuring **[recall]** at
  **[precision or false-alert rate]** for **[error threshold]**, and reducing
  accepted-prediction MAE by **[Z]** at **[Y]% coverage** on **[test scope]**.

Keep all placeholders until the artifacts exist. Detection recall, precision,
and selective coverage are distinct quantities. State whether a relative change
is a percent or an absolute difference. Report null/negative findings honestly.

## Evidence map

| Claim | Required evidence |
|---|---|
| Personal data and sample count | Recording notes, inspection, split manifest |
| Steering accuracy | Clean evaluation JSON and checkpoint hash |
| Autonomous completion | Real telemetry, lap/takeover/failure events, video |
| Robustness change | Per-condition CSV, severity definitions, clean comparator |
| Detection/rejection | Validation cutoff, test metrics, coverage table |

**First-recording offline evidence is available. Annotated lap completion,
generalization, and advance-warning claims remain pending.** The rejection
improvement template above is not supported by the current negative result.
