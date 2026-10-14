# Resume wording and evidence requirements

## Accurate before real simulator experiments

- Implemented a PyTorch behavioral-cloning pipeline with a NVIDIA-inspired steering
  CNN, temporal data splits, recovery-camera augmentation, and a simulator adapter;
  validated training and evaluation using explicitly synthetic smoke data.
- Developed reproducible image-corruption and Monte Carlo dropout tools to study
  steering errors, high-error detection, and uncertainty-based prediction rejection.

Use wording you can personally explain and demonstrate. Do not say the model was
trained on your simulator driving data until you have collected and used it.
Do not describe protocol tests as autonomous-driving success.

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

**Research results: PENDING REAL DATA.**
