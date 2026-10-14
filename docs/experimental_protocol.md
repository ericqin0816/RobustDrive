# Experimental protocol

**Current scientific status: first recording evaluated.** The original protocol
preceded the first real-data analysis; its splits, corruption definitions, error
threshold, and uncertainty quantile were retained. See the
[research report](../reports/research_report.md). Future tuning needs new held-out
data; the first test set has now been inspected.

## A–C: learn steering and test the closed loop

1. Collect, inspect, and freeze one chronological recording. Keep original bytes.
2. Use the same split manifest, seed, crop, optimizer schedule, and epoch limit for
   all comparisons. Do not shuffle rows before splitting or expand cameras first.
3. Train the center-only baseline; choose its checkpoint using validation MSE.
4. Compare +flip, +side cameras, +visual augmentation, then +dropout. The progressive
   ablation identifies the added component relative to the previous row; it does
   not isolate all possible interactions. Side cameras increase sample count and
   optimizer steps per epoch, so this is not a compute-matched comparison.
5. Evaluate each selected checkpoint once on the test block. Report counts, MSE,
   MAE, RMSE, and R² only if target variance is nonzero. Training metrics include
   augmentation/dropout; validation metrics do not, so their distributions differ.
6. Attempt real simulator laps with a fixed start, track, speed, and takeover rule.
   Include failed/abandoned attempts. Preserve event logs and checkpoint hashes.

For errors `e_i = prediction_i − recorded_steering_i`:

- `MSE = mean(e_i²)`
- `MAE = mean(|e_i|)`
- `RMSE = sqrt(MSE)`
- `R² = 1 − sum(e_i²) / sum((steering_i − mean(steering))²)` when defined

The labels are normalized simulator steering, not wheel-angle degrees. Offline
predictions are not clipped; clipping occurs only for vehicle commands.

## D: controlled visual robustness

Corrupt the same **test** RGB images before deterministic preprocessing. Keep
labels fixed. No corruption is applied to training/validation unless explicitly
part of the training-augmentation config. Severity `s=0` returns a pixel-identical
copy; nonzero levels are `1,2,3,4,5`.

| Corruption | Definition at severity s |
|---|---|
| Brightness decrease | multiply RGB by `1 − 0.13s` |
| Brightness increase | multiply RGB by `1 + 0.18s` |
| Contrast reduction | `(RGB − 127.5)(1 − 0.15s) + 127.5` |
| Gaussian blur | kernel `(2s+1)²`, standard deviation `0.5s` |
| Motion blur | horizontal uniform kernel width `2s+3` |
| Gaussian noise | independent noise with pixel standard deviation `5s` |
| Occlusion | black rectangle at 67% image height; width `0.07sW`, height `(0.04+0.025s)H` |
| Horizontal shift | right shift `round(0.015sW)` pixels, reflected boundary |
| Vertical shift | down shift `round(0.012sH)` pixels, reflected boundary |
| Shadow | fixed diagonal half-plane multiplied by `1 − 0.13s` |
| Color shift | red `+8s`, blue `−8s` |

Clip to uint8 [0,255] after corruption. Noise seeds are SHA-256-derived from
experiment seed, source row ID, corruption name, and severity, so conditions and
examples can be reproduced. These settings define this benchmark, not an external
standard. Monotonic severity need not produce monotonic model error.

For every pair, report MAE/RMSE and `delta_MAE = MAE_corrupt − MAE_clean` (similarly
RMSE). The optional aggregate is the unweighted arithmetic mean MAE across selected
**nonzero** pairs. It retains steering units and does not use an arbitrary 0–100
scale. Compare only matching condition lists; missing conditions are not zero.

Translated images are camera-view perturbations with fixed labels, not physically
resimulated trajectories. Extreme occlusion may remove information even a person
would need. Treat clean, corrupted, and closed-loop findings separately.

## E: MC dropout

Train with dropout (`uncertainty.yaml`: 0.25). At inference, leave the model in eval
mode except for dropout. For T=20 draws, calculate:

- Mean steering: `mu = sum(prediction_t) / T`
- Sample variance: `v = sum((prediction_t − mu)²) / (T − 1)`
- Primary uncertainty score: `u = sqrt(v)`

Use MC **mean** error when correlating with uncertainty. Report Pearson and
Spearman correlations, clean/corrupted histograms, and severity curves. The
predictions may vary with hardware, batch size, and library version; keep those
fixed. Constant inputs to a correlation produce `null`, not a fabricated zero.
MC dispersion is an epistemic-uncertainty heuristic, not a calibrated probability.

## F: high-error detection and selective prediction

Before testing, select an interpretable normalized-steering error threshold;
the provided default is `tau_error = 0.15`. Define `dangerous = |mu − label| > tau_error`.
The name is a research proxy, not proof of unsafe physical behavior.

Choose the uncertainty cutoff using **clean validation only**:
`tau_u = quantile(validation_u, 0.90)`. Then freeze it and flag test predictions
where `u > tau_u`. This quantile controls the nominal validation alert rate; it
is not optimization against test labels. Distribution shift can change that rate.

Report precision, recall, F1, ROC-AUC, trapezoidal PR-AUC, average precision,
dangerous/flagged counts, and class prevalence. ROC/PR discrimination metrics are
undefined for a single class and saved as `null`. Zero-denominator precision,
recall, and F1 are reported as 0 with an explicit note. PR-AUC and average precision
are different summaries and must not be interchanged.

For each requested coverage in `{1,.95,.90,.80,.70,.50}`, sort by uncertainty and
accept `max(1, floor(coverage × N))` predictions. Stable ties preserve source order;
they do not inspect errors. Report actual accepted fraction plus MAE/RMSE. These
curves rank the test cohort offline; they are not a cutoff calibrated for future
unseen driving. Nonmonotonic error versus coverage is a valid result.

Clean and per-corruption/per-severity metrics are exported. Pooled plots reuse
the same base images; their rows are not independent and must not be treated as
an enlarged independent sample for significance tests.

## G: error analysis and responsible reporting

Review highest error, highest uncertainty, high error/high uncertainty, high
error/low uncertainty, and low error/high uncertainty. Keep failed detector cases,
especially confidently wrong predictions. Do not select only attractive outcomes.

The present implementation does not estimate statistical confidence intervals.
Before strong claims, repeat across seeds and independent sessions/tracks; use
session/block-aware uncertainty estimates, not IID confidence intervals over
adjacent frames or repeated corrupted versions.

## Autonomous-run metrics

Annotations define attempted/completed/aborted laps. Unfinished attempts count
as incomplete. A lap is unassisted only if it completes without takeover or
failure annotations. No percentage of track distance is inferred from camera frames.

- Penalty autonomy: `100 × max(0, 1 − takeovers × penalty_seconds / duration)`.
- Time autonomy: `100 × (1 − annotated_manual_seconds / duration)`, clamped at zero.
- Steering smoothness: mean absolute successive steering change and RMS change
  per second; sampling rate/gaps affect interpretation.

Record the penalty (default 6 seconds) and evaluation duration. A lap finish with
a takeover differs from an unassisted finish. Off-track and collision event counts
may describe the same incident; do not relabel them as independent failures.
An event-free log has no autonomy percentage: zero recorded events do not prove
there were no failures or interventions. Even nonempty logs require a complete
annotation process. Saved camera frames may be rate-limited; inspect recording
metadata and do not assume one saved image per telemetry row.

## What is needed for the stronger early-warning claim

Keep online uncertainty and camera timestamps alongside human off-track/collision
events. Predefine a warning horizon and event-matching rules; measure recall,
false alerts per minute, and lead time on untouched driving runs. Human reaction
delay and inference latency must be included. This prospective analysis is
**pending**, and the offline detector does not substitute for it.
