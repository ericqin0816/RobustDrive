# RobustDrive: Failure Detection and Uncertainty in End-to-End Autonomous Driving

**Status: first personal simulator recording evaluated, October 13, 2026
(America/Chicago). Annotated autonomous laps and advance warnings remain unmeasured.**

## Abstract

Three PyTorch steering CNNs were trained from scratch on one personal Udacity
simulator recording. On 1,404 held-out center frames, the center-only baseline
achieved MAE 0.07891 and RMSE 0.12519; recovery/visual augmentation and dropout did
not improve either clean metric. Across 55 nonzero corruption conditions, the
baseline also had the lowest aggregate MAE of these three models. MC dropout
failed as a high-error detector: clean ROC-AUC was 0.2837, and retaining the least
uncertain half increased MAE. A real simulator trial verified camera-to-steering
control at approximately 10.85 mph, without sufficient annotations to establish
lap completion. These measurements describe one recording and seed.

## 1. Introduction

Behavioral cloning learns steering from demonstrations, but mistakes change the
observations a policy encounters. Offline average error may conceal difficult
turns and says little about recovery after drift. This project asks whether
prediction dispersion offers useful information about such errors.

## 2. Background

The architecture is inspired by Bojarski et al. (2016), and the collection workflow
by Udacity's behavioral-cloning project. Gal and Ghahramani (2016) motivate dropout
as an approximate uncertainty method. RobustDrive independently implements the
pipeline; behavioral cloning and MC dropout are established prior work.

## 3. Research question

Does uncertainty identify high steering errors under visual changes? Does rejecting
uncertain predictions reduce accepted error? The stronger question of warnings
before physical road departure requires prospective, annotated simulator trials
and is not answered by the offline error proxy.

## 4. Methodology

The [protocol](../docs/experimental_protocol.md) defines preprocessing, chronological
holdouts, corruptions, metrics, and thresholds. All three runs used seed 42, the
same split, MSE/AdamW, a 30-epoch budget, validation-based checkpoint selection,
and patience 6. The baseline stopped after 14 epochs (best epoch 8); augmented
and dropout runs stopped after 7 (best epoch 1). Augmented runs expand each source
row to three cameras, so these comparisons are not compute matched and do not
isolate individual augmentation components.

Training used CPU PyTorch 2.14.1 and source commit
`8566dbe5ef3950ee8a30d445bb9713fe1b64c97e`, with a clean working tree. A
[frozen evidence manifest](results/first_recording/manifest.json) records source
artifact hashes, dataset fingerprints, and checkpoint hashes. Published JSON and
CSV files are copies of measured artifacts; the dataset and model weights remain
local. Models and cutoffs were not retuned after inspecting these test results.

## 5. Dataset

The personal lake-track recording contains 9,463 timestamp rows and 28,389 camera
images from approximately 11 minutes of keyboard driving. The launcher excluded
four stationary rows, preserving 9,459 rows and 28,377 images in an immutable
snapshot. Inspection found no missing or corrupt images in that snapshot.

| Partition | Source rows | Center-only samples | Three-camera training samples |
|---|---:|---:|---:|
| Training | 6,606 | 6,606 | 19,818 |
| Validation | 1,389 | 1,389 | Not expanded |
| Test | 1,404 | 1,404 | Not expanded |
| Discarded boundary gaps | 60 | — | — |

The prepared dataset fingerprint is
`4dba1907e0e2a18b162f211e23f0acd98c032d0f2cdccd20d3e1a925f7209518`.
Usable-row speed averaged 29.999 mph. Labels are heavily imbalanced: 79.62% have
absolute steering below 0.05. Track and controls come from the observed user
workflow; direction and recovery quality were not independently annotated.
Repeated scenery can occur across chronological partitions.

## 6. Behavioral-cloning model

Five convolution layers and dense layers of 100, 50, 10, and 1 units map normalized
3×66×200 RGB tensors to steering. The network has 252,219 trainable parameters.
The baseline uses center images without augmentation. The augmented variant adds
side-camera correction, flips, and visual transforms. The uncertainty variant adds
dropout 0.25. A separate proportional throttle controller sets simulator speed;
the network does not imitate the demonstrator's throttle.

## 7. Robustness benchmark

Eleven corruptions at severities 0–5 were applied to the same held-out frames.
The reported aggregate averages MAE over the 55 nonzero condition pairs, equally
weighted. Severity zero is the clean comparator. Translations retain original
labels and measure camera perturbation sensitivity rather than verified physical
recovery. Per-condition measurements are included in each model's
`robustness.csv` under [the evidence folder](results/first_recording).

## 8. Uncertainty estimation

The dropout model uses 20 stochastic passes with only dropout layers in training
mode. Steering is their mean; uncertainty is sample standard deviation. A cutoff
of **0.04488262** was the clean validation 90th percentile. The original high-error
definition, absolute MC-mean error greater than **0.15**, was retained.

## 9. Evaluation metrics

Errors use normalized simulator steering, not degrees. Clean regression reports
MAE, RMSE, and R². Detection reports precision, recall, F1, ROC-AUC, PR-AUC, and
average precision. Coverage accepts the lowest-uncertainty predictions. Corrupted
copies are repeated measurements on the same frames, not independent samples.
Lap success and advance-warning metrics require separate event evidence.

## 10. Results

| Model | Clean MAE | Clean RMSE | R² | Mean corrupted MAE | Mean ΔMAE |
|---|---:|---:|---:|---:|---:|
| Center-only baseline | 0.07891 | 0.12519 | 0.2493 | 0.10934 | +0.03042 |
| Recovery + visual augmentation | 0.10306 | 0.14813 | −0.0510 | 0.11065 | +0.00759 |
| Augmentation + dropout, deterministic | 0.10000 | 0.14616 | −0.0233 | 0.11675 | +0.01675 |

The augmented model degraded less from its own clean score but did not beat the
baseline's absolute corrupted MAE. Smaller degradation alone is not an overall
robustness improvement. The dropout row above disables dropout at evaluation;
20-pass MC inference instead achieved clean MAE 0.09783 and RMSE 0.14484.

| MC-dropout clean high-error detector | Measured value |
|---|---:|
| High-error frames / test frames | 250 / 1,404 |
| Flagged frames | 178 |
| Precision / recall / F1 | 0.06742 / 0.04800 / 0.05607 |
| ROC-AUC / trapezoidal PR-AUC / average precision | 0.28368 / 0.11814 / 0.11899 |
| Pearson / Spearman uncertainty–error correlation | −0.33685 / −0.60300 |

| Actual accepted clean coverage | Accepted frames | MAE |
|---|---:|---:|
| 100% | 1,404 | 0.09783 |
| 94.943% | 1,333 | 0.10126 |
| 89.957% | 1,263 | 0.10410 |
| 79.986% | 1,123 | 0.11014 |
| 69.943% | 982 | 0.11700 |
| 50% | 702 | 0.13040 |

![First-session offline comparison and rejection curve](results/first_recording/overview.png)

The first live baseline trial produced **65,503 telemetry responses over 524.954
seconds**. Median speed was **10.8495 mph** for a **12 mph** target; inference median
was **1.616 ms**, with 95th percentile **2.030 ms**, excluding network/render time.
Selected saved camera frames and a live Autonomous Mode view were inspected.
There were **zero event annotations**, so lap completion, absence of interventions,
and absence of failures are **not established**. Repeated telemetry frames do not
count as independent observations. The game was closed after review; the old
launcher terminated its server without a session-end file. The published
[drive observation](results/first_recording/drive_observation.json) preserves that
limitation and hashes of the completed logs. No missing endpoint was invented.

## 11. Failure analysis

Exploratory diagnostics after the primary evaluation show baseline MAE of 0.05180
on 1,134 nearly straight test frames, 0.18771 on 233 left-turn frames, and 0.22482
on 37 right-turn frames (threshold magnitude 0.05). A parameter-free always-zero
steering reference achieved MAE **0.05502** but worse RMSE **0.14903** than the
learned baseline. This exposes the effect of imbalance on MAE; it does not imply
that always driving straight is useful. See [diagnostics](results/first_recording/diagnostics.json).

The reviewed confidently-wrong grid includes test source row 8357 under Gaussian
blur severity 5: target +1.000, MC mean −0.07544, error 1.07544, uncertainty 0.03196
(below the alert cutoff). Other examples include strong turns under shadows with
low uncertainty. These image/label comparisons support the negative detector
result; they do not establish that a collision occurred. The full frame grids and
per-frame prediction records are retained locally with the experiments.

## 12. Discussion

The baseline was the strongest trained model under this configuration. The
augmentation bundle and correction of 0.2 may be poorly matched to this recording;
the experiment does not isolate that explanation. Early best epochs in both
augmented runs and the strong straight-driving imbalance merit further study.
Uncertainty ranked clean errors in the opposite direction to the intended
detector. Its threshold was not inverted or retuned on the test labels to present
an improved result. More training is not assumed to fix these issues.

## 13. Limitations

One recording, track, seed, and mostly similar driving direction; correlated
frames and repeated scenery; keyboard demonstration labels; approximate camera
corrections; unequal optimizer-step counts; no independent session holdout or
statistical confidence intervals; uncalibrated dropout; no complete live-event
audit. Offline accuracy cannot prove reliable driving at the training speed.
Nothing here validates real-world autonomous operation.

## 14. Future work

Collect an independent recording with more right turns and controlled recovery,
preserve a new untouched test session, and perform component ablations and seed
repeats. Compare supervised, annotated simulator trials at matched speed before
testing higher targets. Advance-warning experiments require a predeclared warning
horizon and complete event labels. Temporal networks, ensembles, and richer
simulators remain possible later work, not implemented findings.

## 15. Conclusion

Personal simulator training and real camera-to-steering control now work. This
first offline study does not support improved accuracy from the current
augmentation bundle or reliable failure detection from its MC-dropout score.
Those negative findings and the missing lap annotations are part of the result.

## 16. References

1. Bojarski et al. (2016), [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316).
2. Udacity, [Behavioral Cloning project](https://github.com/udacity/CarND-Behavioral-Cloning-P3).
3. Udacity, [Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim).
4. Gal and Ghahramani (2016), [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142).
5. [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html).
