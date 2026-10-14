# RobustDrive

**Failure Detection and Uncertainty in End-to-End Autonomous Driving**

## One-sentence summary

A PyTorch behavioral-cloning project that learns simulator steering from camera
images and studies whether model uncertainty can identify unreliable predictions.

## Problem

A steering network can produce a plausible-looking answer even when the scene
differs from its training examples. Low average prediction error does not reveal
every dangerous mistake or guarantee that a car will remain on the road.

## What I built

An end-to-end software pipeline for personal simulator recordings: data validation,
temporal holdouts, a CNN trained from scratch, recovery-camera augmentation,
checkpointed training, a simulator steering adapter, and timestamped drive logs.
The analysis tools add controlled image corruptions, MC dropout, high-error
detection, selective prediction, ablations, and visual failure review.

## Technical approach

Python, PyTorch, OpenCV, NumPy, Pandas, scikit-learn, Matplotlib, and pytest. A
NVIDIA-inspired CNN maps 66×200 RGB images to normalized steering. Training and
driving share preprocessing. A separate proportional controller sets throttle.
Uncertainty is the standard deviation of repeated dropout predictions.

## Research question

When does the learned steering model fail, and can uncertainty warn of unreliable
predictions? The initial detector uses high steering error as an offline proxy.
Actual advance warning of road departures remains a separate empirical question.

## Experiments

The first real experiments compare center-only training, recovery/augmentation,
corruption severity, and error rejection by uncertainty. Annotated autonomous laps
remain the next step.
Chronological splits, saved configurations, fingerprints, and labeled outputs
make the analysis auditable.

## Results

Trained three models on one personal recording with 9,459 usable rows. The
center-only baseline achieved normalized steering MAE 0.07891 and RMSE 0.12519
on 1,404 chronological test frames, outperforming the two augmented variants.
Its mean MAE across 55 corrupted conditions was 0.10934.

MC dropout did not reliably flag high-error predictions: clean ROC-AUC was 0.2837,
and rejecting the most uncertain half increased MAE. This negative result is
documented alongside label imbalance and limited data diversity. Real simulator
control was observed at approximately 10.85 mph; lap completion remains unmeasured.
See the [research report and evidence](research_report.md).

## What I learned

Use these as reflection prompts after personally reviewing and running the code:
why adjacent-frame leakage matters; why offline MAE differs from driving success;
how camera geometry affects augmentation labels; and why uncertainty can remain
low for incorrect predictions. Write the final reflection in your own words.

## Future work

Collect independent driving/recovery data, evaluate matched autonomous laps, analyze
confidently wrong cases, and test prospective warning lead time. Temporal models,
ensembles, calibration, and richer simulators come after the baseline is validated.

[Source repository](https://github.com/ericqin0816/RobustDrive)
