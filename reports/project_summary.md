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

Planned real experiments compare center-only training, recovery/augmentation,
autonomous laps, corruption severity, and error rejection by uncertainty.
Chronological splits, saved configurations, fingerprints, and labeled outputs
make the analysis auditable.

## Results

**PENDING REAL DATA.** Software tests and synthetic smoke training/evaluation have
run. No real simulator success rate, steering metric, or uncertainty finding is
claimed. Replace this paragraph only with measurements traceable to saved runs.

## What I learned

Use these as reflection prompts after personally reviewing and running the code:
why adjacent-frame leakage matters; why offline MAE differs from driving success;
how camera geometry affects augmentation labels; and why uncertainty can remain
low for incorrect predictions. Write the final reflection in your own words.

## Future work

Collect personal driving/recovery data, evaluate matched autonomous laps, analyze
confidently wrong cases, and test prospective warning lead time. Temporal models,
ensembles, calibration, and richer simulators come after the baseline is validated.

[Source repository](https://github.com/ericqin0816/RobustDrive)
