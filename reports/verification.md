# Software verification

**Verification date: October 13, 2026 (America/Chicago).**

This document records software checks. Real-data measurements from the first
personal recording are in [the research report](research_report.md). Annotated
autonomous lap completion and early-warning claims remain pending.

## Executed locally

- Created an isolated Python 3.12 virtual environment and installed the package.
- Generated 120 **synthetic** timestamp rows with three procedural camera images
  each. The generator marks every picture and writes synthetic provenance.
- Trained the dropout/augmentation smoke model for two epochs on CPU and saved
  best/latest checkpoints, history, configuration, metadata, and split membership.
- Ran clean test evaluation; all 11 corruptions at severities 0–5; and 20-pass MC
  dropout on clean data plus darkening, Gaussian blur/noise, and shadow at levels 1–5.
- Generated regression, robustness, uncertainty, detector, coverage, and failure
  figures. Visually inspected coverage and confidently-wrong image grids; the
  synthetic provenance footer is visible.
- Trained and evaluated all five two-epoch synthetic ablation configurations.
  Generated a separate synthetic comparison table; it is not research evidence.
- Ran the test suite, including exact CPU resume reproducibility, online/offline
  preprocessing equality, data-integrity rejection, metrics, and WebSocket tests.
- Ran all major command-line `--help` entry points and lint checks.

The original local suite passed **40 tests**. Package/PyTorch/torchvision/OpenCV
imports and `pip check` passed. Every exploratory notebook code cell executed;
the committed notebook intentionally retains empty outputs. Local software was
Python 3.12.10, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, and OpenCV 4.14.0.
No CUDA device was available. A full repeat of `smoke.py --ablation` reused the
completed models, re-ran all offline analyses, and succeeded. `make_figures.py`
regenerated the saved figures successfully. All local Markdown links resolved.

The actual `drive.py` CLI was also started on a separate loopback test port with
the synthetic checkpoint. Six generated camera messages passed through the model
with 20 dropout passes and produced bounded steering/throttle replies. The
`evaluate_drive.py` CLI successfully processed those **synthetic protocol logs**.
No vehicle was driven and no real lap was completed during this test.

## GitHub delivery

The project was pushed to the public [RobustDrive repository](https://github.com/ericqin0816/RobustDrive).
The repository description, topics, files, README, and both rendered Mermaid
diagrams were inspected in the signed-in browser. The workflow tests Linux/Python
3.11 and Windows/Python 3.12 using CPU PyTorch. Current CI status is available on
the [Actions page](https://github.com/ericqin0816/RobustDrive/actions/workflows/tests.yml)
and README badge. CI never downloads the simulator or real datasets.

## Windows play-and-learn setup

The official Term 1 Version 2 Windows simulator was downloaded from the URL linked
in Udacity's README, extracted locally, and opened successfully to its menu. A
desktop shortcut and a Documents-folder junction to `data/raw` were created. The
launcher was opened and visually checked while attached to the running simulator.
It waits for the game to close before inspecting, snapshotting, training, and
evaluating the user's recording. No real training was started without a recording.

The expanded suite passed **44 tests**, including snapshot preservation/reuse,
stationary-frame filtering, synthetic-data rejection, and changed-snapshot rejection.
After final launcher refinements, the four recording tests and lint passed again.
Native automated clicks were rejected by Windows (`SendInput`, error 87), so the
user must select Training Mode and the recording folder in the game's menus.
At this initial setup stage, Unity network steering remained unverified. The
subsequent real trial below established live control, with lap performance still
unmeasured.

## First recording and live trial

The launcher completed training and offline analysis of 9,459 usable personal
simulator rows. All three saved checkpoints and their report hashes were checked.
A live Unity Autonomous Mode trial generated 65,503 telemetry rows over 524.954
seconds. Saved camera samples and the running game were visually inspected. The
model produced steering and held speed near 10.85 mph for a 12 mph target.
There were no event annotations; this is not a verified lap-success result.

The viewer now exposes 12–30 mph targets, rate-limits saved images to 5 fps with
a 10,000-image ceiling, and requests graceful server shutdown when the game closes.
Unannotated logs no longer imply 100% autonomy. The updated local suite passed
**48 tests**, including higher speed targets, recording limits with complete
telemetry, the missing-annotation case, and graceful shutdown with an open
WebSocket. Lint and all CLI help checks passed. The updated launcher was reopened
and its visible speed selector and enabled Watch button were checked.

## Reproduce

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src scripts tests
python scripts/check_cli.py
python scripts/smoke.py --ablation
python scripts/make_figures.py
```

Generated data, checkpoints, and numerical outputs are intentionally ignored by
Git. They are reproducible with the commands above. Do not use synthetic metrics
in the research report or resume.

## Not yet verified

- Autonomous lap completion, interventions, or road-departure warning lead time.
- Actual driving performance at 20–30 mph.
- Generalization across tracks, sessions, seeds, hardware, or simulator forks.

The network tests emulate Udacity packets; the separate live observation is
documented above. The offline error detector does not establish collision
prediction. Follow the collection and simulator guides to perform that next phase.
