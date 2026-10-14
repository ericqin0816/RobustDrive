# Software verification

**Verification date: October 13, 2026 (America/Chicago).**

This document records software checks, not research results. **Every real-data
experiment and real autonomous simulator lap remains PENDING REAL DATA.**

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
  Generated a separate synthetic comparison table. Real tables remain pending.
- Ran the test suite, including exact CPU resume reproducibility, online/offline
  preprocessing equality, data-integrity rejection, metrics, and WebSocket tests.
- Ran all major command-line `--help` entry points and lint checks.

The final local suite passed **40 tests**. Package/PyTorch/torchvision/OpenCV
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

- Actual Unity simulator handshake/control on this machine.
- Training/evaluation on personally collected real simulator images.
- Autonomous lap completion, interventions, or road-departure warning lead time.
- Generalization across tracks, sessions, seeds, hardware, or simulator forks.

The network tests emulate Udacity packets; they are not a substitute for driving
the actual simulator. The offline error detector does not establish collision
prediction. Follow the collection and simulator guides to perform that next phase.
