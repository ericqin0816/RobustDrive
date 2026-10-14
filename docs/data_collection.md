# Collect your first real simulator dataset

**Collection guidance.** Suggested quantities below are targets, not measured
counts. The first recording's actual counts are in the
[research report](../reports/research_report.md).

1. Get the official **Term 1 Version 2** simulator using [simulator setup](simulator.md).
2. Create an empty recording folder. The default destination is this repository's
   `data/raw/` folder. Do not mix synthetic data with recordings.
3. Open the lake/easier track in **Training Mode**. Practice steering first.
4. Click **Record**, select the folder, and collect a short sign-check segment:
   steer left, then right slowly. Stop recording and wait for saving to finish.
5. Open the CSV and images. Left steering should be negative, right positive;
   compare the three camera viewpoints from one timestamp. Keep this calibration
   recording separate if it includes poor driving. If a fork reverses conventions,
   adapt labels/networking deliberately before training.
6. Start the main dataset in a clean folder. A reasonable first attempt is
   **4–6 normal laps plus 1–2 lap equivalents of short recovery segments**, roughly
   10–20 minutes depending on speed. Review coverage before collecting more.
7. Record smooth normal driving with curves and straights. If the build supports
   opposite-direction driving, include a portion in the other direction. Keep
   a local note of track, direction, controls, and recording conditions.
8. **Left-edge recovery:** with recording off, position slightly left of the
   normal path; record while gently returning to center; stop before repositioning.
   Repeat at several straights/curves. Do not label the intentional departure
   maneuver as desirable driving.
9. **Right-edge recovery:** repeat from the right edge. Avoid logging repeated
   crashes or sharp oscillations as expert behavior.
10. Mix normal and recovery segments throughout collection, not only at the end.
    Later validation/test sections also need turns. Keep rows in session order.
    If separate folders are generated, preserve unique image names and concatenate
    CSVs in recording order with only one header. Back up each source first.
11. Stop recording and wait for saving. Move both `driving_log.csv` and `IMG/`
    together into `data/raw/`.
12. From the repository root, with the virtual environment active, run:

    ```bash
    python scripts/inspect_dataset.py --data data/raw
    ```

13. Read `reports/tables/dataset_inspection.json` and view
    `reports/figures/dataset/steering_distribution.png`. Fix missing/corrupt images.
    Collect underrepresented turns or curves if needed.
14. Freeze/back up the data and train:

    ```bash
    python scripts/train.py --config configs/baseline.yaml
    python scripts/evaluate.py --checkpoint experiments/results/baseline/best.pt
    ```

15. Attempt a slow supervised autonomous run using [the simulator guide](simulator.md).
    Log failures honestly. Train augmented/dropout variants on the same frozen
    dataset to isolate configuration effects.

## Why center-line driving alone is insufficient

Straight roads create many nearly identical near-zero steering samples. A model
can minimize average error while failing to turn or recover from its own mistakes.
Collect curves, both steering directions, and recoveries. A balanced histogram
alone does not establish coverage of states visited by the learned policy.

Do not shuffle the CSV before splitting. The default first 70%/next 15%/last 15%
chronological blocks discard 30 boundary frames per boundary. Camera expansion
happens after splitting. Do not repeatedly tune against the held-out test set.

## Sign convention

The upstream [InputManager](https://github.com/udacity/self-driving-car-sim/blob/master/ProjectSettings/InputManager.asset)
maps left/right to negative/positive horizontal input. The
[CarController](https://github.com/udacity/self-driving-car-sim/blob/master/Assets/Standard%20Assets/Vehicles/Car/Scripts/CarController.cs)
records steering normalized by maximum wheel angle. Therefore a left-offset
camera gets a positive recovery correction and a right camera a negative one.
Default 0.2 is approximate, not a measured optimal target. Flipping negates it.

A road shifted right in an image approximates the camera being left, so training
translation adds `dx × 0.002`. True calibration would need camera geometry and
vehicle dynamics. Optional small rotations represent camera roll, not yaw;
they leave labels unchanged and are disabled by default.
