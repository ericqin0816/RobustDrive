# Just play: Windows launcher

The desktop shortcut **Play RobustDrive** opens a small launcher. The simulator
is installed locally in `simulator/beta_simulator_windows/`; its binaries are not
committed or redistributed. The launcher needs the project's `.venv` installation.

1. Click **Play & record**, then **Training Mode** inside the simulator.
2. Click **Record** and select **Documents → RobustDrive Recordings**. This is a
   shortcut folder pointing to the project's `data/raw/`. Click **Record** again
   and confirm that the game says **RECORDING**. If the folder picker is elsewhere,
   the launcher's **Copy recording folder path** button gives you the exact path.
3. Drive with the arrow keys or W/A/S/D. Try several smooth laps with left and
   right turns. For a first useful recording, aim for around 5–10 minutes.
4. Press **R** to stop recording. Wait for saving to finish, then close the game.
5. Leave the launcher running. It checks the data, freezes a snapshot, trains the
   baseline/augmented/dropout models, and generates offline evaluations and figures.

Do not close the game before stopping and finishing the save: the simulator buffers
recordings in memory. Closing the launcher while it is busy minimizes it so that
the recording wait or training continues. Models train on CPU if no compatible
GPU is available; the duration depends on your recording and computer.

The automatic workflow excludes stationary frames at <=1 mph, preserves their
original raw files, and records the filtering policy/counts. It requires at least
600 moving frames and examples of both steering directions before training. This
is a basic input check, not a guarantee of sufficient or good training data.
Snapshots are separate copies under `data/processed/play_<fingerprint>/`, so later
recordings cannot change an existing experiment's test set. Synthetic sample data
is rejected by this workflow. Experiments are named `play_<fingerprint>_<model>`.

**Watch my model** becomes usable after successful training. It starts the steering
server and game; select **Autonomous Mode** yourself. Supervise the first attempt.
The full analysis of lap completion/takeovers still requires event annotations as
described in [the simulator guide](simulator.md). No success is inferred from simply
opening Autonomous Mode.

If a run stops, read its message and `local/play_runs/<recording>/training.log`.
**Train saved drive** retries a completed recording and reuses completed models.
No recordings or checkpoints are uploaded. Everything generated is ignored by Git.

To recreate the Windows shortcut after installing the official simulator and the
project environment, run `scripts/setup_windows.ps1` in PowerShell. The official
download is linked by the [Udacity simulator README](https://github.com/udacity/self-driving-car-sim)
under Term 1, Version 2, Windows. Normal Python CLI usage remains supported.
