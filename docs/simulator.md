# Simulator setup and autonomous evaluation

## Obtain the official simulator

Open [Udacity's simulator repository](https://github.com/udacity/self-driving-car-sim).
In **Available Game Builds → Term 1 → Version 2 (2/07/17)**, download your OS's
zip. Extract outside this repository and run the application, keeping its data
folder beside the executable. Do not use the unrelated Term 2/3 simulator.
RobustDrive does not redistribute simulator binaries.

The upstream repository is archived. Old prebuilt apps may not run on every
current OS. The upstream README also explains Unity 2019.4/2020.3 source branches.
Follow official build instructions when necessary. Without a compatible simulator,
offline software checks still run with `python scripts/smoke.py`.

## Your first autonomous trial

1. Collect data with [the collection guide](data_collection.md).
2. Run `python scripts/train.py --config configs/baseline.yaml`.
3. Open two terminals at the repository root, with the virtual environment active.
4. In terminal A:

   ```bash
   python scripts/drive.py --checkpoint experiments/results/baseline/best.pt --output experiments/results/drive_baseline_01 --target-speed 12 --record-frames
   ```

5. Open the simulator and select the same track in **Autonomous Mode**. The server
   listens on `127.0.0.1:4567`; allow local communication if your OS asks.
6. In terminal B, run `python scripts/mark_event.py lap_start` at a chosen start
   line. Use a consistent starting pose/speed between model comparisons.
7. When intervention is needed, physically use the simulator's manual controls
   and timestamp `python scripts/mark_event.py takeover --note "reason"` promptly.
   The marker pauses network steering responses. The original build yields empty
   telemetry during W/S input; verify manual controls at low speed on your build.
   The marker itself does not press keys or apply brakes.
8. Mark `off_track` or `collision` when observed. These are human annotations.
9. After repositioning, mark `resume` to restore model control. To restart a lap,
   mark `lap_abort` followed by a fresh `lap_start`.
10. At the finish, mark `lap_complete`. Before another lap, mark `lap_start` again.
    A lap completed with intervention is not classified as unassisted.
11. Stop terminal A with Ctrl+C, then run:

    ```bash
    python scripts/evaluate_drive.py --log-dir experiments/results/drive_baseline_01 --takeover-penalty 6
    ```

12. Preserve `telemetry.csv`, `events.csv`, `drive_metadata.json`, `session_end.json`,
    and `autonomy.json`. `--record-frames` adds a camera sequence for review. Record
    a screen video separately. Use a new output directory for every trial.

As a starting recommendation, attempt 3–5 laps per model with matched track,
speed, start, and intervention rules. Include abandoned and unfinished laps.
This is a suggested protocol, not a completed experiment.

## Controller and uncertainty

The model predicts steering only. Throttle is
`clip(kp × (target_speed − measured_speed), 0, max_throttle)`, defaulting to 12 mph,
gain 0.08, and maximum 0.3. It coasts above the target; it does not actively brake.

For a dropout model, add `--mc-samples 20`. Add
`--uncertainty-report experiments/results/uncertainty/uncertainty.json` for flags
using the validation cutoff; the checkpoint and pass count must match the report.
Flags are observational. Check logged inference latency before interpreting
closed-loop performance.

## Networking compatibility

Upstream [CommandServer](https://github.com/udacity/self-driving-car-sim/blob/master/Assets/1_SelfDrivingCar/Scripts/CommandServer.cs)
sends `telemetry` with a base64 image and speed and accepts `steer` with string
`steering_angle`/`throttle` values. RobustDrive uses maintained `aiohttp` to handle
this narrow Socket.IO text-over-WebSocket subset: connection handshake, EIO=3/4
query variants, ping/pong, and default-namespace events. The legacy Unity client
can advertise EIO=4 while using older packet behavior. This is not a general
Socket.IO server; polling, binary attachments, and namespaces are unsupported.

Synthetic tests cover the network exchange. **Actual Unity connection and laps
remain PENDING MANUAL SIMULATOR TESTING.** Fork-specific protocol changes belong
in `src/robustdrive/simulator/udacity.py`; training need not change.

## Troubleshooting

- Connection refused: start the server first, free port 4567, and confirm the
  simulator connects to `ws://127.0.0.1:4567/socket.io/?EIO=4&transport=websocket`.
- Wrong steering direction: stop and repeat the manual sign check.
- Missing data: collect it first; a smoke checkpoint cannot substitute for it.
- Wrong crop: inspect real images before training; never change only online code.
- Dataset mismatch: restore frozen originals or train a new named experiment.
- Slow MC inference: reduce speed or passes and evaluate a matching report.
- Empty failure grid: inspect the corresponding CSV; the category may be empty.

`--allow-synthetic` exists solely for protocol testing; those logs retain synthetic
provenance and cannot substantiate autonomous-driving claims.
