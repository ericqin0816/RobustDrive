"""Metrics from recorded simulator telemetry and explicit human lap/event labels."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from robustdrive.utils.logging import write_json


def autonomy_metrics(folder: str | Path, penalty_seconds: float = 6.0) -> dict:
    if penalty_seconds < 0:
        raise ValueError("Takeover penalty cannot be negative.")
    folder = Path(folder)
    metadata = json.loads((folder / "drive_metadata.json").read_text(encoding="utf-8"))
    telemetry = pd.read_csv(folder / "telemetry.csv")
    events = pd.read_csv(folder / "events.csv")
    if len(telemetry) < 2:
        raise ValueError("At least two telemetry frames are required.")
    times = telemetry.elapsed_seconds.to_numpy()
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("Telemetry timestamps must be finite and monotonic.")
    takeovers = events[events.event == "takeover"].elapsed_seconds.to_numpy()
    laps, start = [], None
    for event in events.itertuples():
        if event.event == "lap_start":
            start = float(event.elapsed_seconds)
        elif event.event in {"lap_complete", "lap_abort"} and start is not None:
            end = float(event.elapsed_seconds)
            lap_events = events[(events.elapsed_seconds >= start) & (events.elapsed_seconds <= end)]
            duration = end - start
            interventions = int((lap_events.event == "takeover").sum())
            failures = int(lap_events.event.isin(["off_track", "collision"]).sum())
            laps.append({"lap_seconds": duration, "completed": event.event == "lap_complete",
                         "unassisted": event.event == "lap_complete" and interventions == 0 and failures == 0,
                         "interventions": interventions, "failure_events": failures,
                         "penalty_autonomy_percent": max(0, 1 - interventions * penalty_seconds / duration) * 100 if duration > 0 else None})
            start = None
    duration = float(times[-1] - times[0])
    manual_seconds, manual_start = 0.0, None
    for event in events.itertuples():
        if event.event == "takeover":
            manual_start = max(float(times[0]), float(event.elapsed_seconds))
        elif event.event == "resume" and manual_start is not None:
            manual_seconds += max(0, min(float(times[-1]), float(event.elapsed_seconds)) - manual_start)
            manual_start = None
    if manual_start is not None:
        manual_seconds += max(0, float(times[-1]) - manual_start)
    changes = np.diff(telemetry.steering.to_numpy())
    dt = np.diff(times)
    rates = changes[dt > 0] / dt[dt > 0]
    terminal_laps = len(laps)
    attempts = terminal_laps + int(start is not None)
    actual = metadata.get("source") == "udacity-live" and metadata.get("training_data_kind") == "real"
    result = {"status": "RECORDED SIMULATOR RUN — HUMAN EVENT ANNOTATIONS" if actual else "SYNTHETIC/TEST LOG ONLY — PENDING REAL DATA",
              "telemetry_frames": len(telemetry), "duration_seconds": duration,
              "interventions": len(takeovers), "time_between_interventions_seconds": np.diff(takeovers).tolist(),
              "mean_time_between_interventions_seconds": float(np.diff(takeovers).mean()) if len(takeovers) > 1 else None,
              "off_track_events": int((events.event == "off_track").sum()),
              "collision_events": int((events.event == "collision").sum()),
              "failure_events": int(events.event.isin(["off_track", "collision"]).sum()),
              "lap_attempts": attempts, "terminal_laps": terminal_laps, "unfinished_laps": int(start is not None),
              "lap_completion_percent": 100 * sum(l["completed"] for l in laps) / attempts if attempts else None,
              "unassisted_lap_completion_percent": 100 * sum(l["unassisted"] for l in laps) / attempts if attempts else None,
              "penalty_seconds": penalty_seconds,
              "penalty_autonomy_percent": max(0, 1 - len(takeovers) * penalty_seconds / duration) * 100 if duration > 0 else None,
              "time_autonomy_percent": 100 * max(0, 1 - manual_seconds / duration) if duration > 0 else None,
              "mean_absolute_steering_change": float(abs(changes).mean()),
              "rms_steering_rate_per_second": float(np.sqrt(np.mean(rates ** 2))) if len(rates) else None,
              "laps": laps,
              "limitations": "Events are manual annotations. Multiple failure events may describe one incident. Unfinished laps count as incomplete; fractional distance completion is unavailable."}
    write_json(folder / "autonomy.json", result)
    return result
