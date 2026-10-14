import csv
import time
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image

from robustdrive.data.preprocessing import Preprocessor
from robustdrive.models.uncertainty import mc_predict
from robustdrive.simulator.controller import SpeedController
from robustdrive.training.trainer import checkpoint_digest, load_model
from robustdrive.utils.logging import utc_now, write_json
from robustdrive.utils.seed import choose_device, seed_everything

EVENTS = ("lap_start", "lap_complete", "lap_abort", "takeover", "resume", "off_track", "collision")


class SteeringPolicy(Protocol):
    def predict(self, rgb: np.ndarray) -> dict[str, float | None]: ...


class NeuralSteering:
    def __init__(self, checkpoint: str | Path, mc_samples: int = 0, device: str = "auto"):
        self.device = choose_device(device)
        self.model, self.saved = load_model(checkpoint, self.device)
        config = self.saved["config"]
        seed_everything(config["experiment"]["seed"], config["training"]["cpu_threads"])
        self.preprocess = Preprocessor(**config["preprocessing"])
        if mc_samples and (mc_samples < 2 or config["model"]["dropout"] == 0):
            raise ValueError("Use >=2 MC samples and a checkpoint trained with dropout.")
        self.mc_samples = mc_samples
        self.provenance = {
            "checkpoint_sha256": checkpoint_digest(checkpoint),
            "training_data_kind": config["data"]["kind"],
            "preprocessing": config["preprocessing"],
            "mc_samples": mc_samples,
        }

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray) -> dict[str, float | None]:
        tensor = self.preprocess(rgb).unsqueeze(0).to(self.device)
        if self.mc_samples:
            result = mc_predict(self.model, tensor, self.mc_samples)
            steering, std = result["mean"].item(), result["std"].item()
        else:
            steering, std = self.model(tensor).item(), None
        if not np.isfinite(steering) or (std is not None and not np.isfinite(std)):
            raise ValueError("Nonfinite model prediction.")
        return {
            "prediction": steering,
            "steering": float(np.clip(steering, -1, 1)),
            "uncertainty": std,
        }


class DriveSession:
    """One append-only run, timestamped telemetry and explicit human event annotations."""

    def __init__(
        self,
        policy: SteeringPolicy,
        output: str | Path,
        controller: SpeedController,
        metadata: dict,
        uncertainty_threshold: float | None = None,
        record_frames: bool = False,
        record_fps: float = 5.0,
        max_recorded_frames: int = 10000,
    ):
        if not np.isfinite(record_fps) or record_fps <= 0 or max_recorded_frames < 1:
            raise ValueError("Frame recording needs positive fps and a positive frame limit.")
        self.policy, self.controller = policy, controller
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "telemetry.csv").exists():
            raise FileExistsError("Drive log exists; use a new --output for every run.")
        self.started = time.monotonic()
        self.manual, self.lap_open = False, False
        self.threshold, self.record_frames = uncertainty_threshold, record_frames
        self.frame = 0
        self.record_fps, self.max_recorded_frames = record_fps, max_recorded_frames
        self.recorded_frames = 0
        self.last_recorded_at = float("-inf")
        self.telemetry_file = (self.output / "telemetry.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self.events_file = (self.output / "events.csv").open("w", newline="", encoding="utf-8")
        self.telemetry = csv.DictWriter(
            self.telemetry_file,
            fieldnames=[
                "timestamp",
                "elapsed_seconds",
                "frame",
                "prediction",
                "steering",
                "speed",
                "throttle",
                "uncertainty",
                "risk_flag",
                "manual",
                "inference_ms",
                "image",
            ],
        )
        self.events = csv.DictWriter(
            self.events_file, fieldnames=["timestamp", "elapsed_seconds", "event", "note"]
        )
        self.telemetry.writeheader()
        self.events.writeheader()
        self.telemetry_file.flush()
        self.events_file.flush()
        write_json(
            self.output / "drive_metadata.json",
            {
                "timestamp": utc_now(),
                **metadata,
                "target_speed_mph": controller.target_speed,
                "kp": controller.kp,
                "max_throttle": controller.max_throttle,
                "uncertainty_threshold": uncertainty_threshold,
                "record_frames": record_frames,
                "record_fps_limit": record_fps,
                "max_recorded_frames": max_recorded_frames,
                "annotation_policy": "Human events; no automatic lane/collision/lap ground truth.",
            },
        )

    def event(self, event: str, note: str = "") -> None:
        if event not in EVENTS:
            raise ValueError(f"Choose an event from {EVENTS}.")
        if event == "lap_start":
            if self.lap_open:
                raise ValueError("A lap is already open; mark lap_complete or lap_abort first.")
            self.lap_open = True
        elif event in {"lap_complete", "lap_abort"}:
            if not self.lap_open:
                raise ValueError("Mark lap_start before ending a lap.")
            self.lap_open = False
        if event == "takeover":
            if self.manual:
                raise ValueError("Already in takeover; mark resume first.")
            self.manual = True
        elif event == "resume":
            if not self.manual:
                raise ValueError("No active takeover to resume.")
            self.manual = False
        self.events.writerow(
            {
                "timestamp": utc_now(),
                "elapsed_seconds": time.monotonic() - self.started,
                "event": event,
                "note": note,
            }
        )
        self.events_file.flush()

    def step(self, rgb: np.ndarray, speed: float) -> dict:
        began = time.perf_counter()
        result = self.policy.predict(rgb)
        inference_ms = (time.perf_counter() - began) * 1000
        throttle = 0.0 if self.manual else self.controller(speed)
        std = result["uncertainty"]
        flagged = self.threshold is not None and std is not None and std > self.threshold
        filename = ""
        now = time.monotonic()
        if (
            self.record_frames
            and self.recorded_frames < self.max_recorded_frames
            and now - self.last_recorded_at >= 1 / self.record_fps
        ):
            path = self.output / "frames" / f"{self.frame:08d}.jpg"
            path.parent.mkdir(exist_ok=True)
            Image.fromarray(rgb).save(path)
            filename = str(path.relative_to(self.output))
            self.last_recorded_at = now
            self.recorded_frames += 1
        row = {
            "timestamp": utc_now(),
            "elapsed_seconds": time.monotonic() - self.started,
            "frame": self.frame,
            **result,
            "speed": speed,
            "throttle": throttle,
            "risk_flag": bool(flagged),
            "manual": self.manual,
            "inference_ms": inference_ms,
            "image": filename,
        }
        self.telemetry.writerow(row)
        self.telemetry_file.flush()
        self.frame += 1
        return row

    def close(self) -> None:
        write_json(
            self.output / "session_end.json",
            {
                "timestamp": utc_now(),
                "elapsed_seconds": time.monotonic() - self.started,
                "recorded_frames": self.recorded_frames,
                "frame_limit_reached": self.recorded_frames >= self.max_recorded_frames,
            },
        )
        self.telemetry_file.close()
        self.events_file.close()
