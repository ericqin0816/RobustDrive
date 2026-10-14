"""CSV parsing, auditable image validation, and camera expansion after splitting."""

import csv
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from robustdrive.data.augmentation import Augmenter, camera_steering
from robustdrive.data.preprocessing import Preprocessor, read_rgb
from robustdrive.utils.paths import resolve_image

CAMERAS = ("center", "left", "right")
FIELDS = (*CAMERAS, "steering", "throttle", "brake", "speed")


@dataclass
class Record:
    row_id: int
    steering: float
    images: dict[str, Path]
    throttle: float | None = None
    brake: float | None = None
    speed: float | None = None


@dataclass
class DrivingLog:
    records: list[Record]
    inspection: dict
    fingerprint: str
    kind: str


def load_log(csv_path: str | Path, image_root: str | Path | None = None,
             kind: str = "real", invalid_policy: str = "skip") -> DrivingLog:
    """Read header/headerless logs; report every excluded row/image, never fake pixels."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Collect simulator data first; "
                                "use configs/smoke.yaml for synthetic validation.")
    marker = path.parent / "dataset_metadata.json"
    if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("kind") == "synthetic":
        kind = "synthetic"  # A config cannot promote generated data into a research result.
    if kind not in {"real", "synthetic"} or invalid_policy not in {"skip", "error"}:
        raise ValueError("Invalid dataset kind or invalid_policy.")
    digest = hashlib.sha256(path.read_bytes())
    root = Path(image_root) if image_root else None
    report = {"rows": 0, "valid_label_rows": 0, "invalid_rows": [], "missing_images": [],
              "corrupted_images": [], "camera_counts": dict.fromkeys(CAMERAS, 0), "kind": kind}
    records = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError("Driving log is empty.")
    first = [x.strip().lower() for x in rows[0]]
    has_header = "steering" in first and "center" in first
    fields = first if has_header else list(FIELDS)
    rows = rows[1:] if has_header else rows
    for row_id, values in enumerate(rows):
        if not values or not any(v.strip() for v in values):
            continue
        report["rows"] += 1
        row = dict(zip(fields, (x.strip() for x in values)))
        try:
            steering = float(row["steering"])
            if not np.isfinite(steering) or abs(steering) > 1:
                raise ValueError("Steering must be finite and normalized to [-1, 1].")
        except (KeyError, ValueError) as error:
            report["invalid_rows"].append({"row": row_id, "reason": str(error)})
            continue
        extras = {}
        for field in ("throttle", "brake", "speed"):
            try:
                val = float(row.get(field, ""))
                extras[field] = val if np.isfinite(val) else None
            except ValueError:
                extras[field] = None
        images = {}
        for camera in CAMERAS:
            value = row.get(camera, "")
            if not value:
                if camera == "center":
                    report["missing_images"].append({"row": row_id, "camera": camera, "path": ""})
                continue
            try:
                resolved = resolve_image(value, path.parent, root)
            except FileNotFoundError:
                report["missing_images"].append({"row": row_id, "camera": camera, "path": value})
                continue
            try:
                read_rgb(resolved)
                # Detect file changes even when CSV paths and labels did not change.
                digest.update(resolved.read_bytes())
            except (OSError, ValueError) as error:
                report["corrupted_images"].append({"row": row_id, "camera": camera,
                                                    "reason": str(error)})
                continue
            images[camera] = resolved
            report["camera_counts"][camera] += 1
        records.append(Record(row_id, steering, images, **extras))
    report["valid_label_rows"] = len(records)
    report["usable_images"] = sum(report["camera_counts"].values())
    bad = sum(len(report[k]) for k in ("invalid_rows", "missing_images", "corrupted_images"))
    if bad:
        message = f"Found {bad} invalid row/image entries; details saved in inspection reports."
        if invalid_policy == "error":
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)
    if not records:
        raise ValueError("No valid steering labels in driving log.")
    # Disallow reusing exactly the same file at multiple timestamps/camera positions.
    used = [p for record in records for p in record.images.values()]
    if len(used) != len(set(used)):
        raise ValueError("Duplicate image paths in log risk train/test leakage. Deduplicate first.")
    return DrivingLog(records, report, digest.hexdigest(), kind)


class DrivingDataset(Dataset):
    def __init__(self, records: list[Record], indices: list[int], preprocessing: dict,
                 augmentation: dict, training: bool = False, seed: int = 42):
        self.preprocess = Preprocessor(**preprocessing)
        self.augment = Augmenter(augmentation)
        self.training, self.seed, self.epoch = training, seed, 0
        cameras = CAMERAS if training and augmentation["side_cameras"] else ("center",)
        correction = augmentation["steering_correction"]
        self.samples = [(records[i], cam, camera_steering(records[i].steering, cam, correction))
                        for i in indices for cam in cameras if cam in records[i].images]
        if not self.samples:
            raise ValueError("No readable camera images in this split.")

    def __len__(self) -> int:
        return len(self.samples)

    def raw(self, index: int) -> tuple[np.ndarray, float, dict]:
        record, camera, label = self.samples[index]
        info = {"row_id": record.row_id, "camera": camera, "path": str(record.images[camera])}
        return read_rgb(record.images[camera]), label, info

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label, _ = self.raw(index)
        if self.training:
            # Per-epoch/per-index RNG is deterministic across worker counts and resume.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            image, label = self.augment(image, label, rng)
        return self.preprocess(image), torch.tensor(label, dtype=torch.float32)
