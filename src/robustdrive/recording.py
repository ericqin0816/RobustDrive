"""Turn a finished personal recording into an immutable, auditable training input."""

import csv
import json
import shutil
from pathlib import Path

from robustdrive.data.dataset import CAMERAS, FIELDS, load_log
from robustdrive.utils.logging import utc_now, write_json


def prepare_recording(csv_path: Path, output_root: Path, minimum_frames: int = 600) -> Path:
    """Freeze usable moving frames; never alter recordings or substitute smoke data.

    This is for the desktop workflow. Manual CLI experiments keep their original
    dataset policy. Excluding speed <= 1 mph prevents pre-drive idle recording
    from dominating the training set. All exclusions are counted in metadata.
    """
    if not csv_path.is_file():
        raise ValueError(
            "No driving recording was saved. Choose Training Mode, click Record, "
            "select RobustDrive Recordings, and click Record again. "
            "Press R to stop and wait for saving before closing the game."
        )
    source = load_log(csv_path, kind="real", invalid_policy="error")
    if source.kind != "real":
        raise ValueError("The play workflow needs your own simulator recording, not sample data.")
    moving = [
        record
        for record in source.records
        if "center" in record.images and (record.speed is None or record.speed > 1.0)
    ]
    if len(moving) < minimum_frames:
        raise ValueError(
            f"Only {len(moving)} moving frames were saved. Record several laps, "
            "press R to stop, wait for saving, then close the game. "
            f"At least {minimum_frames} moving frames are needed to start training."
        )
    if not any(r.steering < -0.05 for r in moving) or not any(r.steering > 0.05 for r in moving):
        raise ValueError(
            "The recording needs both left and right turns. Drive a few complete laps."
        )
    output_root = output_root.resolve()
    output = output_root / f"play_{source.fingerprint[:12]}"
    manifest = output / "dataset_metadata.json"
    if manifest.is_file():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        frozen = load_log(output / "driving_log.csv", invalid_policy="error")
        if (
            saved.get("source_fingerprint") == source.fingerprint
            and saved.get("prepared_fingerprint") == frozen.fingerprint
        ):
            return output / "driving_log.csv"
        raise ValueError(
            "An existing recording snapshot changed. Preserve it and use a new recording."
        )
    if output.exists():
        raise ValueError(
            f"An unfinished snapshot exists at {output}. Original recordings are intact."
        )
    (output / "IMG").mkdir(parents=True)
    rows = []
    for record in moving:
        paths = []
        for camera in CAMERAS:
            if camera not in record.images:
                paths.append("")
                continue
            image = record.images[camera]
            relative = Path("IMG") / f"{camera}_{record.row_id:08d}{image.suffix.lower()}"
            shutil.copy2(image, output / relative)
            paths.append(relative.as_posix())
        rows.append([*paths, record.steering, record.throttle, record.brake, record.speed])
    with (output / "driving_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    prepared = load_log(output / "driving_log.csv", invalid_policy="error")
    write_json(
        manifest,
        {
            "kind": "real",
            "created": utc_now(),
            "source_fingerprint": source.fingerprint,
            "prepared_fingerprint": prepared.fingerprint,
            "source_rows": len(source.records),
            "moving_rows": len(moving),
            "excluded_stationary_or_missing_center": len(source.records) - len(moving),
            "filter": "center exists and (speed missing or speed > 1 mph)",
            "original_row_ids": [r.row_id for r in moving],
        },
    )
    return output / "driving_log.csv"
