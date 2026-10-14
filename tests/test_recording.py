import json

import pytest

from robustdrive.data.dataset import load_log
from robustdrive.recording import prepare_recording


def test_play_workflow_rejects_synthetic_data(sample, tmp_path):
    with pytest.raises(ValueError, match="own simulator recording"):
        prepare_recording(sample, tmp_path / "frozen", minimum_frames=10)


def test_snapshot_preserves_recording_and_is_reusable(sample, tmp_path):
    # Test fixture only: simulate a real-log header for snapshot mechanics.
    # No model is trained and no fixture measurement is reported as research.
    (sample.parent / "dataset_metadata.json").write_text('{"kind":"real","purpose":"test fixture"}')
    original = sample.read_bytes()
    before = load_log(sample).fingerprint
    frozen = prepare_recording(sample, tmp_path / "frozen", minimum_frames=10)
    assert sample.read_bytes() == original
    assert load_log(sample).fingerprint == before
    assert len(load_log(frozen).records) == 48
    assert prepare_recording(sample, tmp_path / "frozen", minimum_frames=10) == frozen
    meta = json.loads((frozen.parent / "dataset_metadata.json").read_text())
    assert meta["moving_rows"] == 48
    assert meta["source_fingerprint"] == before
    with sample.open("a") as stream:
        stream.write("\n")
    assert len(load_log(frozen).records) == 48


def test_stationary_frames_excluded_and_short_recordings_rejected(sample, tmp_path):
    (sample.parent / "dataset_metadata.json").write_text('{"kind":"real","purpose":"test fixture"}')
    lines = sample.read_text().splitlines()
    values = lines[1].split(",")
    values[-1] = "0"
    lines[1] = ",".join(values)
    sample.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="moving frames"):
        prepare_recording(sample, tmp_path / "too_short", minimum_frames=600)
    frozen = prepare_recording(sample, tmp_path / "frozen", minimum_frames=10)
    meta = json.loads((frozen.parent / "dataset_metadata.json").read_text())
    assert meta["moving_rows"] == 47
    assert meta["excluded_stationary_or_missing_center"] == 1


def test_modified_snapshot_is_rejected(sample, tmp_path):
    (sample.parent / "dataset_metadata.json").write_text('{"kind":"real","purpose":"test fixture"}')
    frozen = prepare_recording(sample, tmp_path / "frozen", minimum_frames=10)
    with frozen.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="snapshot changed"):
        prepare_recording(sample, tmp_path / "frozen", minimum_frames=10)
