import csv
from pathlib import Path

import pytest

from robustdrive.data.dataset import DrivingDataset, load_log
from robustdrive.data.splits import temporal_split
from robustdrive.utils.paths import resolve_image


def test_moved_windows_and_posix_paths(sample):
    folder = sample.parent
    expected = (folder / "IMG/center_000000.jpg").resolve()
    for value in (
        r"C:\old computer\IMG\center_000000.jpg",
        "/Users/old/IMG/center_000000.jpg",
        " IMG/center_000000.jpg ",
        str(expected),
    ):
        assert resolve_image(value, folder) == expected
    assert resolve_image("center_000000.jpg", Path("unused"), folder / "IMG") == expected


def test_headerless_and_synthetic_provenance(sample):
    original = load_log(sample)
    assert original.kind == "synthetic"
    assert original.inspection["camera_counts"] == {"center": 48, "left": 48, "right": 48}
    sample.write_text("\n".join(sample.read_text().splitlines()[1:]) + "\n")
    no_header = load_log(sample)
    assert [r.steering for r in original.records] == [r.steering for r in no_header.records]


def test_missing_corrupt_and_invalid_rows_are_reported(sample):
    (sample.parent / "IMG/left_000000.jpg").unlink()
    (sample.parent / "IMG/right_000001.jpg").write_bytes(b"invalid image")
    with sample.open("a") as stream:
        stream.write("missing.jpg,,,nan,0,0,0\n")
    with pytest.warns(UserWarning):
        log = load_log(sample)
    assert len(log.inspection["missing_images"]) == 1
    assert len(log.inspection["corrupted_images"]) == 1
    assert len(log.inspection["invalid_rows"]) == 1
    with pytest.raises(ValueError, match="invalid row/image"):
        load_log(sample, invalid_policy="error")


def test_duplicate_files_rejected(sample):
    with sample.open() as stream:
        rows = list(csv.reader(stream))
    rows[2][0] = rows[1][0]
    with sample.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError, match="Duplicate"):
        load_log(sample)


def test_temporal_gaps_and_camera_membership(sample, config):
    log = load_log(sample)
    splits = temporal_split(48, gap=3)
    assert min(splits["validation"]) - max(splits["train"]) == 4
    assert min(splits["test"]) - max(splits["validation"]) == 4
    assert len(splits["discarded"]) == 6
    train = DrivingDataset(
        log.records, splits["train"], config["preprocessing"], config["augmentation"], True
    )
    test = DrivingDataset(
        log.records, splits["test"], config["preprocessing"], config["augmentation"]
    )
    assert len(train) == len(splits["train"]) * 3
    assert len(test) == len(splits["test"])
    assert {r.row_id for r, _, _ in train.samples}.isdisjoint(
        {r.row_id for r, _, _ in test.samples}
    )
    assert all(camera == "center" for _, camera, _ in test.samples)


def test_small_split_fails_instead_of_leaking():
    with pytest.raises(ValueError, match="insufficient"):
        temporal_split(20, gap=30)
