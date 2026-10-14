import pytest

from robustdrive.config import load_config, validate


def test_inheritance_and_typo_rejection(tmp_path):
    c = load_config("configs/uncertainty.yaml")
    assert c["model"]["dropout"] == 0.25
    assert c["augmentation"]["side_cameras"]
    path = tmp_path / "bad.yaml"
    path.write_text("training:\n  epohcs: 3\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)
    path.write_text("extends: bad.yaml\n")
    with pytest.raises(ValueError, match="Cyclic"):
        load_config(path)


def test_invalid_splits_rejected(config):
    config["data"]["test_split"] = 0.8
    with pytest.raises(ValueError, match="sum to one"):
        validate(config)
