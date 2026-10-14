from pathlib import Path

import pytest
import torch

from robustdrive.config import load_config
from robustdrive.data.sample import create_sample_data


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    return create_sample_data(tmp_path / "sample", count=48)


@pytest.fixture
def config(sample: Path, tmp_path: Path) -> dict:
    config = load_config("configs/smoke.yaml")
    config["data"]["csv_path"] = str(sample)
    config["data"]["gap_frames"] = 1
    config["training"].update(epochs=1, batch_size=16, cpu_threads=2)
    config["experiment"].update(output_root=str(tmp_path / "runs"), name="test")
    return config
