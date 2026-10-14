import numpy as np
import pytest
import torch

from robustdrive.data.preprocessing import Preprocessor, read_rgb
from robustdrive.simulator.udacity import decode_telemetry


def test_rgb_normalization_and_shape():
    image = np.zeros((160, 320, 3), dtype=np.uint8)
    image[..., 0] = 255
    result = Preprocessor()(image)
    assert result.shape == (3, 66, 200)
    assert result.dtype == torch.float32
    assert torch.all(result[0] == 1) and torch.all(result[2] == -1)


def test_disk_and_simulator_preprocessing_are_identical(sample):
    import base64

    path = sample.parent / "IMG/center_000000.jpg"
    rgb, speed = decode_telemetry(
        {"image": base64.b64encode(path.read_bytes()).decode(), "speed": "12.5"}
    )
    assert speed == 12.5
    assert torch.equal(Preprocessor()(read_rgb(path)), Preprocessor()(rgb))


def test_impossible_crop_fails():
    with pytest.raises(ValueError, match="Crop"):
        Preprocessor()(np.zeros((50, 100, 3), dtype=np.uint8))
