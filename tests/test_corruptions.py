import numpy as np
import pytest

from robustdrive.evaluation.robustness import CORRUPTIONS, corrupt


@pytest.mark.parametrize("name", CORRUPTIONS)
def test_clean_identity_and_all_severities(name):
    image = np.random.default_rng(0).integers(20, 230, size=(160, 320, 3), dtype=np.uint8)
    np.testing.assert_array_equal(corrupt(image, name, 0), image)
    for severity in range(1, 6):
        a = corrupt(image, name, severity, 12)
        b = corrupt(image, name, severity, 12)
        assert a.shape == image.shape and a.dtype == np.uint8
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, image)


def test_brightness_monotonic_and_invalid_conditions():
    image = np.full((160, 320, 3), 128, dtype=np.uint8)
    values = [corrupt(image, "brightness_down", i).mean() for i in range(6)]
    assert all(a > b for a, b in zip(values, values[1:]))
    with pytest.raises(ValueError):
        corrupt(image, "not-a-corruption", 1)
    with pytest.raises(ValueError):
        corrupt(image, "clean", 1)
