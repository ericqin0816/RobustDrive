import numpy as np
import pytest

from robustdrive.data.augmentation import Augmenter, camera_steering, horizontal_flip


def test_flip_label_and_pixel_geometry():
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    flipped, steering = horizontal_flip(image, 0.4)
    assert steering == -0.4
    np.testing.assert_array_equal(flipped[:, 0], image[:, -1])
    restored, steering = horizontal_flip(flipped, steering)
    np.testing.assert_array_equal(restored, image)
    assert steering == 0.4


def test_recovery_signs():
    assert camera_steering(0.1, "left", 0.2) == pytest.approx(0.3)
    assert camera_steering(0.1, "right", 0.2) == pytest.approx(-0.1)
    assert camera_steering(0.1, "center", 0.2) == 0.1
    assert camera_steering(0.9, "left", 0.2) == 1.0


def test_augmentation_repeatability_and_bounds(config):
    image = np.full((160, 320, 3), 100, dtype=np.uint8)
    augment = Augmenter(config["augmentation"])
    a, ya = augment(image, 0.3, np.random.default_rng(3))
    b, yb = augment(image, 0.3, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)
    assert ya == yb and -1 <= ya <= 1
    assert a.shape == image.shape and a.dtype == np.uint8
