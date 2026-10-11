import numpy as np
import pytest
from PIL import Image

from edgeface.preprocess import PREPROCESS, reject, softmax, tensor_from_rgb


def test_rgb_preprocessing_is_identical_for_pil_and_array():
    rgb = np.zeros((57, 103, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    array = tensor_from_rgb(rgb)
    np.testing.assert_array_equal(array, tensor_from_rgb(Image.fromarray(rgb)))
    assert array.shape == (1, 3, 224, 224) and array.dtype == np.float32
    assert float(array[0, 0, 0, 0]) == pytest.approx((1-0.485)/0.229)
    assert float(array[0, 2, 0, 0]) == pytest.approx((0-0.406)/0.225)


def test_unknown_explicit_and_uncertainty_rejection():
    classes = ["Unknown", "son", "another"]
    assert reject(np.array([0.8, 0.1, 0.1]), classes, 0.5)[0] == "Unknown"
    assert reject(np.array([0.3, 0.4, 0.3]), classes, 0.5)[0] == "Unknown"
    assert reject(np.array([0.05, 0.9, 0.05]), classes, 0.5)[0] == "son"


def test_nonfinite_model_output_is_error_not_unknown():
    with pytest.raises(ValueError, match="non-finite"):
        softmax(np.array([[np.nan, 1]]))
    with pytest.raises(ValueError):
        tensor_from_rgb(np.zeros((20, 20, 3), np.uint8), {**PREPROCESS, "color": "BGR"})
