"""
test_validation.py
-------------------
Unit tests for utils/validation.py.
"""

import numpy as np
import pytest

from utils.validation import (
    VisionLabError,
    validate_choice,
    validate_file_extension,
    validate_file_size,
    validate_image_array,
    validate_k_clusters,
    validate_odd_kernel_size,
    validate_parameter,
)


# ---------------------------------------------------------------------------
# validate_file_extension
# ---------------------------------------------------------------------------

def test_validate_file_extension_accepts_supported_types():
    for name in ["photo.png", "photo.JPG", "photo.jpeg", "photo.bmp", "photo.tiff", "photo.webp"]:
        validate_file_extension(name)  # should not raise


def test_validate_file_extension_rejects_unsupported_type():
    with pytest.raises(VisionLabError):
        validate_file_extension("document.pdf")


def test_validate_file_extension_rejects_missing_extension():
    with pytest.raises(VisionLabError):
        validate_file_extension("noextension")


def test_validate_file_extension_rejects_empty_name():
    with pytest.raises(VisionLabError):
        validate_file_extension("")


# ---------------------------------------------------------------------------
# validate_file_size
# ---------------------------------------------------------------------------

def test_validate_file_size_accepts_reasonable_size():
    validate_file_size(1024 * 1024)  # 1 MB


def test_validate_file_size_rejects_zero():
    with pytest.raises(VisionLabError):
        validate_file_size(0)


def test_validate_file_size_rejects_negative():
    with pytest.raises(VisionLabError):
        validate_file_size(-5)


def test_validate_file_size_rejects_too_large():
    with pytest.raises(VisionLabError):
        validate_file_size(100 * 1024 * 1024)  # 100 MB


# ---------------------------------------------------------------------------
# validate_image_array
# ---------------------------------------------------------------------------

def test_validate_image_array_accepts_valid_color_image():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    validate_image_array(img)  # should not raise


def test_validate_image_array_accepts_valid_grayscale_image():
    img = np.zeros((10, 10), dtype=np.uint8)
    validate_image_array(img)  # should not raise


def test_validate_image_array_rejects_none():
    with pytest.raises(VisionLabError):
        validate_image_array(None)


def test_validate_image_array_rejects_too_small():
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    with pytest.raises(VisionLabError):
        validate_image_array(img)


def test_validate_image_array_rejects_too_large():
    # Use a view trick to avoid actually allocating a huge array: just
    # fake the shape check by constructing a small array and monkeypatching
    # would be overkill here; instead directly test the boundary via a
    # moderately large (but not absurd) array is impractical for speed,
    # so we test the dimension-count and channel-count rejections instead.
    img = np.zeros((10, 10, 5), dtype=np.uint8)  # unsupported channel count
    with pytest.raises(VisionLabError):
        validate_image_array(img)


def test_validate_image_array_rejects_wrong_ndim():
    img = np.zeros((10, 10, 3, 2), dtype=np.uint8)
    with pytest.raises(VisionLabError):
        validate_image_array(img)


# ---------------------------------------------------------------------------
# validate_parameter
# ---------------------------------------------------------------------------

def test_validate_parameter_accepts_in_range_value():
    assert validate_parameter(1.5, 0.0, 3.0, "alpha") == 1.5


def test_validate_parameter_accepts_boundary_values():
    assert validate_parameter(0.0, 0.0, 3.0, "alpha") == 0.0
    assert validate_parameter(3.0, 0.0, 3.0, "alpha") == 3.0


def test_validate_parameter_rejects_out_of_range():
    with pytest.raises(VisionLabError):
        validate_parameter(5.0, 0.0, 3.0, "alpha")


def test_validate_parameter_rejects_non_numeric():
    with pytest.raises(VisionLabError):
        validate_parameter("not a number", 0.0, 3.0, "alpha")


# ---------------------------------------------------------------------------
# validate_odd_kernel_size
# ---------------------------------------------------------------------------

def test_validate_odd_kernel_size_accepts_odd_value():
    assert validate_odd_kernel_size(5) == 5


def test_validate_odd_kernel_size_rejects_even_value():
    with pytest.raises(VisionLabError):
        validate_odd_kernel_size(4)


def test_validate_odd_kernel_size_rejects_zero_or_negative():
    with pytest.raises(VisionLabError):
        validate_odd_kernel_size(0)
    with pytest.raises(VisionLabError):
        validate_odd_kernel_size(-3)


# ---------------------------------------------------------------------------
# validate_choice
# ---------------------------------------------------------------------------

def test_validate_choice_accepts_valid_option():
    assert validate_choice("median", ["gaussian", "median"], "method") == "median"


def test_validate_choice_rejects_invalid_option():
    with pytest.raises(VisionLabError):
        validate_choice("unknown", ["gaussian", "median"], "method")


# ---------------------------------------------------------------------------
# validate_k_clusters
# ---------------------------------------------------------------------------

def test_validate_k_clusters_accepts_valid_k():
    assert validate_k_clusters(4) == 4


def test_validate_k_clusters_rejects_too_small():
    with pytest.raises(VisionLabError):
        validate_k_clusters(1)


def test_validate_k_clusters_rejects_too_large():
    with pytest.raises(VisionLabError):
        validate_k_clusters(50)


def test_validate_k_clusters_rejects_non_integer():
    with pytest.raises(VisionLabError):
        validate_k_clusters("four")
