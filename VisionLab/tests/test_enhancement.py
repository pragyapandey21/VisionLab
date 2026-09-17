"""
test_enhancement.py
--------------------
Unit tests for modules/enhancement.py.
"""

import numpy as np
import pytest

from modules.enhancement import (
    adjust_contrast,
    flip_image,
    gamma_transform,
    histogram_equalization,
    linear_transform,
    log_transform,
    morphology_op,
    remove_noise,
)
from utils.validation import VisionLabError


@pytest.fixture
def color_img():
    rng = np.random.default_rng(42)
    return (rng.random((40, 50, 3)) * 255).astype(np.uint8)


@pytest.fixture
def gray_img():
    rng = np.random.default_rng(42)
    return (rng.random((40, 50)) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# flip_image
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["horizontal", "vertical", "both"])
def test_flip_image_preserves_shape(color_img, mode):
    out = flip_image(color_img, mode)
    assert out.shape == color_img.shape


def test_flip_horizontal_actually_flips(color_img):
    out = flip_image(color_img, "horizontal")
    assert np.array_equal(out, color_img[:, ::-1])


def test_flip_image_rejects_invalid_mode(color_img):
    with pytest.raises(VisionLabError):
        flip_image(color_img, "diagonal")


# ---------------------------------------------------------------------------
# adjust_contrast
# ---------------------------------------------------------------------------

def test_adjust_contrast_preserves_shape_and_dtype(color_img):
    out = adjust_contrast(color_img, 1.5, 10)
    assert out.shape == color_img.shape
    assert out.dtype == np.uint8


def test_adjust_contrast_identity_at_alpha_one_beta_zero(color_img):
    out = adjust_contrast(color_img, 1.0, 0.0)
    assert np.array_equal(out, color_img)


def test_adjust_contrast_rejects_out_of_range_alpha(color_img):
    with pytest.raises(VisionLabError):
        adjust_contrast(color_img, 10.0, 0)


# ---------------------------------------------------------------------------
# remove_noise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["gaussian", "median", "bilateral"])
def test_remove_noise_preserves_shape(color_img, method):
    out = remove_noise(color_img, method, 5)
    assert out.shape == color_img.shape


def test_remove_noise_rejects_even_kernel(color_img):
    with pytest.raises(VisionLabError):
        remove_noise(color_img, "gaussian", 4)


def test_remove_noise_rejects_invalid_method(color_img):
    with pytest.raises(VisionLabError):
        remove_noise(color_img, "sharpen", 5)


# ---------------------------------------------------------------------------
# linear_transform
# ---------------------------------------------------------------------------

def test_linear_transform_preserves_shape(gray_img):
    out = linear_transform(gray_img, 1.2, 5)
    assert out.shape == gray_img.shape


def test_linear_transform_rejects_out_of_range(gray_img):
    with pytest.raises(VisionLabError):
        linear_transform(gray_img, 100, 0)


# ---------------------------------------------------------------------------
# log_transform
# ---------------------------------------------------------------------------

def test_log_transform_preserves_shape_and_dtype(gray_img):
    out = log_transform(gray_img, 1.0)
    assert out.shape == gray_img.shape
    assert out.dtype == np.uint8


def test_log_transform_output_in_valid_range(gray_img):
    out = log_transform(gray_img, 1.0)
    assert out.min() >= 0
    assert out.max() <= 255


def test_log_transform_handles_all_zero_image():
    zeros = np.zeros((10, 10), dtype=np.uint8)
    out = log_transform(zeros, 1.0)
    assert out.shape == zeros.shape
    assert np.all(out == 0)


def test_log_transform_rejects_out_of_range_c(gray_img):
    with pytest.raises(VisionLabError):
        log_transform(gray_img, 10.0)


# ---------------------------------------------------------------------------
# gamma_transform
# ---------------------------------------------------------------------------

def test_gamma_transform_preserves_shape_and_dtype(gray_img):
    out = gamma_transform(gray_img, 2.0)
    assert out.shape == gray_img.shape
    assert out.dtype == np.uint8


def test_gamma_transform_identity_at_gamma_one(gray_img):
    out = gamma_transform(gray_img, 1.0)
    # allow tiny rounding differences from the LUT computation
    assert np.allclose(out, gray_img, atol=1)


def test_gamma_transform_rejects_out_of_range(gray_img):
    with pytest.raises(VisionLabError):
        gamma_transform(gray_img, 10.0)


# ---------------------------------------------------------------------------
# histogram_equalization
# ---------------------------------------------------------------------------

def test_histogram_equalization_grayscale_preserves_shape(gray_img):
    out = histogram_equalization(gray_img)
    assert out.shape == gray_img.shape


def test_histogram_equalization_color_preserves_shape(color_img):
    out = histogram_equalization(color_img)
    assert out.shape == color_img.shape


# ---------------------------------------------------------------------------
# morphology_op
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op", ["erosion", "dilation", "opening", "closing"])
def test_morphology_op_preserves_shape_gray(gray_img, op):
    out = morphology_op(gray_img, op, 3, 1)
    assert out.shape == gray_img.shape


@pytest.mark.parametrize("op", ["erosion", "dilation", "opening", "closing"])
def test_morphology_op_preserves_shape_color(color_img, op):
    out = morphology_op(color_img, op, 3, 1)
    assert out.shape == color_img.shape


def test_morphology_op_rejects_invalid_operation(gray_img):
    with pytest.raises(VisionLabError):
        morphology_op(gray_img, "skeletonize", 3, 1)


def test_morphology_op_rejects_even_kernel(gray_img):
    with pytest.raises(VisionLabError):
        morphology_op(gray_img, "erosion", 4, 1)


def test_morphology_erosion_shrinks_white_region():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 5:15] = 255
    eroded = morphology_op(img, "erosion", 3, 1)
    assert np.count_nonzero(eroded) < np.count_nonzero(img)


def test_morphology_dilation_grows_white_region():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 5:15] = 255
    dilated = morphology_op(img, "dilation", 3, 1)
    assert np.count_nonzero(dilated) > np.count_nonzero(img)
