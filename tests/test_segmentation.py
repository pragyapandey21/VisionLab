"""
test_segmentation.py
---------------------
Unit tests for modules/segmentation.py.
"""

import cv2
import numpy as np
import pytest

from modules.segmentation import watershed_segment
from utils.validation import VisionLabError


@pytest.fixture
def two_blobs_img():
    """Two touching white circles on a black background — the classic
    watershed test case, since a simple threshold alone cannot separate
    them but watershed's distance-transform approach can."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (80, 100), 40, (255, 255, 255), -1)
    cv2.circle(img, (140, 100), 40, (255, 255, 255), -1)
    return img


@pytest.fixture
def blank_img():
    return np.zeros((100, 100, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# watershed_segment
# ---------------------------------------------------------------------------

def test_watershed_segment_output_shape(two_blobs_img):
    out = watershed_segment(two_blobs_img)
    assert out.shape == two_blobs_img.shape
    assert out.dtype == np.uint8


def test_watershed_segment_draws_boundaries_on_touching_blobs(two_blobs_img):
    out = watershed_segment(two_blobs_img)
    red_mask = np.all(out == [0, 0, 255], axis=-1)
    assert red_mask.sum() > 0


def test_watershed_segment_handles_blank_image_without_crashing(blank_img):
    # A uniform/blank image has no real foreground; the function should
    # return early (unmodified image) rather than crash or hang. Note:
    # cv2.watershed always marks the image border as boundary by
    # convention when it does run, so we only assert shape/no-crash here.
    out = watershed_segment(blank_img)
    assert out.shape == blank_img.shape
    assert out.dtype == np.uint8


def test_watershed_segment_accepts_grayscale_input(two_blobs_img):
    gray = cv2.cvtColor(two_blobs_img, cv2.COLOR_BGR2GRAY)
    out = watershed_segment(gray)
    assert out.ndim == 3
    assert out.shape[:2] == gray.shape


def test_watershed_segment_rejects_out_of_range_fg_ratio(two_blobs_img):
    with pytest.raises(VisionLabError):
        watershed_segment(two_blobs_img, fg_threshold_ratio=1.5)


def test_watershed_segment_rejects_out_of_range_kernel_size(two_blobs_img):
    with pytest.raises(VisionLabError):
        watershed_segment(two_blobs_img, morph_kernel_size=100)


def test_watershed_segment_does_not_mutate_input(two_blobs_img):
    original_copy = two_blobs_img.copy()
    watershed_segment(two_blobs_img)
    assert np.array_equal(two_blobs_img, original_copy)
