"""
test_feature_detection.py
--------------------------
Unit tests for modules/feature_detection.py.
"""

import cv2
import numpy as np
import pytest

from modules.feature_detection import canny_edges, detect_corners, hough_lines
from utils.validation import VisionLabError


@pytest.fixture
def shapes_img():
    """A synthetic BGR image with a rectangle and a line — gives Canny,
    corner detection, and Hough line detection real features to find."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (150, 150), (255, 255, 255), 2)
    cv2.line(img, (10, 190), (190, 190), (255, 255, 255), 3)
    return img


@pytest.fixture
def blank_img():
    return np.zeros((100, 100, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# canny_edges
# ---------------------------------------------------------------------------

def test_canny_edges_output_shape_and_dtype(shapes_img):
    out = canny_edges(shapes_img, 50, 150)
    assert out.shape == shapes_img.shape[:2]
    assert out.dtype == np.uint8


def test_canny_edges_finds_edges_on_shapes(shapes_img):
    out = canny_edges(shapes_img, 50, 150)
    assert np.count_nonzero(out) > 0


def test_canny_edges_finds_no_edges_on_blank_image(blank_img):
    out = canny_edges(blank_img, 50, 150)
    assert np.count_nonzero(out) == 0


def test_canny_edges_accepts_grayscale_input(shapes_img):
    gray = cv2.cvtColor(shapes_img, cv2.COLOR_BGR2GRAY)
    out = canny_edges(gray, 50, 150)
    assert out.shape == gray.shape


def test_canny_edges_rejects_out_of_range_threshold(shapes_img):
    with pytest.raises(VisionLabError):
        canny_edges(shapes_img, 600, 150)


# ---------------------------------------------------------------------------
# detect_corners
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["shi-tomasi", "harris"])
def test_detect_corners_output_shape(shapes_img, method):
    out = detect_corners(shapes_img, method, 50, 0.01, 10)
    assert out.shape == shapes_img.shape
    assert out.dtype == np.uint8


def test_detect_corners_returns_bgr_for_grayscale_input(shapes_img):
    gray = cv2.cvtColor(shapes_img, cv2.COLOR_BGR2GRAY)
    out = detect_corners(gray, "shi-tomasi")
    assert out.ndim == 3
    assert out.shape[:2] == gray.shape


def test_detect_corners_finds_markers_on_shapes(shapes_img):
    out = detect_corners(shapes_img, "shi-tomasi", 50, 0.01, 10)
    # The drawn image should differ from the plain BGR-converted original
    # somewhere, i.e. at least one corner marker was drawn.
    assert not np.array_equal(out, shapes_img)


def test_detect_corners_rejects_invalid_method(shapes_img):
    with pytest.raises(VisionLabError):
        detect_corners(shapes_img, "fast")


def test_detect_corners_rejects_out_of_range_max_corners(shapes_img):
    with pytest.raises(VisionLabError):
        detect_corners(shapes_img, "shi-tomasi", max_corners=5000)


# ---------------------------------------------------------------------------
# hough_lines
# ---------------------------------------------------------------------------

def test_hough_lines_output_shape(shapes_img):
    out = hough_lines(shapes_img, 50, 150, 50, 30, 10)
    assert out.shape == shapes_img.shape
    assert out.dtype == np.uint8


def test_hough_lines_detects_line_on_shapes(shapes_img):
    out = hough_lines(shapes_img, 50, 150, 50, 30, 10)
    # Green (0, 255, 0) pixels indicate at least one detected line segment.
    green_mask = np.all(out == [0, 255, 0], axis=-1)
    assert green_mask.sum() > 0


def test_hough_lines_finds_nothing_on_blank_image(blank_img):
    out = hough_lines(blank_img, 50, 150, 50, 30, 10)
    green_mask = np.all(out == [0, 255, 0], axis=-1)
    assert green_mask.sum() == 0


def test_hough_lines_accepts_grayscale_input(shapes_img):
    gray = cv2.cvtColor(shapes_img, cv2.COLOR_BGR2GRAY)
    out = hough_lines(gray)
    assert out.ndim == 3


def test_hough_lines_rejects_out_of_range_threshold(shapes_img):
    with pytest.raises(VisionLabError):
        hough_lines(shapes_img, threshold=1000)


def test_hough_lines_rejects_out_of_range_max_gap(shapes_img):
    with pytest.raises(VisionLabError):
        hough_lines(shapes_img, max_line_gap=500)
