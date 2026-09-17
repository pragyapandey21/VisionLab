"""
test_clustering.py
-------------------
Unit tests for modules/clustering.py.
"""

import cv2
import numpy as np
import pytest

from modules.clustering import kmeans_cluster_map, kmeans_segment
from utils.validation import VisionLabError


@pytest.fixture
def three_color_img():
    """An image with three distinct, solid color blocks — a clean case
    for verifying K-Means recovers the right number of clusters."""
    img = np.zeros((60, 90, 3), dtype=np.uint8)
    img[:, :30] = (255, 0, 0)
    img[:, 30:60] = (0, 255, 0)
    img[:, 60:] = (0, 0, 255)
    return img


# ---------------------------------------------------------------------------
# kmeans_segment
# ---------------------------------------------------------------------------

def test_kmeans_segment_output_shape_and_dtype(three_color_img):
    out = kmeans_segment(three_color_img, k=3)
    assert out.shape == three_color_img.shape
    assert out.dtype == np.uint8


def test_kmeans_segment_recovers_correct_cluster_count(three_color_img):
    out = kmeans_segment(three_color_img, k=3)
    unique_colors = np.unique(out.reshape(-1, 3), axis=0)
    assert len(unique_colors) <= 3


def test_kmeans_segment_accepts_grayscale_input(three_color_img):
    gray = cv2.cvtColor(three_color_img, cv2.COLOR_BGR2GRAY)
    out = kmeans_segment(gray, k=2)
    assert out.ndim == 3
    assert out.shape[:2] == gray.shape


def test_kmeans_segment_is_reproducible_with_fixed_seed(three_color_img):
    out1 = kmeans_segment(three_color_img, k=3, random_state=7)
    out2 = kmeans_segment(three_color_img, k=3, random_state=7)
    assert np.array_equal(out1, out2)


def test_kmeans_segment_rejects_k_too_small(three_color_img):
    with pytest.raises(VisionLabError):
        kmeans_segment(three_color_img, k=1)


def test_kmeans_segment_rejects_k_too_large(three_color_img):
    with pytest.raises(VisionLabError):
        kmeans_segment(three_color_img, k=100)


# ---------------------------------------------------------------------------
# kmeans_cluster_map
# ---------------------------------------------------------------------------

def test_kmeans_cluster_map_output_shape_and_dtype(three_color_img):
    out = kmeans_cluster_map(three_color_img, k=3)
    assert out.ndim == 2
    assert out.shape == three_color_img.shape[:2]
    assert out.dtype == np.uint8


def test_kmeans_cluster_map_has_at_most_k_unique_values(three_color_img):
    out = kmeans_cluster_map(three_color_img, k=3)
    unique_vals = np.unique(out)
    assert len(unique_vals) <= 3


def test_kmeans_cluster_map_rejects_invalid_k(three_color_img):
    with pytest.raises(VisionLabError):
        kmeans_cluster_map(three_color_img, k=0)
