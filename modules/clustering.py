"""
clustering.py
--------------
Clustering half of the Segmentation & Clustering functional module.

Covers the K-Means portion of syllabus Module 4.

Uses scikit-learn's KMeans (the only place in VisionLab that depends on
scikit-learn, per the project's tech-stack constraints) to cluster image
pixels by color similarity, then recolors each pixel with its cluster's
centroid color to produce a posterized/segmented result.
"""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans

from utils.image_utils import to_bgr
from utils.validation import validate_k_clusters, validate_parameter


def kmeans_segment(
    img: np.ndarray,
    k: int = 4,
    max_iter: int = 100,
    random_state: int = 42,
) -> np.ndarray:
    """Segment an image by clustering pixel colors with K-Means.

    Each pixel is treated as a 3D point (B, G, R) and assigned to one of
    `k` clusters; the output image recolors every pixel with its
    cluster's centroid color, producing a posterized effect that groups
    visually similar regions together.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or grayscale; grayscale is converted to BGR
        first so clustering always operates in a consistent 3-channel
        color space).
    k : int
        Number of clusters. Allowed range: 2 to 20.
    max_iter : int
        Maximum K-Means iterations per run. Allowed range: 10 to 1000.
    random_state : int
        Seed for reproducible cluster assignments across runs.

    Returns
    -------
    np.ndarray
        BGR image, same shape as input, where every pixel has been
        replaced by its cluster's centroid color.
    """
    k = validate_k_clusters(k)
    max_iter = int(validate_parameter(max_iter, 10, 1000, "max iterations"))

    bgr_img = to_bgr(img)
    height, width = bgr_img.shape[:2]

    # Flatten to a list of (B, G, R) pixels for scikit-learn.
    pixel_values = bgr_img.reshape((-1, 3)).astype(np.float32)

    kmeans = KMeans(
        n_clusters=k,
        max_iter=max_iter,
        random_state=random_state,
        n_init=10,
    )
    labels = kmeans.fit_predict(pixel_values)
    centers = np.uint8(kmeans.cluster_centers_)

    # Recolor every pixel with its cluster's centroid color.
    segmented_pixels = centers[labels]
    segmented_img = segmented_pixels.reshape((height, width, 3))

    return segmented_img


def kmeans_cluster_map(
    img: np.ndarray,
    k: int = 4,
    max_iter: int = 100,
    random_state: int = 42,
) -> np.ndarray:
    """Like `kmeans_segment`, but returns a single-channel label map
    instead of a recolored image — useful for inspecting raw cluster
    assignment or building further analysis on top of it.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or grayscale).
    k : int
        Number of clusters. Allowed range: 2 to 20.
    max_iter : int
        Maximum K-Means iterations per run. Allowed range: 10 to 1000.
    random_state : int
        Seed for reproducible cluster assignments across runs.

    Returns
    -------
    np.ndarray
        Single-channel uint8 image where each pixel's value is its
        cluster label (0 to k-1), rescaled to 0-255 for visibility when
        displayed directly.
    """
    k = validate_k_clusters(k)
    max_iter = int(validate_parameter(max_iter, 10, 1000, "max iterations"))

    bgr_img = to_bgr(img)
    height, width = bgr_img.shape[:2]

    pixel_values = bgr_img.reshape((-1, 3)).astype(np.float32)

    kmeans = KMeans(
        n_clusters=k,
        max_iter=max_iter,
        random_state=random_state,
        n_init=10,
    )
    labels = kmeans.fit_predict(pixel_values)

    label_map = labels.reshape((height, width)).astype(np.uint8)

    # Rescale label range [0, k-1] to [0, 255] purely for visual contrast
    # when shown as a grayscale image.
    if k > 1:
        scale = 255.0 / (k - 1)
        label_map = (label_map * scale).astype(np.uint8)

    return label_map
