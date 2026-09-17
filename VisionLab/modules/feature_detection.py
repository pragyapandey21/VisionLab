"""
feature_detection.py
---------------------
Feature Detection functional module for VisionLab.

Covers syllabus Module 3:
  - Canny edge detection
  - Corner detection (Harris and Shi-Tomasi/goodFeaturesToTrack)
  - Hough line detection

All functions accept a BGR or grayscale image and internally convert to
grayscale where required by the underlying OpenCV algorithm. Detection
results that are naturally point/line coordinates (corners, lines) are
drawn onto a BGR copy of the input so the result is directly viewable;
Canny's output is itself a displayable edge map.
"""

from __future__ import annotations

import cv2
import numpy as np

from utils.image_utils import to_bgr, to_gray
from utils.validation import validate_choice, validate_odd_kernel_size, validate_parameter


# ---------------------------------------------------------------------------
# Canny edge detection
# ---------------------------------------------------------------------------

def canny_edges(img: np.ndarray, low_threshold: float = 50, high_threshold: float = 150) -> np.ndarray:
    """Detect edges using the Canny algorithm.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or grayscale).
    low_threshold : float
        Lower hysteresis threshold. Allowed range: 0 to 500.
    high_threshold : float
        Upper hysteresis threshold. Allowed range: 0 to 500. Should
        generally be 2-3x the low threshold, but this is not enforced.

    Returns
    -------
    np.ndarray
        Single-channel binary edge map (0 or 255), same height/width as input.
    """
    low_threshold = validate_parameter(low_threshold, 0, 500, "Canny low threshold")
    high_threshold = validate_parameter(high_threshold, 0, 500, "Canny high threshold")

    gray = to_gray(img)
    return cv2.Canny(gray, low_threshold, high_threshold)


# ---------------------------------------------------------------------------
# Corner detection
# ---------------------------------------------------------------------------

CORNER_METHODS = {"harris", "shi-tomasi"}


def detect_corners(
    img: np.ndarray,
    method: str = "shi-tomasi",
    max_corners: int = 100,
    quality_level: float = 0.01,
    min_distance: int = 10,
) -> np.ndarray:
    """Detect corners and draw them as circles on a copy of the image.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or grayscale).
    method : str
        One of "harris", "shi-tomasi".
    max_corners : int
        Maximum number of corners to return (Shi-Tomasi only; Harris
        returns all points above threshold, capped to this value for
        display purposes). Allowed range: 1 to 1000.
    quality_level : float
        Minimal accepted quality of corners, relative to the best corner
        (Shi-Tomasi) or relative to the max Harris response (Harris).
        Allowed range: 0.001 to 1.0.
    min_distance : int
        Minimum possible Euclidean distance between returned corners
        (Shi-Tomasi only). Allowed range: 1 to 100.

    Returns
    -------
    np.ndarray
        BGR image (copy of input) with detected corners drawn as small
        filled circles.
    """
    method = validate_choice(method, CORNER_METHODS, "corner detection method")
    max_corners = int(validate_parameter(max_corners, 1, 1000, "max corners"))
    quality_level = validate_parameter(quality_level, 0.001, 1.0, "quality level")
    min_distance = int(validate_parameter(min_distance, 1, 100, "min distance"))

    gray = to_gray(img)
    output = to_bgr(img).copy()

    if method == "shi-tomasi":
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
        )
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(output, (int(x), int(y)), 4, (0, 255, 0), -1)
        return output

    # Harris corner detection
    gray_f32 = np.float32(gray)
    harris_response = cv2.cornerHarris(gray_f32, blockSize=2, ksize=3, k=0.04)
    harris_response = cv2.dilate(harris_response, None)  # enlarge marker points

    threshold = quality_level * harris_response.max()
    ys, xs = np.where(harris_response > threshold)

    # Cap the number of drawn points for both performance and readability.
    if len(xs) > max_corners:
        idx = np.linspace(0, len(xs) - 1, max_corners).astype(int)
        xs, ys = xs[idx], ys[idx]

    for x, y in zip(xs, ys):
        cv2.circle(output, (int(x), int(y)), 4, (0, 0, 255), -1)

    return output


# ---------------------------------------------------------------------------
# Hough line detection
# ---------------------------------------------------------------------------

def hough_lines(
    img: np.ndarray,
    canny_low: float = 50,
    canny_high: float = 150,
    threshold: int = 100,
    min_line_length: int = 50,
    max_line_gap: int = 10,
) -> np.ndarray:
    """Detect straight lines using the Probabilistic Hough Transform and
    draw them on a copy of the image.

    The image is first run through Canny edge detection (using the given
    thresholds) since Hough line detection expects a binary edge map as
    input.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or grayscale).
    canny_low, canny_high : float
        Thresholds for the internal Canny edge-detection pre-step.
        Allowed range: 0 to 500 each.
    threshold : int
        Minimum number of votes (intersections in Hough space) needed to
        consider a line detected. Allowed range: 1 to 500.
    min_line_length : int
        Minimum length of a line segment to be accepted. Allowed range:
        1 to 1000.
    max_line_gap : int
        Maximum allowed gap between points on the same line to link them.
        Allowed range: 0 to 100.

    Returns
    -------
    np.ndarray
        BGR image (copy of input) with detected line segments drawn in
        green.
    """
    canny_low = validate_parameter(canny_low, 0, 500, "Hough Canny low threshold")
    canny_high = validate_parameter(canny_high, 0, 500, "Hough Canny high threshold")
    threshold = int(validate_parameter(threshold, 1, 500, "Hough vote threshold"))
    min_line_length = int(validate_parameter(min_line_length, 1, 1000, "min line length"))
    max_line_gap = int(validate_parameter(max_line_gap, 0, 100, "max line gap"))

    gray = to_gray(img)
    edges = cv2.Canny(gray, canny_low, canny_high)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    output = to_bgr(img).copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return output
