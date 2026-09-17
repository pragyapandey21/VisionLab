"""
segmentation.py
----------------
Segmentation half of the Segmentation & Clustering functional module.

Covers the watershed segmentation portion of syllabus Module 4.

Implements the standard OpenCV watershed recipe:
  1. Grayscale + Otsu threshold -> binary foreground/background guess
  2. Morphological opening to remove small noise
  3. Distance transform + threshold -> sure foreground markers
  4. Dilation of the binary mask -> sure background
  5. Subtract to get the unknown region
  6. Connected components on sure foreground -> initial markers
  7. cv2.watershed() to grow markers and find boundaries

K-Means clustering lives separately in `clustering.py`.
"""

from __future__ import annotations

import cv2
import numpy as np

from utils.image_utils import to_bgr, to_gray
from utils.validation import validate_parameter


def watershed_segment(
    img: np.ndarray,
    fg_threshold_ratio: float = 0.5,
    morph_kernel_size: int = 3,
) -> np.ndarray:
    """Segment an image into regions using marker-based watershed.

    Best suited for images with reasonably distinct, roughly convex
    foreground objects on a contrasting background (e.g. coins, cells,
    seeds). Detected region boundaries are drawn in red on a copy of
    the original image.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or grayscale).
    fg_threshold_ratio : float
        Fraction of the max distance-transform value used to decide the
        "sure foreground" region. Lower values mark more of the image as
        foreground (more, smaller regions); higher values are stricter.
        Allowed range: 0.1 to 0.9.
    morph_kernel_size : int
        Size of the square kernel used for the noise-removal morphological
        opening step. Allowed range: 1 to 15 (odd or even both accepted;
        OpenCV does not require an odd kernel here).

    Returns
    -------
    np.ndarray
        BGR image (copy of input) with watershed region boundaries drawn
        in red.
    """
    fg_threshold_ratio = validate_parameter(
        fg_threshold_ratio, 0.1, 0.9, "foreground threshold ratio"
    )
    morph_kernel_size = int(
        validate_parameter(morph_kernel_size, 1, 15, "morphology kernel size")
    )

    gray = to_gray(img)
    output = to_bgr(img).copy()

    # 1. Binary threshold via Otsu (auto-picks a good global threshold).
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 2. Remove small noise specks with morphological opening.
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. Sure background: dilate the opened mask outward.
    sure_bg = cv2.dilate(opened, kernel, iterations=3)

    # 4. Sure foreground: threshold the distance transform.
    dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    max_dist = dist_transform.max()
    if max_dist <= 0:
        # No clear foreground found (e.g. blank/uniform image) — return
        # the original image unmodified rather than erroring out.
        return output

    _, sure_fg = cv2.threshold(
        dist_transform, fg_threshold_ratio * max_dist, 255, 0
    )
    sure_fg = np.uint8(sure_fg)

    # 5. Unknown region = background minus foreground.
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6. Label connected components in the sure foreground as initial markers.
    num_labels, markers = cv2.connectedComponents(sure_fg)
    # Shift labels up by 1 so background is 1, not 0 (0 is reserved for
    # "unknown" by cv2.watershed's convention).
    markers = markers + 1
    markers[unknown == 255] = 0

    # 7. Run watershed; it modifies `markers` in place, setting boundary
    # pixels to -1.
    cv2.watershed(output, markers)
    output[markers == -1] = (0, 0, 255)

    return output
