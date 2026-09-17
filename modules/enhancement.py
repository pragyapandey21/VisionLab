"""
enhancement.py
--------------
Image Enhancement functional module for VisionLab.

Covers syllabus Modules 1 & 2:
  - Image representation / flipping
  - Contrast adjustment
  - Noise removal
  - Linear transformation
  - Log transformation
  - Power-law (gamma) transformation
  - Histogram equalization
  - Basic morphology: erosion, dilation, opening, closing

Every public function takes a BGR or grayscale `np.ndarray` (as produced
by `modules.preprocessing`) and returns a new array of the same kind —
none of these functions mutate their input in place. All user-supplied
parameters are validated via `utils.validation` before use, so callers
(typically `app.py`) can catch `VisionLabError` and show a clean message.
"""

from __future__ import annotations

import cv2
import numpy as np

from utils.validation import (
    validate_choice,
    validate_odd_kernel_size,
    validate_parameter,
)


# ---------------------------------------------------------------------------
# Flip
# ---------------------------------------------------------------------------

FLIP_MODES = {
    "horizontal": 1,
    "vertical": 0,
    "both": -1,
}


def flip_image(img: np.ndarray, mode: str) -> np.ndarray:
    """Flip an image horizontally, vertically, or both.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or grayscale).
    mode : str
        One of "horizontal", "vertical", "both".

    Returns
    -------
    np.ndarray
        The flipped image.
    """
    mode = validate_choice(mode, FLIP_MODES.keys(), "flip mode")
    return cv2.flip(img, FLIP_MODES[mode])


# ---------------------------------------------------------------------------
# Contrast adjustment
# ---------------------------------------------------------------------------

def adjust_contrast(img: np.ndarray, alpha: float, beta: float = 0.0) -> np.ndarray:
    """Adjust image contrast (and optionally brightness) using the
    standard linear model: output = alpha * input + beta.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or grayscale).
    alpha : float
        Contrast gain. 1.0 = unchanged, >1.0 increases contrast,
        <1.0 decreases contrast. Allowed range: 0.1 to 3.0.
    beta : float
        Brightness offset added after scaling. Allowed range: -100 to 100.

    Returns
    -------
    np.ndarray
        Contrast-adjusted image, same shape/dtype as input.
    """
    alpha = validate_parameter(alpha, 0.1, 3.0, "contrast (alpha)")
    beta = validate_parameter(beta, -100, 100, "brightness (beta)")
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


# ---------------------------------------------------------------------------
# Noise removal
# ---------------------------------------------------------------------------

NOISE_METHODS = {"gaussian", "median", "bilateral"}


def remove_noise(img: np.ndarray, method: str = "gaussian", ksize: int = 5) -> np.ndarray:
    """Remove noise from an image using a classical smoothing filter.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or grayscale).
    method : str
        One of "gaussian", "median", "bilateral".
    ksize : int
        Kernel size (must be a positive odd integer). Used for Gaussian
        and median filters. Ignored (a fixed neighborhood is used
        instead) for "bilateral", which uses its own diameter parameter.

    Returns
    -------
    np.ndarray
        Denoised image, same shape/dtype as input.
    """
    method = validate_choice(method, NOISE_METHODS, "noise removal method")
    ksize = validate_odd_kernel_size(ksize, "noise removal kernel size")

    if method == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    if method == "median":
        return cv2.medianBlur(img, ksize)
    # bilateral: edge-preserving smoothing; d derived from ksize, capped for speed
    diameter = min(ksize, 9)
    return cv2.bilateralFilter(img, d=diameter, sigmaColor=75, sigmaSpace=75)


# ---------------------------------------------------------------------------
# Point transformations: linear, log, power-law (gamma)
# ---------------------------------------------------------------------------

def linear_transform(img: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Apply a general linear intensity transformation: output = alpha * input + beta.

    This is conceptually the same operation as `adjust_contrast` but is kept
    as a separate function to match the syllabus's explicit "linear
    transformation" topic, with its own (wider) parameter range for
    experimentation.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or grayscale).
    alpha : float
        Multiplicative gain. Allowed range: 0.0 to 5.0.
    beta : float
        Additive offset. Allowed range: -255 to 255.

    Returns
    -------
    np.ndarray
        Transformed image, same shape/dtype as input (uint8, clipped).
    """
    alpha = validate_parameter(alpha, 0.0, 5.0, "linear alpha")
    beta = validate_parameter(beta, -255, 255, "linear beta")
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def log_transform(img: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Apply a logarithmic intensity transformation: output = c * log(1 + input),
    rescaled back to the 0-255 range.

    Log transformation expands dark pixel values while compressing bright
    ones, useful for enhancing detail in dark regions.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or grayscale), dtype uint8.
    c : float
        Scaling constant applied before rescaling to 0-255. Allowed
        range: 0.1 to 5.0. Larger values brighten the result.

    Returns
    -------
    np.ndarray
        Log-transformed image, same shape as input, dtype uint8.
    """
    c = validate_parameter(c, 0.1, 5.0, "log transform constant (c)")

    img_float = img.astype(np.float64)
    log_img = c * np.log1p(img_float)  # log1p(x) = log(1 + x), avoids log(0)

    # Rescale to 0-255 based on this image's own max, then clip for safety.
    max_val = log_img.max()
    if max_val <= 0:
        return np.zeros_like(img, dtype=np.uint8)

    scaled = (log_img / max_val) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def gamma_transform(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply a power-law (gamma) intensity transformation:
    output = 255 * (input / 255) ** gamma.

    gamma < 1 brightens the image (expands dark tones);
    gamma > 1 darkens the image (expands bright tones).

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or grayscale), dtype uint8.
    gamma : float
        Exponent. Allowed range: 0.1 to 5.0.

    Returns
    -------
    np.ndarray
        Gamma-corrected image, same shape as input, dtype uint8.
    """
    gamma = validate_parameter(gamma, 0.1, 5.0, "gamma")

    # Build a 256-entry lookup table for speed, then apply via cv2.LUT.
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype(np.uint8)
    return cv2.LUT(img, table)


# ---------------------------------------------------------------------------
# Histogram equalization
# ---------------------------------------------------------------------------

def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to improve global contrast.

    For grayscale images, equalizes directly. For color (BGR) images,
    equalizes the luminance (Y) channel in YCrCb space so that color
    information is preserved.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or grayscale), dtype uint8.

    Returns
    -------
    np.ndarray
        Contrast-equalized image, same shape as input.
    """
    if img.ndim == 2:
        return cv2.equalizeHist(img)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    merged = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


# ---------------------------------------------------------------------------
# Basic morphology
# ---------------------------------------------------------------------------

MORPH_OPS = {
    "erosion": cv2.MORPH_ERODE,
    "dilation": cv2.MORPH_DILATE,
    "opening": cv2.MORPH_OPEN,
    "closing": cv2.MORPH_CLOSE,
}


def morphology_op(
    img: np.ndarray,
    op: str,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """Apply a basic morphological operation: erosion, dilation, opening,
    or closing.

    Morphology is typically applied to binary/grayscale images; if a
    color image is passed, the operation is applied per-channel via
    OpenCV's native support.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or grayscale).
    op : str
        One of "erosion", "dilation", "opening", "closing".
    kernel_size : int
        Size of the square structuring element (must be a positive odd
        integer).
    iterations : int
        Number of times to apply the operation (erosion/dilation only;
        ignored for opening/closing, which are single-pass by
        definition). Allowed range: 1 to 10.

    Returns
    -------
    np.ndarray
        Morphologically transformed image, same shape as input.
    """
    op = validate_choice(op, MORPH_OPS.keys(), "morphology operation")
    kernel_size = validate_odd_kernel_size(kernel_size, "morphology kernel size")
    iterations = int(validate_parameter(iterations, 1, 10, "morphology iterations"))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    morph_type = MORPH_OPS[op]

    if op in ("erosion", "dilation"):
        return cv2.morphologyEx(img, morph_type, kernel, iterations=iterations)
    # opening/closing are already multi-step (erode+dilate or vice versa);
    # OpenCV's `iterations` parameter still applies to the erode/dilate
    # sub-steps, so we pass it through for consistency.
    return cv2.morphologyEx(img, morph_type, kernel, iterations=iterations)
