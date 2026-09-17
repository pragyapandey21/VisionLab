"""
image_utils.py
---------------
Shared, low-level image I/O and conversion helpers used across VisionLab.

These functions wrap OpenCV/NumPy operations that would otherwise be
duplicated in every CV module: decoding uploaded bytes, resizing large
images for performance, converting between color spaces, and encoding
a processed image back to bytes for download.

Validation of *content* (is this a real image? is it too big?) lives in
`utils/validation.py`. This module focuses on the *mechanics* of getting
image data into and out of NumPy arrays.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from utils.validation import MAX_IMAGE_DIMENSION, VisionLabError, validate_image_array


def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """Decode raw file bytes (e.g. from Streamlit's file_uploader) into a
    BGR NumPy image array, as OpenCV expects.

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes of the uploaded image file.

    Returns
    -------
    np.ndarray
        Decoded image in BGR order, dtype uint8.

    Raises
    ------
    VisionLabError
        If the bytes cannot be decoded into a valid image.
    """
    if not file_bytes:
        raise VisionLabError("The uploaded file appears to be empty.")

    file_array = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    validate_image_array(img)
    return img


def resize_if_needed(img: np.ndarray, max_dim: int = MAX_IMAGE_DIMENSION) -> np.ndarray:
    """Downscale an image (preserving aspect ratio) if either side exceeds
    `max_dim`. Used to keep processing responsive on very large uploads.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or grayscale).
    max_dim : int
        Maximum allowed size, in pixels, for the longer side.

    Returns
    -------
    np.ndarray
        The original image if already within bounds, otherwise a resized copy.
    """
    height, width = img.shape[:2]
    longest_side = max(height, width)

    if longest_side <= max_dim:
        return img

    scale = max_dim / float(longest_side)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert an image to RGB for display in Streamlit/Matplotlib.

    Handles grayscale (1-channel), BGR (3-channel), and BGRA (4-channel)
    inputs. Returns the input unchanged if it is already single-channel
    and the caller wants grayscale (use `to_gray` explicitly for that).

    Parameters
    ----------
    img : np.ndarray
        Input image in grayscale, BGR, or BGRA format.

    Returns
    -------
    np.ndarray
        Image converted to RGB (3-channel).
    """
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    channels = img.shape[2]
    if channels == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if channels == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    raise VisionLabError(f"Cannot convert image with {channels} channels to RGB.")


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert an image to single-channel grayscale.

    If the image is already grayscale, it is returned unchanged.

    Parameters
    ----------
    img : np.ndarray
        Input image in grayscale, BGR, or BGRA format.

    Returns
    -------
    np.ndarray
        Single-channel grayscale image.
    """
    if img.ndim == 2:
        return img

    channels = img.shape[2]
    if channels == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if channels == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    raise VisionLabError(f"Cannot convert image with {channels} channels to grayscale.")


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Ensure an image is 3-channel BGR, converting from grayscale if needed.

    Parameters
    ----------
    img : np.ndarray
        Input image in grayscale or BGR format.

    Returns
    -------
    np.ndarray
        3-channel BGR image.
    """
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def encode_for_download(img: np.ndarray, ext: str = ".png") -> bytes:
    """Encode a NumPy image array back into file bytes for a Streamlit
    download button.

    Parameters
    ----------
    img : np.ndarray
        Image to encode (grayscale or BGR — NOT RGB; convert back to BGR
        first if you displayed it as RGB).
    ext : str
        Target file extension including the dot, e.g. ".png" or ".jpg".

    Returns
    -------
    bytes
        Encoded image bytes suitable for `st.download_button`.

    Raises
    ------
    VisionLabError
        If encoding fails (e.g. unsupported extension).
    """
    success, buffer = cv2.imencode(ext, img)
    if not success:
        raise VisionLabError(f"Failed to encode image as '{ext}' for download.")
    return buffer.tobytes()


def get_image_dimensions(img: np.ndarray) -> Tuple[int, int]:
    """Return (width, height) of an image, in that order for display purposes.

    Parameters
    ----------
    img : np.ndarray
        Input image.

    Returns
    -------
    Tuple[int, int]
        (width, height) in pixels.
    """
    height, width = img.shape[:2]
    return width, height
