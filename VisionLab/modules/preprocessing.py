"""
preprocessing.py
-----------------
Entry point of the VisionLab pipeline: takes a raw uploaded file (name,
bytes, size) and produces a validated, safely-sized image ready for the
Enhancement, Feature Detection, and Segmentation & Clustering modules.

This module deliberately does NOT implement any "real" computer vision
transform (that lives in `enhancement.py`, `feature_detection.py`, etc.).
It only handles the load -> validate -> resize step of the workflow.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from utils.image_utils import load_image_from_bytes, resize_if_needed, to_gray
from utils.validation import (
    VisionLabError,
    validate_file_extension,
    validate_file_size,
    validate_image_array,
)


def prepare_uploaded_image(
    filename: str,
    file_bytes: bytes,
    file_size: int,
) -> np.ndarray:
    """Validate and decode an uploaded file into a ready-to-use BGR image.

    This is the single function `app.py` should call right after a user
    uploads a file. It performs, in order:
      1. Extension check
      2. File size check
      3. Decoding to a NumPy array
      4. Image-content validation (dimensions, channels)
      5. Resizing if the image exceeds the safe processing dimension

    Parameters
    ----------
    filename : str
        Original name of the uploaded file (used to check the extension).
    file_bytes : bytes
        Raw bytes of the uploaded file.
    file_size : int
        Size of the uploaded file in bytes.

    Returns
    -------
    np.ndarray
        A validated, appropriately-sized BGR image ready for processing.

    Raises
    ------
    VisionLabError
        If any validation step fails. The message is safe to show
        directly to the user in the Streamlit UI.
    """
    validate_file_extension(filename)
    validate_file_size(file_size)

    img = load_image_from_bytes(file_bytes)
    validate_image_array(img)

    img = resize_if_needed(img)
    return img


def load_sample_image(path: str) -> np.ndarray:
    """Load one of the bundled sample images from `sample_data/` on disk.

    Parameters
    ----------
    path : str
        Filesystem path to the sample image.

    Returns
    -------
    np.ndarray
        A validated, appropriately-sized BGR image ready for processing.

    Raises
    ------
    VisionLabError
        If the file does not exist or cannot be read as an image.
    """
    if not os.path.isfile(path):
        raise VisionLabError(f"Sample image not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    validate_image_array(img)

    img = resize_if_needed(img)
    return img


def as_working_copy(img: np.ndarray) -> np.ndarray:
    """Return a defensive copy of an image so downstream operations never
    mutate the original in place (important since Streamlit reruns the
    script on every interaction and may reuse cached arrays).

    Parameters
    ----------
    img : np.ndarray
        Source image.

    Returns
    -------
    np.ndarray
        An independent copy of the image.
    """
    return img.copy()


def ensure_grayscale_if_requested(img: np.ndarray, want_gray: bool) -> np.ndarray:
    """Convenience helper: convert to grayscale only if the caller asked for it.

    Several algorithms downstream (Canny, corner detection, watershed distance
    transform) require grayscale input regardless of user preference; those
    modules call `to_gray` directly. This helper is for the general "let the
    user preview grayscale" toggle in the UI.

    Parameters
    ----------
    img : np.ndarray
        Source image (BGR or grayscale).
    want_gray : bool
        Whether the caller wants a grayscale version.

    Returns
    -------
    np.ndarray
        Grayscale image if `want_gray` is True, otherwise the original image.
    """
    if want_gray:
        return to_gray(img)
    return img


def get_supported_sample_images(sample_dir: str) -> list[str]:
    """List available sample image filenames in `sample_data/`.

    Parameters
    ----------
    sample_dir : str
        Path to the sample_data directory.

    Returns
    -------
    list[str]
        Filenames (not full paths) of supported sample images found.
        Returns an empty list if the directory does not exist or is empty.
    """
    if not os.path.isdir(sample_dir):
        return []

    from utils.validation import ALLOWED_EXTENSIONS

    files = []
    for name in sorted(os.listdir(sample_dir)):
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext in ALLOWED_EXTENSIONS:
            files.append(name)
    return files
