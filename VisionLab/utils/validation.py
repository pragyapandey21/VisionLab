"""
validation.py
--------------
Centralized input validation and error-handling helpers for VisionLab.

These functions never raise raw, unhandled exceptions for expected
"bad input" situations. Instead they raise `VisionLabError`, a single
custom exception type that `app.py` can catch and display cleanly to
the user via Streamlit, satisfying the project's error-handling /
validation non-functional requirement.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

# Formats we accept from Streamlit's file_uploader (by extension).
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"}

# Hard cap to keep processing responsive (performance non-functional requirement).
MAX_FILE_SIZE_MB = 15
MAX_IMAGE_DIMENSION = 4000  # pixels, either side, before we require a resize


class VisionLabError(Exception):
    """Raised for any expected, user-facing validation failure."""


def validate_file_extension(filename: str) -> None:
    """Ensure the uploaded file has a supported image extension.

    Parameters
    ----------
    filename : str
        Name of the uploaded file, e.g. "photo.png".

    Raises
    ------
    VisionLabError
        If the filename has no extension or an unsupported one.
    """
    if not filename or "." not in filename:
        raise VisionLabError(
            "Could not determine the file type. Please upload an image "
            f"with one of these extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise VisionLabError(
            f"Unsupported file type '.{ext}'. Please upload one of: "
            f"{', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )


def validate_file_size(size_bytes: int) -> None:
    """Ensure the uploaded file is not larger than MAX_FILE_SIZE_MB.

    Parameters
    ----------
    size_bytes : int
        Size of the uploaded file in bytes.

    Raises
    ------
    VisionLabError
        If the file exceeds the configured size limit.
    """
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if size_bytes <= 0:
        raise VisionLabError("The uploaded file appears to be empty.")
    if size_bytes > max_bytes:
        raise VisionLabError(
            f"File is too large ({size_bytes / (1024 * 1024):.1f} MB). "
            f"Please upload an image smaller than {MAX_FILE_SIZE_MB} MB."
        )


def validate_image_array(img: Optional[np.ndarray]) -> None:
    """Validate that a decoded image is usable by the rest of the pipeline.

    Parameters
    ----------
    img : np.ndarray or None
        The decoded image (as produced by OpenCV/`image_utils.load_image`).

    Raises
    ------
    VisionLabError
        If the image failed to decode, has invalid dimensions, or an
        unsupported number of channels.
    """
    if img is None:
        raise VisionLabError(
            "The file could not be read as an image. It may be corrupted "
            "or in an unsupported format."
        )

    if not isinstance(img, np.ndarray):
        raise VisionLabError("Internal error: decoded image is not a valid array.")

    if img.ndim not in (2, 3):
        raise VisionLabError("Image has an unexpected number of dimensions.")

    if img.ndim == 3 and img.shape[2] not in (1, 3, 4):
        raise VisionLabError(
            f"Image has an unsupported number of channels: {img.shape[2]}."
        )

    height, width = img.shape[:2]
    if height < 2 or width < 2:
        raise VisionLabError("Image is too small to process (must be at least 2x2 pixels).")

    if height > MAX_IMAGE_DIMENSION or width > MAX_IMAGE_DIMENSION:
        raise VisionLabError(
            f"Image is too large ({width}x{height}). Maximum supported "
            f"dimension is {MAX_IMAGE_DIMENSION}px per side."
        )


def validate_parameter(
    value: float,
    min_val: float,
    max_val: float,
    name: str,
) -> float:
    """Validate that a numeric parameter falls within an inclusive range.

    Parameters
    ----------
    value : float
        The value to check (e.g. a slider result from Streamlit).
    min_val, max_val : float
        Inclusive bounds the value must fall within.
    name : str
        Human-readable parameter name, used in the error message.

    Returns
    -------
    float
        The validated value, unchanged.

    Raises
    ------
    VisionLabError
        If the value is not numeric or falls outside the allowed range.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise VisionLabError(f"Parameter '{name}' must be a number.") from None

    if value < min_val or value > max_val:
        raise VisionLabError(
            f"Parameter '{name}' must be between {min_val} and {max_val} "
            f"(got {value})."
        )
    return value


def validate_odd_kernel_size(value: int, name: str = "kernel size") -> int:
    """Validate that a value is a positive odd integer, as required by many
    OpenCV kernel-based operations (blur, morphology, etc.).

    Parameters
    ----------
    value : int
        Proposed kernel size.
    name : str
        Human-readable name for error messages.

    Returns
    -------
    int
        The validated kernel size.

    Raises
    ------
    VisionLabError
        If the value is not a positive odd integer.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise VisionLabError(f"Parameter '{name}' must be an integer.") from None

    if value <= 0:
        raise VisionLabError(f"Parameter '{name}' must be a positive integer.")

    if value % 2 == 0:
        raise VisionLabError(f"Parameter '{name}' must be odd (got {value}).")

    return value


def validate_choice(value: str, allowed: Iterable[str], name: str) -> str:
    """Validate that a value is one of a set of allowed string choices.

    Parameters
    ----------
    value : str
        The chosen value, e.g. from a Streamlit selectbox.
    allowed : Iterable[str]
        The set/list of valid choices.
    name : str
        Human-readable parameter name, used in the error message.

    Returns
    -------
    str
        The validated value, unchanged.

    Raises
    ------
    VisionLabError
        If the value is not among the allowed choices.
    """
    allowed = list(allowed)
    if value not in allowed:
        raise VisionLabError(
            f"Invalid choice for '{name}': '{value}'. Must be one of: "
            f"{', '.join(allowed)}."
        )
    return value


def validate_k_clusters(k: int, max_k: int = 20) -> int:
    """Validate the number of clusters requested for K-Means.

    Parameters
    ----------
    k : int
        Requested number of clusters.
    max_k : int
        Upper bound to keep computation fast and results meaningful.

    Returns
    -------
    int
        The validated cluster count.

    Raises
    ------
    VisionLabError
        If k is not an integer >= 2 or exceeds max_k.
    """
    try:
        k = int(k)
    except (TypeError, ValueError):
        raise VisionLabError("Number of clusters (k) must be an integer.") from None

    if k < 2:
        raise VisionLabError("Number of clusters (k) must be at least 2.")
    if k > max_k:
        raise VisionLabError(f"Number of clusters (k) must not exceed {max_k}.")
    return k
