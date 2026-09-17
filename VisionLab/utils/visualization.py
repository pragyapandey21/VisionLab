"""
visualization.py
-----------------
Matplotlib-based plotting helpers for VisionLab.

These functions build `matplotlib.figure.Figure` objects that `app.py`
passes straight to `st.pyplot()`. Keeping plotting logic here (rather
than inline in `app.py` or the CV modules) keeps those files focused on
their own responsibilities and makes the plots reusable/testable.

All functions expect images in the color order produced by
`utils.image_utils.to_rgb` / `to_gray` (i.e. already display-ready),
NOT raw BGR from OpenCV.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def show_side_by_side(
    original: np.ndarray,
    processed: np.ndarray,
    original_title: str = "Original",
    processed_title: str = "Processed",
) -> Figure:
    """Build a two-panel figure comparing an original and processed image.

    Parameters
    ----------
    original : np.ndarray
        Display-ready original image (RGB or grayscale).
    processed : np.ndarray
        Display-ready processed image (RGB or grayscale).
    original_title, processed_title : str
        Titles for each panel.

    Returns
    -------
    Figure
        A Matplotlib figure with the two images side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    cmap_original = "gray" if original.ndim == 2 else None
    cmap_processed = "gray" if processed.ndim == 2 else None

    axes[0].imshow(original, cmap=cmap_original)
    axes[0].set_title(original_title)
    axes[0].axis("off")

    axes[1].imshow(processed, cmap=cmap_processed)
    axes[1].set_title(processed_title)
    axes[1].axis("off")

    fig.tight_layout()
    return fig


def plot_histogram(
    img: np.ndarray,
    title: str = "Histogram",
) -> Figure:
    """Plot the intensity histogram of an image.

    For grayscale images, plots a single histogram. For color (RGB)
    images, plots one histogram line per channel.

    Parameters
    ----------
    img : np.ndarray
        Display-ready image (RGB or grayscale), dtype uint8.
    title : str
        Title for the plot.

    Returns
    -------
    Figure
        A Matplotlib figure containing the histogram.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    if img.ndim == 2:
        ax.hist(img.ravel(), bins=256, range=(0, 256), color="gray")
    else:
        colors = ("red", "green", "blue")
        for i, color in enumerate(colors):
            ax.hist(
                img[:, :, i].ravel(),
                bins=256,
                range=(0, 256),
                color=color,
                alpha=0.5,
                label=color.capitalize(),
            )
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def plot_histogram_comparison(
    original: np.ndarray,
    processed: np.ndarray,
    original_title: str = "Original Histogram",
    processed_title: str = "Processed Histogram",
) -> Figure:
    """Plot original vs. processed histograms side by side.

    Useful for demonstrating the effect of histogram equalization or
    contrast/transformation operations.

    Parameters
    ----------
    original, processed : np.ndarray
        Display-ready images (RGB or grayscale), dtype uint8.
    original_title, processed_title : str
        Titles for each subplot.

    Returns
    -------
    Figure
        A Matplotlib figure with two histogram subplots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, image, title in (
        (axes[0], original, original_title),
        (axes[1], processed, processed_title),
    ):
        if image.ndim == 2:
            ax.hist(image.ravel(), bins=256, range=(0, 256), color="gray")
        else:
            colors = ("red", "green", "blue")
            for i, color in enumerate(colors):
                ax.hist(
                    image[:, :, i].ravel(),
                    bins=256,
                    range=(0, 256),
                    color=color,
                    alpha=0.5,
                    label=color.capitalize(),
                )
            ax.legend()
        ax.set_title(title)
        ax.set_xlabel("Pixel intensity")
        ax.set_ylabel("Frequency")

    fig.tight_layout()
    return fig


def show_single_image(img: np.ndarray, title: str = "Image") -> Figure:
    """Build a single-panel figure for displaying one image.

    Useful for showing an intermediate result (e.g. a K-Means cluster
    map or a watershed marker overlay) without a comparison.

    Parameters
    ----------
    img : np.ndarray
        Display-ready image (RGB or grayscale).
    title : str
        Title for the plot.

    Returns
    -------
    Figure
        A Matplotlib figure with the single image.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = "gray" if img.ndim == 2 else None
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def close_figure(fig: Optional[Figure]) -> None:
    """Explicitly close a Matplotlib figure to free memory.

    Streamlit apps that generate many figures across reruns can leak
    memory if figures aren't closed; `app.py` should call this after
    passing a figure to `st.pyplot()` when it's no longer needed.

    Parameters
    ----------
    fig : Figure or None
        The figure to close. Safe to call with None.
    """
    if fig is not None:
        plt.close(fig)
