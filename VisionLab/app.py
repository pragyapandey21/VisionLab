"""
app.py
------
VisionLab — Image Processing and Computer Vision Analysis Tool.

Streamlit entry point. This file is intentionally "thin": it handles
UI layout, widget wiring, and orchestration only. All actual image
processing lives in `modules/*`, all shared helpers in `utils/*`.

Workflow:
  1. User uploads an image or picks a bundled sample.
  2. `modules.preprocessing` validates/loads/resizes it.
  3. User picks a functional module (tab) and an operation within it.
  4. User adjusts parameters via sidebar/inline widgets.
  5. The corresponding `modules.*` function processes the image.
  6. `utils.visualization` renders an original-vs-processed comparison
     (and a histogram where relevant).
  7. User can download the processed result.
"""

from __future__ import annotations

import os

import streamlit as st

from modules import clustering, enhancement, feature_detection, preprocessing, segmentation
from utils.image_utils import encode_for_download, get_image_dimensions, to_bgr, to_gray, to_rgb
from utils.validation import VisionLabError
from utils.visualization import (
    close_figure,
    plot_histogram_comparison,
    show_side_by_side,
)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")

st.set_page_config(page_title="VisionLab", layout="wide")


# ---------------------------------------------------------------------------
# Image loading (sidebar)
# ---------------------------------------------------------------------------

def load_input_image():
    """Render the sidebar image-source controls and return a loaded image,
    or None if nothing is loaded yet.

    Returns
    -------
    np.ndarray or None
        A validated, ready-to-process BGR image, or None.
    """
    st.sidebar.header("1. Image Source")
    source = st.sidebar.radio("Choose input", ["Upload image", "Use sample image"])

    if source == "Upload image":
        uploaded = st.sidebar.file_uploader(
            "Upload an image", type=["png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"]
        )
        if uploaded is None:
            return None
        try:
            file_bytes = uploaded.getvalue()
            return preprocessing.prepare_uploaded_image(
                uploaded.name, file_bytes, len(file_bytes)
            )
        except VisionLabError as e:
            st.sidebar.error(str(e))
            return None

    # Sample image source
    samples = preprocessing.get_supported_sample_images(SAMPLE_DIR)
    if not samples:
        st.sidebar.warning("No sample images found in sample_data/.")
        return None

    chosen = st.sidebar.selectbox("Choose a sample", samples)
    try:
        return preprocessing.load_sample_image(os.path.join(SAMPLE_DIR, chosen))
    except VisionLabError as e:
        st.sidebar.error(str(e))
        return None


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------

def render_result(original_bgr, processed, caption: str, show_histogram: bool = False):
    """Render an original-vs-processed comparison (and optional histogram
    comparison) plus a download button for the processed image.

    Parameters
    ----------
    original_bgr : np.ndarray
        The original image, in BGR (as loaded).
    processed : np.ndarray
        The processed image, in BGR or grayscale (as returned by a
        `modules.*` function).
    caption : str
        Short label describing the operation, used in the plot title and
        download filename.
    show_histogram : bool
        Whether to also show a before/after histogram comparison.
    """
    original_display = to_rgb(original_bgr)
    processed_display = to_rgb(processed) if processed.ndim == 3 else processed

    fig = show_side_by_side(
        original_display, processed_display, "Original", caption
    )
    st.pyplot(fig)
    close_figure(fig)

    if show_histogram:
        hist_fig = plot_histogram_comparison(
            original_display, processed_display, "Original Histogram", f"{caption} Histogram"
        )
        st.pyplot(hist_fig)
        close_figure(hist_fig)

    # Prepare a BGR/gray version (not RGB) for correct-color download.
    downloadable = processed if processed.ndim == 2 else processed
    try:
        data = encode_for_download(downloadable, ".png")
        st.download_button(
            "Download result (PNG)",
            data=data,
            file_name=f"{caption.lower().replace(' ', '_')}.png",
            mime="image/png",
        )
    except VisionLabError as e:
        st.warning(f"Could not prepare download: {e}")


# ---------------------------------------------------------------------------
# Module tabs
# ---------------------------------------------------------------------------

def enhancement_tab(img):
    st.subheader("Image Enhancement")
    operation = st.selectbox(
        "Operation",
        [
            "Flip",
            "Contrast Adjustment",
            "Noise Removal",
            "Linear Transformation",
            "Log Transformation",
            "Power-Law (Gamma) Transformation",
            "Histogram Equalization",
            "Morphology",
        ],
    )

    try:
        if operation == "Flip":
            mode = st.radio("Flip mode", ["horizontal", "vertical", "both"])
            result = enhancement.flip_image(img, mode)
            render_result(img, result, "Flipped")

        elif operation == "Contrast Adjustment":
            alpha = st.slider("Contrast (alpha)", 0.1, 3.0, 1.0, 0.1)
            beta = st.slider("Brightness (beta)", -100, 100, 0, 1)
            result = enhancement.adjust_contrast(img, alpha, beta)
            render_result(img, result, "Contrast Adjusted", show_histogram=True)

        elif operation == "Noise Removal":
            method = st.selectbox("Method", ["gaussian", "median", "bilateral"])
            ksize = st.slider("Kernel size (odd)", 3, 15, 5, 2)
            result = enhancement.remove_noise(img, method, ksize)
            render_result(img, result, "Denoised")

        elif operation == "Linear Transformation":
            alpha = st.slider("Alpha (gain)", 0.0, 5.0, 1.0, 0.1)
            beta = st.slider("Beta (offset)", -255, 255, 0, 5)
            result = enhancement.linear_transform(img, alpha, beta)
            render_result(img, result, "Linear Transformed", show_histogram=True)

        elif operation == "Log Transformation":
            c = st.slider("Constant (c)", 0.1, 5.0, 1.0, 0.1)
            result = enhancement.log_transform(img, c)
            render_result(img, result, "Log Transformed", show_histogram=True)

        elif operation == "Power-Law (Gamma) Transformation":
            gamma = st.slider("Gamma", 0.1, 5.0, 1.0, 0.1)
            result = enhancement.gamma_transform(img, gamma)
            render_result(img, result, "Gamma Transformed", show_histogram=True)

        elif operation == "Histogram Equalization":
            result = enhancement.histogram_equalization(img)
            render_result(img, result, "Histogram Equalized", show_histogram=True)

        elif operation == "Morphology":
            op = st.selectbox("Operation", ["erosion", "dilation", "opening", "closing"])
            kernel_size = st.slider("Kernel size (odd)", 1, 15, 3, 2)
            iterations = st.slider("Iterations", 1, 10, 1)
            result = enhancement.morphology_op(img, op, kernel_size, iterations)
            render_result(img, result, f"Morphology: {op.capitalize()}")

    except VisionLabError as e:
        st.error(str(e))


def feature_detection_tab(img):
    st.subheader("Feature Detection")
    operation = st.selectbox(
        "Operation", ["Canny Edge Detection", "Corner Detection", "Hough Line Detection"]
    )

    try:
        if operation == "Canny Edge Detection":
            low = st.slider("Low threshold", 0, 500, 50, 5)
            high = st.slider("High threshold", 0, 500, 150, 5)
            result = feature_detection.canny_edges(img, low, high)
            render_result(img, result, "Canny Edges")

        elif operation == "Corner Detection":
            method = st.selectbox("Method", ["shi-tomasi", "harris"])
            max_corners = st.slider("Max corners", 1, 500, 100, 1)
            quality = st.slider("Quality level", 0.001, 1.0, 0.01, 0.001)
            min_dist = st.slider("Min distance", 1, 100, 10, 1)
            result = feature_detection.detect_corners(img, method, max_corners, quality, min_dist)
            render_result(img, result, f"Corners ({method})")

        elif operation == "Hough Line Detection":
            canny_low = st.slider("Canny low threshold", 0, 500, 50, 5)
            canny_high = st.slider("Canny high threshold", 0, 500, 150, 5)
            threshold = st.slider("Vote threshold", 1, 500, 100, 1)
            min_len = st.slider("Min line length", 1, 1000, 50, 1)
            max_gap = st.slider("Max line gap", 0, 100, 10, 1)
            result = feature_detection.hough_lines(
                img, canny_low, canny_high, threshold, min_len, max_gap
            )
            render_result(img, result, "Hough Lines")

    except VisionLabError as e:
        st.error(str(e))


def segmentation_clustering_tab(img):
    st.subheader("Segmentation & Clustering")
    operation = st.selectbox("Operation", ["Watershed Segmentation", "K-Means Clustering"])

    try:
        if operation == "Watershed Segmentation":
            fg_ratio = st.slider("Foreground threshold ratio", 0.1, 0.9, 0.5, 0.05)
            kernel_size = st.slider("Morphology kernel size", 1, 15, 3, 1)
            result = segmentation.watershed_segment(img, fg_ratio, kernel_size)
            render_result(img, result, "Watershed Segmented")

        elif operation == "K-Means Clustering":
            k = st.slider("Number of clusters (k)", 2, 20, 4, 1)
            max_iter = st.slider("Max iterations", 10, 500, 100, 10)
            result = clustering.kmeans_segment(img, k, max_iter)
            render_result(img, result, f"K-Means (k={k})")

    except VisionLabError as e:
        st.error(str(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("VisionLab")
    st.caption("Image Processing and Computer Vision Analysis Tool")

    img = load_input_image()

    if img is None:
        st.info("Upload an image or choose a sample from the sidebar to get started.")
        return

    width, height = get_image_dimensions(img)
    st.sidebar.success(f"Image loaded: {width} x {height} px")

    st.sidebar.header("2. Module")
    tab1, tab2, tab3 = st.tabs(
        ["Image Enhancement", "Feature Detection", "Segmentation & Clustering"]
    )

    with tab1:
        enhancement_tab(img)
    with tab2:
        feature_detection_tab(img)
    with tab3:
        segmentation_clustering_tab(img)


if __name__ == "__main__":
    main()
