# VisionLab — Image Processing and Computer Vision Analysis Tool

A Streamlit web application demonstrating core Computer Vision concepts through simple, understandable implementations. Built as a course project for the VITyarthi Computer Vision flipped-course evaluation.

## Overview

VisionLab lets you upload an image (or use a bundled sample) and interactively apply classical computer vision operations — enhancement, feature detection, and segmentation/clustering — while viewing before/after comparisons and downloading the results. There is no deep learning, database, or external API involved: every operation is a direct, inspectable implementation using OpenCV and NumPy (with scikit-learn used only for K-Means clustering).

## Features

The application is organized into three functional modules, matching the course syllabus:

### 1. Image Enhancement
- Flip (horizontal / vertical / both)
- Contrast adjustment
- Noise removal (Gaussian, median, bilateral filtering)
- Linear transformation
- Log transformation
- Power-law (gamma) transformation
- Histogram equalization (grayscale and color-preserving)
- Basic morphology: erosion, dilation, opening, closing

### 2. Feature Detection
- Canny edge detection
- Corner detection (Shi-Tomasi and Harris methods)
- Hough line detection (probabilistic Hough transform)

### 3. Segmentation & Clustering
- Watershed segmentation (marker-based, distance-transform driven)
- K-Means clustering (color-based image segmentation)

Every operation includes:
- A clear input (uploaded/sample image + adjustable parameters via sliders/dropdowns)
- A clear output (processed image, viewable side-by-side with the original, with histograms where relevant)
- A download button for the processed result
- Input validation and friendly error messages for invalid parameters or corrupted files

## Technologies / Tools Used

- **Python 3**
- **Streamlit** — web application UI
- **OpenCV** (`opencv-python`) — core image processing and computer vision algorithms
- **NumPy** — array operations
- **Matplotlib** — histogram and side-by-side comparison plots
- **scikit-learn** — K-Means clustering only
- **pytest** — unit testing

## Project Structure

```
VisionLab/
├── app.py                       # Streamlit entry point
├── modules/
│   ├── preprocessing.py         # Image loading, validation, resizing
│   ├── enhancement.py           # Image Enhancement module
│   ├── feature_detection.py     # Feature Detection module
│   ├── segmentation.py          # Watershed segmentation
│   └── clustering.py            # K-Means clustering
├── utils/
│   ├── image_utils.py           # Shared image conversion/encoding helpers
│   ├── visualization.py         # Matplotlib plotting helpers
│   └── validation.py            # Input/parameter validation
├── tests/                       # pytest unit tests for all modules
├── sample_data/                 # Bundled sample images
├── requirements.txt
├── README.md
├── statement.md
└── PROJECT_STATE.md
```

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd VisionLab
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

From the project root, run:

```bash
streamlit run app.py
```

This will open the app in your default browser (typically at `http://localhost:8501`). If it doesn't open automatically, copy the URL printed in the terminal into your browser.

### Using the app

1. In the sidebar, choose **Upload image** to upload your own image, or **Use sample image** to pick one of the bundled samples.
2. Once an image is loaded, select a module tab: **Image Enhancement**, **Feature Detection**, or **Segmentation & Clustering**.
3. Choose an operation from the dropdown and adjust its parameters with the sliders/controls.
4. View the original vs. processed comparison (and histogram, where applicable).
5. Click **Download result (PNG)** to save the processed image.

## Testing

Unit tests are written with `pytest` and cover every processing module plus the validation layer.

Run the full test suite from the project root:

```bash
pytest tests/ -v
```

Run a specific test file:

```bash
pytest tests/test_enhancement.py -v
```

The test suite includes:
- `test_validation.py` — input/parameter validation
- `test_enhancement.py` — all Image Enhancement operations
- `test_feature_detection.py` — Canny, corner detection, Hough lines
- `test_segmentation.py` — watershed segmentation
- `test_clustering.py` — K-Means clustering

All tests use synthetically generated images (solid shapes, color blocks, blank images) so they run quickly and deterministically without requiring external image files.

## Known Limitations

- Designed for classical/traditional computer vision techniques only — no deep learning, object detection, or tracking is implemented (by design, per project scope).
- Very large images are automatically downscaled to keep processing responsive; extremely small images (below 2x2 pixels) are rejected.
- Watershed segmentation works best on images with reasonably distinct, separable foreground objects (e.g. coins, cells, seeds) rather than complex natural scenes.

## License

This project was built for academic purposes as part of a VITyarthi course evaluation.
