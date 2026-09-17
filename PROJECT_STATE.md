# PROJECT_STATE.md — VisionLab

## 1. Project Overview

**Name:** VisionLab — Image Processing and Computer Vision Analysis Tool
**Course:** Computer Vision (VITyarthi flipped-course evaluation)
**Type:** Streamlit web application
**Goal:** Demonstrate core Computer Vision syllabus concepts through a simple, reliable, modular tool — no deep learning, no databases, no auth, no APIs, no object tracking/detection, no Naive Bayes/KNN/PCA.

## 2. Syllabus Mapping

| Syllabus Module | Topics | VisionLab Coverage |
|---|---|---|
| Module 1 — Intro & Image Representation | Image formation, flipping, contrast | `enhancement.py`: flip, contrast adjustment |
| Module 2 — Image Enhancement & Transformation | Noise removal, linear/log/power-law transform, morphology, histogram equalization | `enhancement.py`: all transforms, noise removal, morphology (erosion, dilation, opening, closing), histogram equalization |
| Module 3 — Edge/Corner/Hough | Canny, corner detection, Hough line transform | `feature_detection.py`: Canny, corner detection, Hough lines |
| Module 4 — Segmentation & Clustering | Watershed, K-Means | `segmentation.py` (watershed), `clustering.py` (K-Means) |
| Module 5 — Classification/PCA/Detection/Tracking | Naive Bayes, KNN, PCA, object detection/tracking | **Explicitly out of scope** per project rules |

## 3. Functional Modules (VITyarthi requires 3+)

1. **Image Enhancement** — load/represent image, flip, contrast adjustment, noise removal, linear/log/power-law transformation, histogram equalization, morphology (erosion, dilation, opening, closing)
2. **Feature Detection** — Canny edge detection, corner detection, Hough line detection
3. **Segmentation & Clustering** — watershed segmentation, K-Means clustering

Each module: clear input (uploaded image + parameters via Streamlit widgets) → processing → output (processed image + optional plots/histograms), with a straightforward linear workflow (upload → choose module → choose operation → adjust parameters → view result → optionally download).

## 4. Non-Functional Requirements (VITyarthi requires 4+)

1. **Performance** — operate on reasonably sized images (auto-resize/cap large uploads) so operations stay responsive.
2. **Reliability** — deterministic, well-tested classical CV algorithms (OpenCV/NumPy), no flaky external services.
3. **Usability** — simple Streamlit sidebar/tab navigation, sensible parameter defaults, immediate visual feedback.
4. **Maintainability** — clean modular package structure, one responsibility per module, docstrings/comments.
5. **Error handling / Validation** — graceful handling of invalid uploads, unsupported formats, out-of-range parameters (see `utils/validation.py`).
6. **Resource efficiency** — avoid unnecessary copies of images in memory; release/cleanup where relevant.

(6 listed — exceeds the minimum of 4.)

## 5. Architecture

```
VisionLab/
├── app.py                       # Streamlit entry point — UI, navigation, orchestration
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py         # image loading, decoding, resizing, format checks
│   ├── enhancement.py           # flip, contrast, noise removal, linear/log/power-law, hist eq, morphology
│   ├── feature_detection.py     # Canny, corner detection, Hough lines
│   ├── segmentation.py          # watershed segmentation
│   └── clustering.py            # K-Means clustering
├── utils/
│   ├── __init__.py
│   ├── image_utils.py           # shared helpers: color conversions, encode/decode, download buffer
│   ├── visualization.py         # matplotlib/Streamlit plotting helpers (histograms, side-by-side views)
│   └── validation.py            # input validation, parameter bounds checking, error messages
├── tests/                       # unit tests for modules/ and utils/
├── sample_data/                 # a few sample images for demoing without upload
├── requirements.txt
├── README.md
├── statement.md
└── PROJECT_STATE.md
```

**Data flow (workflow):**
1. User opens app → uploads image or picks a sample from `sample_data/`.
2. `preprocessing.py` validates and loads the image (via `validation.py` + `image_utils.py`).
3. User selects a module (Enhancement / Feature Detection / Segmentation & Clustering) via sidebar/tabs.
4. User selects an operation within that module and adjusts parameters via Streamlit widgets.
5. Selected `modules/*` function processes the image (NumPy/OpenCV, scikit-learn only for K-Means).
6. `visualization.py` renders original vs. processed image (and histograms/plots where relevant).
7. User can download the processed result.

This satisfies the "3+ functional modules, clear I/O, logical workflow" requirement and keeps file count in the 5–10 meaningful files range (5 module files + 3 util files + app.py = 9 core files, plus tests).

## 6. Tech Stack

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Streamlit
- Matplotlib (histograms/plots)
- scikit-learn (K-Means only)
- pytest (testing)

## 7. Explicitly Out of Scope (per project rules)

Deep learning, databases, authentication, external APIs, object tracking, complex object detection, Naive Bayes, KNN, PCA, and any other feature not listed in Core Modules above — even though some appear later in the syllabus (Module 5), they are intentionally excluded for this project's scope.

## 8. File Plan / Development Order

1. `PROJECT_STATE.md` ✅ (this file)
2. `requirements.txt`
3. `utils/validation.py`
4. `utils/image_utils.py`
5. `modules/preprocessing.py`
6. `utils/visualization.py`
7. `modules/enhancement.py`
8. `modules/feature_detection.py`
9. `modules/segmentation.py`
10. `modules/clustering.py`
11. `app.py`
12. `tests/` (test files)
13. `README.md`
14. `statement.md`
15. Final documentation: architecture diagram, UML diagrams, workflow diagram, project report (PDF)

Two `__init__.py` files (`modules/__init__.py`, `utils/__init__.py`) will be created alongside their respective first module file, as simple empty/marker files.

## 9. Status Tracking

| File | Status |
|---|---|
| PROJECT_STATE.md | ✅ Done |
| requirements.txt | ⬜ Not started |
| modules/__init__.py | ⬜ Not started |
| utils/__init__.py | ⬜ Not started |
| utils/validation.py | ⬜ Not started |
| utils/image_utils.py | ⬜ Not started |
| modules/preprocessing.py | ⬜ Not started |
| utils/visualization.py | ⬜ Not started |
| modules/enhancement.py | ⬜ Not started |
| modules/feature_detection.py | ⬜ Not started |
| modules/segmentation.py | ⬜ Not started |
| modules/clustering.py | ⬜ Not started |
| app.py | ⬜ Not started |
| tests/ | ⬜ Not started |
| README.md | ⬜ Not started |
| statement.md | ⬜ Not started |
| Diagrams / final report | ⬜ Not started |

## 10. Important Functions/Classes (planned, to be finalized as files are built)

- `preprocessing.load_image(file) -> np.ndarray`
- `preprocessing.resize_if_needed(img, max_dim) -> np.ndarray`
- `enhancement.flip_image(img, mode)`
- `enhancement.adjust_contrast(img, alpha)`
- `enhancement.remove_noise(img, method, ksize)`
- `enhancement.linear_transform(img, alpha, beta)`
- `enhancement.log_transform(img, c)`
- `enhancement.gamma_transform(img, gamma)`
- `enhancement.histogram_equalization(img)`
- `enhancement.morphology_op(img, op, kernel_size, iterations)`
- `feature_detection.canny_edges(img, low, high)`
- `feature_detection.detect_corners(img, method, params)`
- `feature_detection.hough_lines(img, params)`
- `segmentation.watershed_segment(img)`
- `clustering.kmeans_segment(img, k)`
- `validation.validate_image_file(file)`
- `validation.validate_parameter(value, min_val, max_val, name)`
- `image_utils.to_rgb / to_gray / encode_for_download`
- `visualization.show_side_by_side(orig, processed)`
- `visualization.plot_histogram(img)`

## 11. Dependencies (planned for requirements.txt)

```
streamlit
opencv-python
numpy
matplotlib
scikit-learn
pytest
```

## 12. Known Issues

None yet — project just initialized.

## 13. Testing Status

Not started. Plan: pytest unit tests per module in `tests/`, covering valid inputs, edge cases (e.g., grayscale vs. color input, very small images), and invalid input handling (via `validation.py`).

## 14. Next File

`requirements.txt`

Waiting for: `GO FOR NEXT FILE: requirements.txt`
