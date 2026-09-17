# Problem Statement

Students learning Computer Vision typically encounter core algorithms — image enhancement, edge/corner detection, segmentation, clustering — in isolated lecture demos or notebook snippets. There is rarely a single, cohesive tool that lets a learner apply these algorithms interactively to their own images, compare results side-by-side, and build intuition for how each parameter affects the outcome.

**VisionLab** addresses this gap: it is a self-contained, interactive application that brings together the classical computer vision techniques covered in the course syllabus into one simple, understandable tool, allowing hands-on experimentation without writing any code.

## Scope of the Project

VisionLab implements three functional modules covering syllabus Modules 1–4:

1. **Image Enhancement** — image loading/representation, flipping, contrast adjustment, noise removal, linear/log/power-law transformations, histogram equalization, and basic morphology (erosion, dilation, opening, closing).
2. **Feature Detection** — Canny edge detection, corner detection (Shi-Tomasi and Harris), and Hough line detection.
3. **Segmentation & Clustering** — watershed segmentation and K-Means clustering.

**Explicitly out of scope** (per project design decisions, even though later syllabus modules touch on them): deep learning, databases, user authentication, external APIs, object tracking, complex object detection, Naive Bayes, KNN, and PCA. The goal is a focused, classical-CV demonstration tool, not a general-purpose ML platform.

## Target Users

- Students studying Computer Vision who want an interactive way to explore how classical algorithms behave on real images.
- Anyone wanting a quick, no-code way to apply standard image processing operations (denoising, edge detection, segmentation) to an image.
- Evaluators/reviewers of this project who want to verify each syllabus concept has a working, testable implementation.

## High-Level Features

- Upload any image (PNG, JPG, JPEG, BMP, TIFF, WEBP) or choose from bundled sample images.
- Interactive parameter controls (sliders, dropdowns) for every operation, with sensible defaults.
- Side-by-side original vs. processed image comparison for every operation.
- Histogram comparison views for operations that affect pixel intensity distribution (contrast, transformations, histogram equalization).
- One-click download of any processed result as a PNG file.
- Robust input validation: friendly error messages for unsupported file types, oversized files, corrupted images, and out-of-range parameters — the app never crashes on bad input.
- A modular codebase (separate modules for enhancement, feature detection, segmentation, and clustering) with an accompanying pytest test suite covering all core logic.
