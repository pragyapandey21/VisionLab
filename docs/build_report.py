"""
build_report.py
----------------
One-off script that generates docs/VisionLab_Project_Report.pdf using
reportlab. Not part of the VisionLab application itself — this is
tooling used to produce the required project report deliverable.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle,
    ListFlowable, ListItem,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

ASSETS = "docs/report_assets"
OUT = "docs/VisionLab_Project_Report.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitle", fontSize=26, leading=32, alignment=TA_CENTER, spaceAfter=20, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="CoverSub", fontSize=14, leading=20, alignment=TA_CENTER, spaceAfter=10))
styles.add(ParagraphStyle(name="H1", fontSize=17, leading=22, spaceBefore=18, spaceAfter=10, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="H2", fontSize=13, leading=18, spaceBefore=12, spaceAfter=6, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="Body", fontSize=10.5, leading=15, spaceAfter=8, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="Caption", fontSize=9, leading=12, alignment=TA_CENTER, textColor=colors.grey, spaceAfter=14))

story = []

def h1(text):
    story.append(Paragraph(text, styles["H1"]))

def h2(text):
    story.append(Paragraph(text, styles["H2"]))

def p(text):
    story.append(Paragraph(text, styles["Body"]))

def bullets(items):
    story.append(ListFlowable(
        [ListItem(Paragraph(i, styles["Body"])) for i in items],
        bulletType="bullet", leftIndent=18,
    ))

def figure(path, caption, width=5.8*inch):
    img = Image(path)
    aspect = img.imageHeight / float(img.imageWidth)
    img.drawWidth = width
    img.drawHeight = width * aspect
    story.append(img)
    story.append(Paragraph(caption, styles["Caption"]))

# ---------------------------------------------------------------------------
# 1. Cover Page
# ---------------------------------------------------------------------------
story.append(Spacer(1, 1.5*inch))
story.append(Paragraph("VisionLab", styles["CoverTitle"]))
story.append(Paragraph("Image Processing and Computer Vision Analysis Tool", styles["CoverSub"]))
story.append(Spacer(1, 0.4*inch))
story.append(Paragraph("Project Report", styles["CoverSub"]))
story.append(Spacer(1, 1.2*inch))
story.append(Paragraph("Course: Computer Vision (VITyarthi Flipped Course Evaluation)", styles["CoverSub"]))
story.append(Paragraph("Submitted by: Dev", styles["CoverSub"]))
story.append(Paragraph("B.Tech CSE", styles["CoverSub"]))
story.append(PageBreak())

# ---------------------------------------------------------------------------
# 2. Introduction
# ---------------------------------------------------------------------------
h1("2. Introduction")
p("VisionLab is an interactive Streamlit application built to demonstrate the core Computer Vision "
  "concepts covered across the course syllabus. Rather than isolated notebook snippets, VisionLab brings "
  "image enhancement, feature detection, and segmentation/clustering algorithms together into a single, "
  "cohesive tool that lets a user upload an image and immediately see the effect of each classical CV "
  "algorithm, with adjustable parameters and side-by-side before/after comparisons.")
p("The project deliberately restricts itself to classical, well-understood computer vision techniques "
  "implemented directly with OpenCV and NumPy (with scikit-learn used only for K-Means clustering). No "
  "deep learning, database, authentication, or external API is used, keeping the implementation "
  "transparent, inspectable, and appropriate in scope for a course project.")

# ---------------------------------------------------------------------------
# 3. Problem Statement
# ---------------------------------------------------------------------------
h1("3. Problem Statement")
p("Students learning Computer Vision typically encounter core algorithms in isolated lecture demos or "
  "notebook snippets, with no single tool that lets them apply these algorithms interactively to their own "
  "images and build intuition for how parameters affect results. VisionLab addresses this by providing a "
  "self-contained, interactive application covering the syllabus's classical CV techniques, usable without "
  "writing any code.")

# ---------------------------------------------------------------------------
# 4. Functional Requirements
# ---------------------------------------------------------------------------
h1("4. Functional Requirements")
p("VisionLab implements three major functional modules, each with a clear input/output structure and a "
  "logical, linear user workflow (upload/select image &rarr; choose module &rarr; choose operation &rarr; "
  "adjust parameters &rarr; view result &rarr; download).")

h2("4.1 Image Enhancement")
bullets([
    "Flip (horizontal / vertical / both)",
    "Contrast adjustment",
    "Noise removal (Gaussian, median, bilateral)",
    "Linear transformation",
    "Log transformation",
    "Power-law (gamma) transformation",
    "Histogram equalization (grayscale and color-preserving)",
    "Basic morphology: erosion, dilation, opening, closing",
])

h2("4.2 Feature Detection")
bullets([
    "Canny edge detection",
    "Corner detection (Shi-Tomasi and Harris)",
    "Hough line detection (probabilistic Hough transform)",
])

h2("4.3 Segmentation &amp; Clustering")
bullets([
    "Watershed segmentation (marker-based, distance-transform driven)",
    "K-Means clustering (color-based image segmentation)",
])

# ---------------------------------------------------------------------------
# 5. Non-Functional Requirements
# ---------------------------------------------------------------------------
h1("5. Non-Functional Requirements")
nfr_data = [
    ["Requirement", "How VisionLab addresses it"],
    ["Performance", "Uploaded images are automatically downscaled beyond a safe dimension so every operation stays responsive."],
    ["Reliability", "All algorithms are deterministic, well-established OpenCV/NumPy/scikit-learn implementations."],
    ["Usability", "Simple sidebar + tabbed navigation, sensible parameter defaults, immediate visual feedback."],
    ["Maintainability", "Clean modular package structure — one responsibility per file, docstrings throughout."],
    ["Error handling / Validation", "A dedicated validation layer checks every file and parameter before processing; invalid input produces a friendly message instead of a crash."],
    ["Resource efficiency", "Images are processed as in-memory NumPy arrays with defensive copies only where needed, and Matplotlib figures are explicitly closed after rendering."],
]
t = Table(nfr_data, colWidths=[1.7*inch, 4.3*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1e3a8a")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f1f5f9")]),
]))
story.append(t)
story.append(Spacer(1, 8))

# ---------------------------------------------------------------------------
# 6. System Architecture
# ---------------------------------------------------------------------------
h1("6. System Architecture")
p("VisionLab follows a simple three-layer architecture: a Streamlit presentation layer (app.py), a set of "
  "CV processing modules (one file per functional area), and a shared utilities layer (validation, image "
  "I/O, and visualization) used by both the UI and the processing modules.")
figure(f"{ASSETS}/00_architecture.png", "Figure 1: VisionLab system architecture")

# ---------------------------------------------------------------------------
# 7. Design Diagrams
# ---------------------------------------------------------------------------
h1("7. Design Diagrams")
p("Full Use Case, Class/Component, and Sequence diagrams (in Mermaid format, rendering natively on "
  "GitHub) are provided in <b>docs/diagrams.md</b> in the project repository, alongside the Process Flow / "
  "Workflow diagram. A summary of each is given below.")

h2("7.1 Use Case Diagram (summary)")
p("The single actor, <i>User</i>, can: upload an image, select a sample image, apply an enhancement "
  "operation, apply feature detection, apply segmentation/clustering, view the before/after comparison, "
  "view a histogram comparison, and download the processed result. Every processing use case includes the "
  "\"view before/after comparison\" use case.")

h2("7.2 Class / Component Diagram (summary)")
p("VisionLab is implemented as a functional module library rather than a class hierarchy. The key "
  "components are: <b>app</b> (UI orchestration), <b>preprocessing</b> (image loading), <b>enhancement</b>, "
  "<b>feature_detection</b>, <b>segmentation</b>, and <b>clustering</b> (the CV modules), and the shared "
  "<b>validation</b>, <b>image_utils</b>, and <b>visualization</b> utilities that every module depends on.")

h2("7.3 Sequence Diagram (summary)")
p("A typical interaction — e.g. applying gamma transformation — flows as: the UI passes the uploaded file "
  "to <i>preprocessing</i>, which validates and decodes it; the UI then passes the image and the user's "
  "gamma value to <i>enhancement.gamma_transform</i>, which validates the parameter, applies a LUT-based "
  "gamma correction, and returns the result; the UI renders a side-by-side comparison via "
  "<i>visualization</i> and offers a download.")

h2("7.4 Process Flow / Workflow Diagram (summary)")
p("The end-to-end workflow is: choose an image source (upload or sample) &rarr; validate/load &rarr; "
  "choose a module and operation &rarr; adjust parameters &rarr; process &rarr; display comparison "
  "(and histogram, where relevant) &rarr; download. Invalid input or out-of-range parameters loop back to "
  "the parameter-adjustment step with a clear error message rather than crashing the app.")

story.append(PageBreak())

# ---------------------------------------------------------------------------
# 8. Design Decisions & Rationale
# ---------------------------------------------------------------------------
h1("8. Design Decisions &amp; Rationale")
bullets([
    "<b>Functional modules over classes:</b> Each CV operation is a standalone, pure function (input image + "
    "parameters &rarr; output image). This keeps every operation independently testable and easy to reason "
    "about, and matches how the syllabus presents each algorithm as a discrete technique.",
    "<b>Centralized validation:</b> A single validation.py module owns all parameter-range and file-validation "
    "logic, raising one custom VisionLabError type. This means every module and the UI can handle errors "
    "uniformly, and validation rules are defined in exactly one place.",
    "<b>Defensive copies over mutation:</b> Every processing function returns a new array rather than mutating "
    "its input in place. This matters specifically because Streamlit reruns the entire script on every "
    "widget interaction, so cached/session images must never be silently altered.",
    "<b>YCrCb histogram equalization for color images:</b> Rather than equalizing each BGR channel "
    "independently (which distorts color balance), the luminance channel alone is equalized in YCrCb space, "
    "preserving the image's original colors while still improving contrast.",
    "<b>scikit-learn confined to clustering.py:</b> Per the project's tech-stack constraints, scikit-learn is "
    "used only for K-Means; all other algorithms use OpenCV/NumPy directly, keeping the dependency footprint "
    "minimal and each module's purpose unambiguous.",
    "<b>Automatic image downscaling:</b> Images above a safe maximum dimension are downscaled before "
    "processing (preserving aspect ratio) to keep the app responsive on large uploads without requiring the "
    "user to resize images themselves.",
])

# ---------------------------------------------------------------------------
# 9. Implementation Details
# ---------------------------------------------------------------------------
h1("9. Implementation Details")
p("The codebase is organized into three layers, described below.")

h2("9.1 Utilities (utils/)")
bullets([
    "<b>validation.py</b> — VisionLabError exception plus validators for file extension, file size, decoded "
    "image arrays, numeric parameters, odd kernel sizes, string choices, and K-Means cluster count.",
    "<b>image_utils.py</b> — decoding uploaded bytes to a BGR NumPy array, safe downscaling, RGB/gray/BGR "
    "color-space conversions, PNG/JPG encoding for download, and image dimension lookup.",
    "<b>visualization.py</b> — Matplotlib figure builders: side-by-side comparison, single and comparative "
    "histograms, single-image display, and explicit figure cleanup to avoid memory buildup across Streamlit "
    "reruns.",
])

h2("9.2 Processing Modules (modules/)")
bullets([
    "<b>preprocessing.py</b> — ties validation and image_utils together into the load &rarr; validate &rarr; "
    "resize pipeline used by both file uploads and bundled sample images.",
    "<b>enhancement.py</b> — flip, contrast, noise removal (three filter types), linear/log/gamma point "
    "transforms, histogram equalization, and morphology (four operations), each validating its own parameters.",
    "<b>feature_detection.py</b> — Canny edge detection returns a binary edge map directly; corner detection "
    "(Shi-Tomasi via goodFeaturesToTrack, Harris via cornerHarris) and Hough line detection draw their "
    "results onto a copy of the original image so results are immediately viewable.",
    "<b>segmentation.py</b> — implements the standard OpenCV marker-based watershed recipe: Otsu threshold, "
    "morphological opening to remove noise, distance transform to find sure foreground, connected components "
    "for initial markers, and cv2.watershed to grow boundaries.",
    "<b>clustering.py</b> — K-Means over the image's (B, G, R) pixel values via scikit-learn, offered both as "
    "a recolored segmentation (kmeans_segment) and a raw label map (kmeans_cluster_map).",
])

h2("9.3 Application (app.py)")
p("app.py is intentionally thin: it wires Streamlit widgets to the module functions above and renders results "
  "via the visualization helpers. It contains no image-processing logic of its own, keeping the UI layer and "
  "the algorithmic layer cleanly separated.")

story.append(PageBreak())

# ---------------------------------------------------------------------------
# 10. Screenshots / Results
# ---------------------------------------------------------------------------
h1("10. Screenshots / Results")
p("The figures below were generated directly from VisionLab's own modules (not mocked), run against a "
  "synthetic test image containing shapes, gradients, and injected noise, to demonstrate each operation's "
  "real output.")

figure(f"{ASSETS}/01_gamma.png", "Figure 2: Gamma transformation (Image Enhancement)")
figure(f"{ASSETS}/02_denoise.png", "Figure 3: Median noise removal (Image Enhancement)")
figure(f"{ASSETS}/03_histeq.png", "Figure 4: Histogram equalization (Image Enhancement)")
figure(f"{ASSETS}/04_morph.png", "Figure 5: Morphological closing (Image Enhancement)")
story.append(PageBreak())
figure(f"{ASSETS}/05_canny.png", "Figure 6: Canny edge detection (Feature Detection)")
figure(f"{ASSETS}/06_corners.png", "Figure 7: Shi-Tomasi corner detection (Feature Detection)")
figure(f"{ASSETS}/07_hough.png", "Figure 8: Hough line detection (Feature Detection)")
story.append(PageBreak())
figure(f"{ASSETS}/08_watershed.png", "Figure 9: Watershed segmentation (Segmentation &amp; Clustering)")
figure(f"{ASSETS}/09_kmeans.png", "Figure 10: K-Means clustering, k=4 (Segmentation &amp; Clustering)")

story.append(PageBreak())

# ---------------------------------------------------------------------------
# 11. Testing Approach
# ---------------------------------------------------------------------------
h1("11. Testing Approach")
p("VisionLab uses pytest for unit testing, with a dedicated test file per module. Tests use synthetically "
  "generated images (solid shapes, color blocks, blank/noise images) rather than external files, so the "
  "suite runs quickly and deterministically in any environment.")

test_data = [
    ["Test file", "Tests", "Focus"],
    ["test_validation.py", "27", "File/parameter validation, all error paths"],
    ["test_enhancement.py", "36", "All 8 enhancement operations: shape/dtype checks, identity cases, edge cases"],
    ["test_feature_detection.py", "17", "Canny, both corner-detection methods, Hough lines, on shapes vs. blank images"],
    ["test_segmentation.py", "7", "Watershed on touching shapes, blank-image edge case, no-mutation check"],
    ["test_clustering.py", "9", "K-Means cluster-count recovery, reproducibility, grayscale input"],
    ["Total", "96", "All passing"],
]
t2 = Table(test_data, colWidths=[1.9*inch, 0.8*inch, 3.3*inch])
t2.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1e3a8a")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTNAME", (0,-1), (-1,-1), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("ROWBACKGROUNDS", (0,1), (-1,-2), [colors.white, colors.HexColor("#f1f5f9")]),
]))
story.append(t2)
story.append(Spacer(1, 8))
p("Each test module covers: correct output shape/dtype for valid input, at least one behaviorally meaningful "
  "assertion (e.g. erosion shrinks a white region, K-Means on a 3-color image recovers at most 3 unique "
  "colors, Hough lines are found on a line image but not a blank one), grayscale-input handling, and "
  "rejection of invalid parameters via VisionLabError.")

# ---------------------------------------------------------------------------
# 12. Challenges Faced
# ---------------------------------------------------------------------------
h1("12. Challenges Faced")
bullets([
    "<b>Watershed on a blank image:</b> initial assumption was that a blank/uniform image would produce zero "
    "boundary pixels. In practice, cv2.watershed always marks the image border as boundary (-1) by "
    "convention, even with no real foreground. The corresponding test was corrected to check for a graceful, "
    "crash-free result and correct output shape rather than an exact zero-boundary count &mdash; a good "
    "reminder to verify assumptions against the actual library behavior rather than intuition.",
    "<b>Keeping color information through histogram equalization:</b> naively equalizing each BGR channel "
    "independently shifts the color balance of the image. Equalizing only the luminance (Y) channel in "
    "YCrCb space solves this while still achieving the contrast improvement.",
    "<b>Balancing flexibility with input safety:</b> every user-adjustable parameter (kernel sizes, "
    "thresholds, cluster counts) needed a sensible allowed range that is wide enough to be useful but narrow "
    "enough to avoid pathological runtimes (e.g. capping K-Means cluster count and max Hough vote threshold).",
])

# ---------------------------------------------------------------------------
# 13. Learnings & Key Takeaways
# ---------------------------------------------------------------------------
h1("13. Learnings &amp; Key Takeaways")
bullets([
    "A small, centralized validation layer dramatically simplifies error handling across an entire "
    "application &mdash; every module gets consistent, user-friendly error messages for free.",
    "Classical computer vision techniques (thresholding, distance transforms, connected components) remain "
    "powerful and interpretable building blocks, and watershed segmentation in particular showed how several "
    "simple steps compose into a much more capable algorithm.",
    "Writing tests against synthetic, purpose-built images (rather than trying to test against arbitrary "
    "real photos) makes it possible to assert specific, meaningful outcomes (e.g. \"exactly 3 colors "
    "recovered\") instead of only checking that code runs without crashing.",
    "Streamlit's rerun-on-every-interaction model has real implications for state management &mdash; "
    "processing functions must never mutate their input in place, or a later interaction can silently corrupt "
    "an earlier result.",
])

# ---------------------------------------------------------------------------
# 14. Future Enhancements
# ---------------------------------------------------------------------------
h1("14. Future Enhancements")
bullets([
    "Add a batch-processing mode to apply the same operation across multiple uploaded images at once.",
    "Allow chaining multiple operations together (e.g. denoise &rarr; then Canny) within a single pipeline, "
    "with the ability to save and reuse a pipeline configuration.",
    "Add additional segmentation techniques (e.g. GrabCut, mean-shift) to broaden the Segmentation &amp; "
    "Clustering module.",
    "Add a side-by-side parameter-sweep view (e.g. show gamma at several values at once) for faster visual "
    "comparison.",
])

# ---------------------------------------------------------------------------
# 15. References
# ---------------------------------------------------------------------------
h1("15. References")
bullets([
    "OpenCV Documentation &mdash; https://docs.opencv.org/",
    "NumPy Documentation &mdash; https://numpy.org/doc/",
    "Streamlit Documentation &mdash; https://docs.streamlit.io/",
    "scikit-learn KMeans Documentation &mdash; https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html",
    "Matplotlib Documentation &mdash; https://matplotlib.org/stable/",
    "OpenCV Watershed Tutorial (Image Segmentation) &mdash; https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html",
    "VITyarthi Computer Vision course materials (syllabus reference for module mapping).",
])

doc = SimpleDocTemplate(
    OUT, pagesize=letter,
    topMargin=0.75*inch, bottomMargin=0.75*inch,
    leftMargin=0.85*inch, rightMargin=0.85*inch,
)
doc.build(story)
print("Report built:", OUT)
