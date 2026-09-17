# VisionLab — Design Diagrams

This document contains the design artefacts required by the VITyarthi project guidelines: System Architecture, Process Flow/Workflow, Use Case, Class/Component, and Sequence diagrams. All diagrams are written in Mermaid syntax, which renders natively on GitHub and most Markdown viewers.

No database/ER diagram is included, as VisionLab uses no persistent storage — all processing happens in memory on the uploaded image.

## 1. System Architecture Diagram

Shows the high-level layering of the application: the Streamlit UI layer, the CV processing modules, and the shared utility layer.

```mermaid
graph TD
    subgraph UI["Presentation Layer"]
        A[app.py - Streamlit UI]
    end

    subgraph Modules["Processing Modules"]
        B[preprocessing.py]
        C[enhancement.py]
        D[feature_detection.py]
        E[segmentation.py]
        F[clustering.py]
    end

    subgraph Utils["Shared Utilities"]
        G[validation.py]
        H[image_utils.py]
        I[visualization.py]
    end

    A --> B
    A --> C
    A --> D
    A --> E
    A --> F
    A --> I

    B --> G
    B --> H
    C --> G
    D --> G
    D --> H
    E --> H
    F --> H

    C -.uses.-> H
    E -.uses.-> H
```

## 2. Process Flow / Workflow Diagram

Shows the end-to-end user journey from opening the app to downloading a result.

```mermaid
flowchart TD
    Start([User opens VisionLab]) --> Source{Choose image source}
    Source -->|Upload| Upload[Upload image file]
    Source -->|Sample| Sample[Select sample image]

    Upload --> Validate{Valid image?}
    Sample --> Load[Load sample from disk]

    Validate -->|No| Error[Show error message]
    Error --> Source

    Validate -->|Yes| Ready[Image loaded and ready]
    Load --> Ready

    Ready --> ModuleChoice{Choose module}
    ModuleChoice -->|Enhancement| Enh[Pick enhancement operation]
    ModuleChoice -->|Feature Detection| Feat[Pick detection operation]
    ModuleChoice -->|Segmentation/Clustering| Seg[Pick segmentation/clustering operation]

    Enh --> Params[Adjust parameters via widgets]
    Feat --> Params
    Seg --> Params

    Params --> Process[Run selected CV function]
    Process --> ParamValid{Parameters valid?}
    ParamValid -->|No| ShowError[Show validation error]
    ShowError --> Params

    ParamValid -->|Yes| Display[Show original vs processed side-by-side]
    Display --> Histogram{Histogram relevant?}
    Histogram -->|Yes| ShowHist[Show histogram comparison]
    Histogram -->|No| Download
    ShowHist --> Download[Offer PNG download]
    Download --> ModuleChoice
```

## 3. Use Case Diagram

```mermaid
graph LR
    User((User))

    UC1([Upload image])
    UC2([Select sample image])
    UC3([Apply enhancement operation])
    UC4([Apply feature detection])
    UC5([Apply segmentation/clustering])
    UC6([View before/after comparison])
    UC7([View histogram])
    UC8([Download processed image])

    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC4
    User --> UC5
    User --> UC6
    User --> UC7
    User --> UC8

    UC3 -.includes.-> UC6
    UC4 -.includes.-> UC6
    UC5 -.includes.-> UC6
    UC3 -.includes.-> UC7
```

## 4. Class / Component Diagram

Represents the module-level components and their key functions (VisionLab is written as a functional module library rather than an OOP class hierarchy, so this diagram shows components/functions grouped by module).

```mermaid
classDiagram
    class app {
        +main()
        +load_input_image()
        +render_result()
        +enhancement_tab()
        +feature_detection_tab()
        +segmentation_clustering_tab()
    }

    class preprocessing {
        +prepare_uploaded_image()
        +load_sample_image()
        +as_working_copy()
        +ensure_grayscale_if_requested()
        +get_supported_sample_images()
    }

    class enhancement {
        +flip_image()
        +adjust_contrast()
        +remove_noise()
        +linear_transform()
        +log_transform()
        +gamma_transform()
        +histogram_equalization()
        +morphology_op()
    }

    class feature_detection {
        +canny_edges()
        +detect_corners()
        +hough_lines()
    }

    class segmentation {
        +watershed_segment()
    }

    class clustering {
        +kmeans_segment()
        +kmeans_cluster_map()
    }

    class validation {
        +VisionLabError
        +validate_file_extension()
        +validate_file_size()
        +validate_image_array()
        +validate_parameter()
        +validate_odd_kernel_size()
        +validate_choice()
        +validate_k_clusters()
    }

    class image_utils {
        +load_image_from_bytes()
        +resize_if_needed()
        +to_rgb()
        +to_gray()
        +to_bgr()
        +encode_for_download()
        +get_image_dimensions()
    }

    class visualization {
        +show_side_by_side()
        +plot_histogram()
        +plot_histogram_comparison()
        +show_single_image()
        +close_figure()
    }

    app --> preprocessing
    app --> enhancement
    app --> feature_detection
    app --> segmentation
    app --> clustering
    app --> visualization

    preprocessing --> validation
    preprocessing --> image_utils
    enhancement --> validation
    feature_detection --> validation
    feature_detection --> image_utils
    segmentation --> image_utils
    clustering --> image_utils
```

## 5. Sequence Diagram — Applying an Enhancement Operation

Illustrates the interaction sequence when a user applies, for example, gamma transformation to their uploaded image.

```mermaid
sequenceDiagram
    actor User
    participant UI as app.py
    participant Pre as preprocessing.py
    participant Val as validation.py
    participant Enh as enhancement.py
    participant Viz as visualization.py

    User->>UI: Upload image file
    UI->>Pre: prepare_uploaded_image(name, bytes, size)
    Pre->>Val: validate_file_extension(name)
    Val-->>Pre: OK
    Pre->>Val: validate_file_size(size)
    Val-->>Pre: OK
    Pre->>Pre: load_image_from_bytes(bytes)
    Pre->>Val: validate_image_array(img)
    Val-->>Pre: OK
    Pre-->>UI: validated image

    User->>UI: Select "Gamma Transformation", set gamma=2.0
    UI->>Enh: gamma_transform(img, gamma=2.0)
    Enh->>Val: validate_parameter(gamma, 0.1, 5.0, "gamma")
    Val-->>Enh: 2.0
    Enh->>Enh: apply LUT-based gamma correction
    Enh-->>UI: processed image

    UI->>Viz: show_side_by_side(original, processed)
    Viz-->>UI: matplotlib Figure
    UI-->>User: Render comparison + download button

    User->>UI: Click "Download result"
    UI-->>User: PNG file
```
