# Vascular Calcification Analysis using DBSCAN (DBV2)

This project is a compact medical imaging software application that utilizes the **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** algorithm to automatically identify, cluster, and analyze vascular calcifications from standard DICOM images (CT scans) or regular morphological images.

The project features an intuitive graphical user interface (GUI) built with `PyQt6` and is optimized with smart Multithreading, preventing the application from freezing during heavy processing.

## 🌟 Key Features

- **DICOM Reading & Processing:** Automatically standardizes and maps pixels to their original physiological Hounsfield Unit (HU) values using the `Rescale/Slope` metadata for accurate medical data.
- **Strict Morphological Preprocessing:** Automatically eliminates background noise and performs Skull-Stripping to remove excess structures using `OpenCV` morphological matrix operations.
- **Non-blocking GUI:** The `QThread` background worker architecture keeps the UI incredibly smooth even when DBSCAN calculates huge datasets.
- **Customizable Parameters:** Doctors and researchers can freely adjust:
  - Scanning neighborhood radius `Epsilon (eps)`
  - Minimum neighborhood samples `Min Points`
  - Calcification diagnostic threshold via `Hounsfield Unit (HU)`

## 📁 Project Structure

The source code follows a standard modular architecture:

```text
├── main.py              # Contains the Graphical User Interface (PyQt6) and application entry point
├── utils.py             # Image preprocessing logic (OpenCV) and physical calculation (pydicom)
├── dbscan_worker.py     # Module containing scikit-learn DBSCAN algorithm running via QThread
├── dbv_2_old.py         # The original legacy source code (Backup)
└── README.md            # This documentation file
```

## System Requirements & Installation

1. Requires **Python 3.8+**.
2. You need to install specialized core libraries for Data Science and GUI:

```bash
pip install numpy opencv-python pydicom scikit-learn PyQt6 Pillow
```

## Usage Guide

1. You can launch the application immediately via Terminal/Command Prompt in the project folder with:
   ```bash
   python main.py
   ```
2. GUI instructions:
   - Click **Chọn ảnh** (Select Image) and point to a valid medical image (`.dcm`, `.jpg`, `.png`).
   - Click **Tiền xử lý** (Preprocess) for the app to discard garbage data, binarize, and extract potential calcification coordinates.
   - Adjust the `eps` slider, `min samples` spinbox, and `HU Threshold`.
   - Click **Phân tích DBSCAN** (Analyze DBSCAN). The app will show a loading state, and when the background process successfully finishes, a detailed quantitative table will be printed out for each calcification cluster including: Area, HU properties, Circularity, Bounding Box, and Confidence.
   - Click **Lưu kết quả** (Save Result) or **Lưu ảnh màu** (Save Color Image) if you need to export a report or use the images for your Research thesis.

## Measurement Reference 

A standard generic Hounsfield Unit reference for verifying tissue systems:
- Air: `-1000 HU`
- Water: `0 HU`
- Fat tissue: `-100 -> 60 HU`
- Soft muscle tissue: `40 -> 60 HU`
- Bone / Calcification (Mild to solid): `130 HU -> 1000+ HU` (The program defaults at a threshold of 130).

---

*Developed by TVD*
