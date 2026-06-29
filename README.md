# Improving Medical Diagnostic Models in Colonoscopy Through Data-Centric AI

## Overview

This repository contains the source code and experimentation notebooks for a Final Degree Project in Computer Science focused on Data-Centric AI for colonoscopy-based medical diagnosis. The project develops and evaluates a pipeline for improving multiclass polyp classification by acting on the training data, rather than relying on continuous changes to the model architecture.

The experimental workflow uses EfficientNet-B0 and ViT-Small classifiers as baseline and comparison architectures. The repository is intended to document the complete project logic, experiment structure, and execution order while keeping clinical data and generated artifacts outside version control.

## Pipeline

- **Phase 0 - Dataset normalization and grouped split:** normalizes the clinical repository data, prepares the image metadata, crops the relevant regions, and creates a grouped train-validation split to reduce leakage between related samples.
- **Phase 1 - Baseline model training:** trains the initial EfficientNet-B0 and ViT-Small classifiers on the normalized image dataset.
- **Phase 2 - Video-based data ingestion:** expands the training data using colonoscopy videos, the included YOLO detector, and ByteTrack-based tracking.
- **Phase 3 - Data curation and deduplication:** evaluates quality filtering strategies and temporal deduplication using SSIM and pHash to reduce redundant or low-quality samples.

## Repository Structure

```text
.
|-- src/                         # Core pipeline, training, phase logic, reporting, and curation code
|-- phase0.ipynb                 # Dataset normalization and grouped split workflow
|-- phase1.ipynb                 # EfficientNet-B0 Phase 1 baseline experiment
|-- phase1ViT.ipynb              # ViT-Small Phase 1 baseline experiment
|-- phase2.ipynb                 # EfficientNet-B0 Phase 2 video ingestion experiment
|-- phase2ViT.ipynb              # ViT-Small Phase 2 video ingestion experiment
|-- phase3.ipynb                 # EfficientNet-B0 Phase 3 curation experiment
|-- phase3ViT.ipynb              # ViT-Small Phase 3 curation experiment
|-- phase3_sweep.ipynb           # Phase 3 curation and filtering sweep
|-- final_results.ipynb          # Analysis of generated experiment results
|-- utils/                       # Shared utilities, metrics, plotting, and dataset helpers
|-- utils/thresholds/            # Threshold exploration notebooks and helper modules
|-- utils/model/
|   `-- CVC_ClinicDB_yolov8m.pt  # YOLO detector required by Phase 2
|-- dropbox_utils/               # Optional helpers for remote dataset inventory/download workflows
|-- scripts/                     # Auxiliary figure-generation scripts
|-- requirements.txt             # Python dependencies
`-- LICENSE                     # Project license
```

## Included Resources

The repository includes:

- Python source code for the full pipeline;
- experimentation notebooks for the main project phases;
- configuration files and utility modules;
- threshold-selection notebooks and helper modules;
- the YOLO detector weight required by Phase 2: `utils/model/CVC_ClinicDB_yolov8m.pt`;
- dependency and documentation files.

The included YOLO weight was obtained from the [YOLO_SAM2 repository](https://github.com/sajjad-sh33/YOLO_SAM2/).

## Excluded Resources

The repository does not include:

- clinical images;
- colonoscopy videos;
- clinical CSVs or metadata;
- generated results;
- training figures;
- EfficientNet-B0 or ViT-Small checkpoints.

These exclusions are intentional because of clinical privacy, repository size, and the separation between source code and generated experiment artifacts.

## Installation

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or activate it on Unix systems:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Local Data and Execution Requirements

The full pipeline requires local resources that are intentionally not versioned:

- a `data/` directory containing the images, videos, CSVs, and metadata prepared according to the structure expected by the project;
- a `results/` directory used as the destination for generated metrics, plots, checkpoints, and intermediate outputs;
- a `.env` file only if the optional Dropbox utilities are used.

The repository documents the complete experimental workflow, but it is not self-contained for full execution because the clinical data are not included.

## Recommended Execution Order

1. `phase0.ipynb`
2. `phase1.ipynb` or `phase1ViT.ipynb`
3. `phase2.ipynb` or `phase2ViT.ipynb`
4. `phase3.ipynb`, `phase3ViT.ipynb`, or `phase3_sweep.ipynb`

`final_results.ipynb` is intended for analyzing already generated results and requires `results/` to exist locally.

## Expected Outputs

When the required local data are available, the notebooks and Python modules generate experiment artifacts under `results/`, including:

- trained classifier checkpoints;
- training curves and confusion matrices;
- per-run and aggregated metrics;
- CSV summaries for phase comparisons and final analysis.

These outputs are generated artifacts and are intentionally excluded from Git.

## Academic Context

- **Author:** Luis Martínez
- **Supervisor:** Yael Tudela
- **Institution:** UAB School of Engineering
- **Academic year:** 2025-26
- **Project type:** Final Degree Project in Computer Science

## License

This project is distributed under the terms of the license included in `LICENSE`.
