# Data-Centric AI Polyp Classification Pipeline

This repository contains a Data-Centric AI pipeline for multiclass polyp classification in colonoscopy images. The project organizes dataset preparation, baseline training, video-based data ingestion, and data curation experiments across Python modules and Jupyter notebooks.

## Phases

- **Phase 0:** dataset normalization, crop preprocessing, and grouped train/validation split.
- **Phase 1:** baseline model training on the normalized image dataset.
- **Phase 2:** video-based data ingestion using external YOLO weights and candidate frame extraction.
- **Phase 3:** data curation, quality filtering, and temporal deduplication.

## Repository Structure

- `src/`: core experiment code, training logic, phase configuration, curation, quality filtering, and reporting utilities.
- `utils/`: shared utilities, plotting, metrics, dataset helpers, and threshold-selection notebooks/modules.
- `dropbox_utils/`: Dropbox inventory, download, and temporary video helpers.
- `scripts/`: auxiliary figure-generation scripts.
- Root notebooks: main experiment notebooks for phases 0-3, ViT variants, sweep experiments, and final results review.
- `requirements.txt`: external Python dependencies detected from the source code and notebooks.
- `LICENSE`: project license.

## Environment

Create and activate a Python environment, then install the project dependencies with:

```bash
pip install -r requirements.txt
```

The repository does not include clinical data, videos, generated results, model weights, or serialized artifacts. These files are excluded for privacy, size, and reproducibility reasons.

External YOLO weights must be provided locally at the path configured by the project:

```text
utils/model/CVC_ClinicDB_yolov8m.pt
```

The code resolves experiment data under `data/` and generated outputs under `results/`. Those directories must be recreated locally when running the experiments.

## Recommended Notebook Order

Run the main notebooks in this order:

1. `phase0.ipynb`
2. `phase1.ipynb`
3. `phase1ViT.ipynb`
4. `phase2.ipynb`
5. `phase2ViT.ipynb`
6. `phase3.ipynb`
7. `phase3ViT.ipynb`
8. `phase3_sweep.ipynb`
9. `final_results.ipynb`

The notebooks in `utils/thresholds/` document threshold exploration and selection for quality filtering and deduplication.
