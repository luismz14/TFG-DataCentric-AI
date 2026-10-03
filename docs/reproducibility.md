# Reproducibility and result provenance

## Public evidence

[`results-summary.csv`](results-summary.csv) preserves 24 historical aggregate evaluation rows: two architectures, six dataset configurations, and validation/external-test splits. It contains only architecture/configuration identifiers, split names, seed counts, and Macro-F1 mean/standard deviation. No patient identifiers, sample paths, clinical images, credentials, embeddings, or model weights are included.

The values were extracted without recomputation from the local project archive's `results/final_lineage_validation_test_metrics_by_architecture.csv` on 2026-10-03. The source file SHA-256 is `704d37df963fef61390ae773bce35d41fef1fa7be02b041a34a6c3a5485b9298`. The imported values retain the original decimal precision. The source archive is not modified by this extraction.

This is a historical aggregate summary, not an independently rerun evaluation. The summary contains additional configurations beyond the selected README table. In particular, `ViT-Small/train` records validation Macro-F1 0.6298, higher than the README's selected final-deduplication result 0.6233, while its external-test Macro-F1 is 0.4238. Interpret comparisons in the context of the dataset branch and evaluation split; the overall ranking and thesis wording need author review.

## Configuration identifiers

| Key | Meaning in the final-results workflow |
| --- | --- |
| `phase1` | Image-only baseline |
| `train` | Phase 2 training dataset |
| `conf060` | Phase 2 confidence-only ingestion at threshold 0.60 |
| `train_conf040` | Confidence-filtered Phase 2 training branch at threshold 0.40 |
| `train_conf040_dedup_p75_25` | Adaptive deduplication of the confidence-0.40 branch |
| `conf060_dedup_p75_25` | Adaptive deduplication of the confidence-0.60 branch |

The seeds are 42, 123, 456, and 789. Exploratory sweep notebooks can use a different seed count and should not be confused with this final aggregate evaluation.

## Local execution requirements

Use the recommended Python 3.11 environment and existing `requirements.txt`. Open notebooks from the repository root so `src` and `utils` imports resolve. The direct dependency list is unpinned; the original complete environment is not available in the public repository.

`utils.common` resolves dataset paths under local `data/` and generated artifacts under local `results/`. Phase configuration files define the exact filenames and directories; examples include `data/phase1_train.csv`, `data/validation.csv`, and `data/test/external_test.csv`. Metadata contracts are defined by `utils/constants.py` and phase modules. Do not substitute synthetic data for clinical data when interpreting scientific results.

The logical workflow is Phase 0 preparation, Phase 1 baselines, Phase 2 dataset ingestion followed by Phase 2 training, and Phase 3 curation/training. `src.phase2.experiment.ingest_phase2_dataset` exposes ingestion as a Python function. `final_results.ipynb` reads existing experiment artifacts and may recompute evaluations when the required cache is missing; do not run it merely to view the public aggregate CSV.

Full reproduction requires private clinical images, videos, metadata/inventories, trained classifiers, and the exact locally supplied third-party detector at `utils/model/CVC_ClinicDB_yolov8m.pt`. Those resources are excluded from version control. Pretrained classifier downloads and optional Dropbox access can require network access and local credentials. No expensive experiment was rerun during maintenance.

## Material retained outside the public repository

The local archive includes clinical data, per-sample embeddings/projections, model checkpoints, training curves, confusion matrices, and presentation candidate figures. Aggregate curves or schematic diagrams may be useful public evidence after content and rights review. Clinical example images, patient-linked outputs, private inventories, and credentials should remain private. The parent workspace cleanup report records these review candidates.


## Publication decision

See [publication-scope.md](publication-scope.md). The archived workflow diagram embeds clinical imagery and is excluded; it must not be imported as an image-free schematic. No further archive artifacts were imported.
