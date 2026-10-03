# Public evidence and private reproduction inputs

Publication decision: 2026-10-03. The public tree contains authored source and reviewed aggregate evidence, including `docs/results-summary.csv`. This is historical evidence, not an independently rerun evaluation. The existing MIT source license remains unchanged and does not automatically license clinical data or third-party weights.

Clinical images, crops, videos/frames, patient/session metadata, pathology/medical text and backups are intentionally not distributed. Per-sample projections, embeddings, clinical presentation bundles and clinical-trained checkpoints remain private unless separately cleared. Credentials and `.env` files are never imported.

The external project archive remains completely unchanged. Its workflow diagram embeds actual colonoscopy imagery and is not a purely schematic public asset. It was not imported. No new archive curves, model artifacts or results were imported in this pass.

Exact reproduction requires authorized private inputs, the historical environment and separately supplied model artifacts, including the local detector. Public source and aggregate CSV alone are insufficient. Do not substitute new/synthetic data and claim reproduction. See [reproducibility.md](reproducibility.md) for paths and provenance.

Scientific/methodological and conclusion-selection findings remain unresolved. No experiments, metrics, results or conclusions were changed. Git history remains untouched; any historical cleanup is a separate manual owner decision.
