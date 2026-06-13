"""Compatibility facade for Phase 3 data-curation workflows.

New code should import from `src.phase3.curation`.
"""

from src.phase3.curation import (
    DedupMode,
    Operation,
    PHASE3_DEDUP_MODES,
    PHASE3_QUALITY_MODES,
    PHASE3_CURATION_DATA_DIR,
    PHASE3_TOP_FRACTIONS,
    Phase3ExperimentSpec,
    Phase3SourceSpec,
    QualityMode,
    build_phase3_all_experiment_specs,
    build_phase3_combined_experiment_specs,
    build_phase3_individual_experiment_specs,
    curate_phase3_train_conf040_dataset,
    curate_phase3_top_dataset,
    phase1_scorer_checkpoint_paths,
    phase3_final_experiment_specs,
    phase3_source_specs,
    prepare_phase3_experiment_dataset,
    resolve_phase3_dedup_thresholds,
    resolve_phase3_quality_params,
    score_phase3_source_with_phase1_ensemble,
)

__all__ = [
    "DedupMode",
    "Operation",
    "PHASE3_DEDUP_MODES",
    "PHASE3_QUALITY_MODES",
    "PHASE3_CURATION_DATA_DIR",
    "PHASE3_TOP_FRACTIONS",
    "Phase3ExperimentSpec",
    "Phase3SourceSpec",
    "QualityMode",
    "build_phase3_all_experiment_specs",
    "build_phase3_combined_experiment_specs",
    "build_phase3_individual_experiment_specs",
    "curate_phase3_train_conf040_dataset",
    "curate_phase3_top_dataset",
    "phase1_scorer_checkpoint_paths",
    "phase3_final_experiment_specs",
    "phase3_source_specs",
    "prepare_phase3_experiment_dataset",
    "resolve_phase3_dedup_thresholds",
    "resolve_phase3_quality_params",
    "score_phase3_source_with_phase1_ensemble",
]
