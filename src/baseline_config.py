"""Fixed training recipe shared by all data-centric phases."""

from dataclasses import fields

from src.architecture import EFFICIENTNET_B0
from src.training import TrainingConfig


# This configuration is intentionally fixed across phases so metric changes can
# be attributed to data changes instead of training recipe changes. The values
# match the phase-1 winner from the advisor reruns: properlr_focal_strongaug.
BASELINE_CONFIG = TrainingConfig(
    architecture=EFFICIENTNET_B0,
    random_state=42,
    input_size=224,
    val_resize_size=256,
    batch_size=32,
    num_workers=8,
    num_epochs=100,
    warmup_epochs=3,
    head_lr=1e-3,
    fine_tune_head_lr=1e-4,
    backbone_lr=1e-5,
    weight_decay=5e-4,
    dropout=0.3,
    stochastic_depth_prob=0.1,
    label_smoothing=0.05,
    scheduler_factor=0.5,
    scheduler_patience=4,
    early_stopping_patience=16,
    min_lr=1e-7,
    gradient_clip_norm=1.0,
    loss="focal",
    use_weighted_loss=False,
    class_weight_exponent=0.7,
    sampler_exponent=0.4,
    focal_gamma=2.0,
    augment="strong",
    checkpoint_selection="macro_f1",
)


def build_training_config(architecture: str = EFFICIENTNET_B0) -> TrainingConfig:
    """Return the fixed baseline recipe for the selected architecture."""
    config = TrainingConfig(
        **{
            field.name: getattr(BASELINE_CONFIG, field.name)
            for field in fields(BASELINE_CONFIG)
        }
    )
    config.architecture = architecture
    return config
