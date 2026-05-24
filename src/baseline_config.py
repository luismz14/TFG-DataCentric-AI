"""Fixed baseline training recipe shared by all data-centric phases."""

from src.ModelTrain import TrainingConfig


# This configuration is intentionally fixed across phases so metric changes can
# be attributed to data changes instead of training recipe changes.
BASELINE_CONFIG = TrainingConfig(
    random_state=42,
    input_size=224,
    val_resize_size=256,
    batch_size=32,
    num_workers=2,
    num_epochs=200,
    warmup_epochs=3,
    head_lr=1e-4,
    fine_tune_head_lr=1e-4,
    backbone_lr=1e-6,
    weight_decay=5e-4,
    dropout=0.3,
    stochastic_depth_prob=0.1,
    label_smoothing=0.02,
    scheduler_factor=0.5,
    scheduler_patience=4,
    early_stopping_patience=12,
    min_lr=1e-7,
    gradient_clip_norm=1.0,
    use_weighted_loss=False,
    class_weight_exponent=0.5,
)
