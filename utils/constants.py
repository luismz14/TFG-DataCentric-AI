from pathlib import Path

import src.ModelTrain as ModelTrain


VALIDATION_CSV = Path("validation.csv")


# This configuration is used on the baseline model, and fixed for all the experiments in the project. 
# Doing this we ensure that the evolution of the results is only due to the changes in the data.
BASELINE_CONFIG = ModelTrain.TrainingConfig(    
    train_ratio=0.8,
    random_state=42,

    input_size=224,
    val_resize_size=256,
    batch_size=32,
    num_workers=0,
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
    class_weight_exponent=0.5
)
