from pathlib import Path

VALIDATION_CSV = Path("validation.csv")
VALIDATION_IMAGES_DIR = Path("images_cropped")

CLASS_NAMES = [
    "Adenoma",
    "Sessile_serrated_adenoma",
    "Hyperplastic",
    "Adenocarcinoma",
]
LABEL_MAP = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES)}

GROUP_COLUMNS = ["patient_id", "day", "R", "F"]
BASE_METADATA_COLUMNS = [*GROUP_COLUMNS, "hour", "histology", "filename"]
