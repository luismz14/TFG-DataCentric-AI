from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "reports" / "figures"

MAIN_CSV = DATA_DIR / "phase1_train.csv"
MAIN_IMAGES_DIR = DATA_DIR / "unified_images"
EXTERNAL_CSV = DATA_DIR / "test" / "external_test.csv"
EXTERNAL_IMAGES_DIR = DATA_DIR / "test" / "external_test"
YOLO_WEIGHTS = ROOT / "utils" / "model" / "CVC_ClinicDB_yolov8m.pt"

OUTPUT_STEM = "dataset_examples"

CLASS_ORDER = [
    "Adenoma",
    "Sessile_serrated_adenoma",
    "Hyperplastic",
    "Adenocarcinoma",
]

CLASS_LABELS = {
    "Adenoma": "Adenoma",
    "Sessile_serrated_adenoma": "Adenoma serrado s\u00e9sil",
    "Hyperplastic": "Hiperpl\u00e1sico",
    "Adenocarcinoma": "Adenocarcinoma",
}


def load_image(image_path: Path, row: pd.Series | None = None) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if row is not None and all(column in row for column in ["crop_x", "crop_y", "crop_w", "crop_h"]):
        x = int(float(row["crop_x"]))
        y = int(float(row["crop_y"]))
        w = int(float(row["crop_w"]))
        h = int(float(row["crop_h"]))
        image = image.crop((x, y, x + w, y + h))
    return image


def add_image(ax, image_path: Path, row: pd.Series | None = None) -> None:
    image = load_image(image_path, row)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#374151")


def is_readable_image(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except Exception:
        return False


def crop_area_score(dataframe: pd.DataFrame) -> pd.Series:
    widths = pd.to_numeric(dataframe["crop_w"], errors="coerce").fillna(0)
    heights = pd.to_numeric(dataframe["crop_h"], errors="coerce").fillna(0)
    return widths * heights


def yolo_best_confidence(model, image_path: Path, row: pd.Series | None = None) -> tuple[float, float]:
    image = load_image(image_path, row)
    try:
        results = model.predict(image, conf=0.01, imgsz=640, verbose=False)
    except Exception:
        return 0.0, 0.0
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return 0.0, 0.0

    boxes = results[0].boxes
    best_index = int(boxes.conf.argmax().item())
    confidence = float(boxes.conf[best_index].item())
    x1, y1, x2, y2 = [float(value) for value in boxes.xyxy[best_index].cpu().numpy()]
    image = Image.open(image_path)
    image_area = max(1.0, float(image.width * image.height))
    area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / image_area
    return confidence, area_ratio


def artifact_penalty(image_path: Path, row: pd.Series | None = None) -> float:
    image = load_image(image_path, row).resize((240, 180))
    pixels = np.asarray(image)
    total = max(1, int(pixels.shape[0] * pixels.shape[1]))

    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    blue_overlay = np.count_nonzero((blue > 150) & (red < 80) & (green < 120)) / total
    bright_text = np.count_nonzero((red > 235) & (green > 235) & (blue > 235)) / total
    return 5.0 * blue_overlay + 1.2 * bright_text


def select_examples(
    dataframe: pd.DataFrame,
    images_dir: Path,
    candidates_per_class: int = 90,
) -> dict[str, str]:
    from ultralytics import YOLO

    model = YOLO(str(YOLO_WEIGHTS))
    selections: dict[str, str] = {}

    for histology in CLASS_ORDER:
        subset = dataframe[dataframe["histology"] == histology].copy()
        subset = subset[
            subset["filename"]
            .astype(str)
            .map(lambda name: (images_dir / name).exists() and is_readable_image(images_dir / name))
        ]
        if subset.empty:
            raise ValueError(f"No images found for class {histology} in {images_dir}")

        subset["_crop_area"] = crop_area_score(subset)
        subset = subset.sort_values("_crop_area", ascending=False).head(candidates_per_class)

        scored_rows = []
        for _, row in subset.iterrows():
            filename = str(row["filename"])
            confidence, area_ratio = yolo_best_confidence(model, images_dir / filename, row)
            penalty = artifact_penalty(images_dir / filename, row)
            score = confidence + 0.20 * min(area_ratio, 0.35) - penalty
            scored_rows.append((score, confidence, area_ratio, filename))

        scored_rows.sort(reverse=True)
        selections[histology] = scored_rows[0][3]

    return selections


def build_figure() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    main_df = pd.read_csv(MAIN_CSV)
    external_df = pd.read_csv(EXTERNAL_CSV)
    main_rows = main_df.set_index("filename", drop=False)
    external_rows = external_df.set_index("filename", drop=False)

    main_examples = select_examples(main_df, MAIN_IMAGES_DIR)
    external_examples = select_examples(external_df, EXTERNAL_IMAGES_DIR)

    fig = plt.figure(figsize=(6.7, 8.2), facecolor="white")
    grid = fig.add_gridspec(
        nrows=4,
        ncols=3,
        width_ratios=[0.78, 1.0, 1.0],
        height_ratios=[1, 1, 1, 1],
        left=0.06,
        right=0.97,
        top=0.90,
        bottom=0.07,
        wspace=0.16,
        hspace=0.24,
    )

    fig.text(
        0.43,
        0.94,
        "Conjunto principal\n(Hospital Cl\u00ednic)",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="semibold",
        color="#111827",
        linespacing=1.05,
    )
    fig.text(
        0.76,
        0.94,
        "Test externo\n(PIBAdb + PICCOLO)",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="semibold",
        color="#111827",
        linespacing=1.05,
    )

    for row_idx, histology in enumerate(CLASS_ORDER):
        label_ax = fig.add_subplot(grid[row_idx, 0])
        label_ax.axis("off")
        label_ax.text(
            0.98,
            0.5,
            "\n".join(textwrap.wrap(CLASS_LABELS[histology], width=18)),
            ha="right",
            va="center",
            fontsize=9.5,
            color="#1f2933",
            fontweight="semibold",
            linespacing=1.05,
        )

        main_ax = fig.add_subplot(grid[row_idx, 1])
        main_filename = main_examples[histology]
        add_image(main_ax, MAIN_IMAGES_DIR / main_filename, main_rows.loc[main_filename])

        external_ax = fig.add_subplot(grid[row_idx, 2])
        external_filename = external_examples[histology]
        add_image(
            external_ax,
            EXTERNAL_IMAGES_DIR / external_filename,
            external_rows.loc[external_filename],
        )

    border = patches.Rectangle(
        (0.30, 0.055),
        0.67,
        0.86,
        transform=fig.transFigure,
        fill=False,
        linewidth=0.0,
        edgecolor="none",
    )
    fig.add_artist(border)

    output_pdf = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
    output_png = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Selected examples:")
    for histology in CLASS_ORDER:
        print(
            f"{histology}: main={main_examples[histology]} | "
            f"external={external_examples[histology]}"
        )

    return output_pdf


if __name__ == "__main__":
    print(build_figure())
