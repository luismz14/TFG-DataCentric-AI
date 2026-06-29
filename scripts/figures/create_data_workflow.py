from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "reports" / "figures"

RAW_IMAGES_DIR = DATA_DIR / "unified_images"
FRAMES_DIR = DATA_DIR / "phase2" / "frames"
PHASE3_SOURCE_CSV = DATA_DIR / "phase3" / "phase3_train_conf040.csv"
PHASE3_REFINED_CSV = DATA_DIR / "phase3" / "phase3_train_conf040_dedup_p75_25.csv"
YOLO_WEIGHTS = ROOT / "utils" / "model" / "CVC_ClinicDB_yolov8m.pt"

CASE = {
    "patient_id": "5764199",
    "day": 20250227,
    "R": "R1",
    "F": "F0",
    "video_filename": "20250227_120856_R1_0bc7a3aa84f8e046.mp4",
    "histology": "Hyperplastic",
}

BASE_FRAME = "20250227_120857_R1_F0_S1_0bc7a3aa84f8e046.jpg"
OUTPUT_STEM = "data_workflow_alt"


def read_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def crop_from_row(image: np.ndarray, row: pd.Series) -> np.ndarray:
    x = int(row["crop_x"])
    y = int(row["crop_y"])
    w = int(row["crop_w"])
    h = int(row["crop_h"])
    return image[y : y + h, x : x + w]


def softened(path: Path, alpha: float = 0.45) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = ImageEnhance.Color(image).enhance(0.45)
    white = Image.new("RGB", image.size, "white")
    return np.asarray(Image.blend(image, white, alpha))


def make_filmstrip(filenames: list[str], thumb_size: tuple[int, int] = (92, 68)) -> np.ndarray:
    thumbs = []
    for filename in filenames:
        image = Image.open(FRAMES_DIR / filename).convert("RGB")
        image.thumbnail(thumb_size, Image.LANCZOS)
        frame = Image.new("RGB", thumb_size, "black")
        frame.paste(
            image,
            ((thumb_size[0] - image.width) // 2, (thumb_size[1] - image.height) // 2),
        )
        thumbs.append(frame)

    gap = 4
    border = 8
    width = border * 2 + len(thumbs) * thumb_size[0] + (len(thumbs) - 1) * gap
    height = thumb_size[1] + border * 2
    strip = Image.new("RGB", (width, height), "#111827")
    for i, thumb in enumerate(thumbs):
        strip.paste(thumb, (border + i * (thumb_size[0] + gap), border))
    return np.asarray(strip)


def add_image(ax, image: np.ndarray, border: str = "#3b3b3b", linewidth: float = 0.8):
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
        spine.set_edgecolor(border)


def add_relative_box(
    ax,
    xy: tuple[float, float] = (0.25, 0.16),
    wh: tuple[float, float] = (0.42, 0.48),
    color: str = "#1a8f5a",
):
    ax.add_patch(
        patches.Rectangle(
            xy,
            wh[0],
            wh[1],
            transform=ax.transAxes,
            fill=False,
            linewidth=1.8,
            edgecolor=color,
        )
    )


def add_pixel_box(
    ax,
    bbox: tuple[float, float, float, float] | None,
    color: str = "#1a8f5a",
):
    if bbox is None:
        return

    x1, y1, x2, y2 = bbox
    ax.add_patch(
        patches.Rectangle(
            (x1, y1),
            max(0.0, x2 - x1),
            max(0.0, y2 - y1),
            fill=False,
            linewidth=1.6,
            edgecolor=color,
        )
    )


def add_cross(ax, color: str = "#b53434"):
    ax.plot([0.08, 0.92], [0.10, 0.90], transform=ax.transAxes, color=color, lw=2.4)
    ax.plot([0.92, 0.08], [0.10, 0.90], transform=ax.transAxes, color=color, lw=2.4)


def add_text(fig, x: float, y: float, text: str, size: int = 9, weight: str = "normal"):
    fig.text(
        x,
        y,
        text,
        ha="center",
        va="top",
        fontsize=size,
        fontweight=weight,
        color="#1f2933",
        linespacing=1.05,
    )


def add_stage_text(fig, x: float, title: str, subtitle: str | None = None):
    title = "\n".join(textwrap.wrap(title, width=20))
    add_text(fig, x, 0.215, title, size=6.6, weight="semibold")
    if subtitle:
        add_text(fig, x, 0.122, "\n".join(textwrap.wrap(subtitle, width=22)), size=5.8)


def add_arrow(fig, x0: float, x1: float, y: float = 0.55):
    arrow = patches.FancyArrowPatch(
        (x0, y),
        (x1, y),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.1,
        color="#6b7280",
    )
    fig.add_artist(arrow)


def add_plus(fig, x: float, y: float):
    fig.text(x, y, "+", ha="center", va="center", fontsize=14, color="#4b5563")


def add_thumbnail_axes(fig, left: float, bottom: float, width: float, height: float):
    return fig.add_axes([left, bottom, width, height])


def case_dataframe(source_df: pd.DataFrame, refined_df: pd.DataFrame) -> pd.DataFrame:
    selected = set(refined_df["filename"].astype(str))
    df = source_df.copy()
    df["selected"] = df["filename"].astype(str).isin(selected)
    mask = (
        (df["patient_id"].astype(str) == CASE["patient_id"])
        & (df["day"].astype(int) == CASE["day"])
        & (df["R"] == CASE["R"])
        & (df["F"] == CASE["F"])
        & (df["video_filename"] == CASE["video_filename"])
        & (df["histology"] == CASE["histology"])
    )
    return df.loc[mask].sort_values(["elapsed_seconds", "filename"]).reset_index(drop=True)


def class_examples(refined_df: pd.DataFrame) -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    for histology in [
        "Adenoma",
        "Sessile_serrated_adenoma",
        "Hyperplastic",
        "Adenocarcinoma",
    ]:
        subset = refined_df[refined_df["histology"] == histology].copy()
        subset["bbox_area_ratio"] = pd.to_numeric(
            subset["bbox_area_ratio"], errors="coerce"
        ).fillna(0)
        subset["detection_confidence"] = pd.to_numeric(
            subset["detection_confidence"], errors="coerce"
        ).fillna(0)
        subset = subset.sort_values(
            ["detection_confidence", "bbox_area_ratio"],
            ascending=False,
        )
        for filename in subset["filename"].astype(str):
            if (FRAMES_DIR / filename).exists():
                examples.append((histology, filename))
                break
    return examples


def detect_best_boxes(filenames: list[str]) -> dict[str, tuple[float, float, float, float] | None]:
    from ultralytics import YOLO

    model = YOLO(str(YOLO_WEIGHTS))
    boxes_by_filename: dict[str, tuple[float, float, float, float] | None] = {}

    for filename in dict.fromkeys(filenames):
        image_path = FRAMES_DIR / filename
        if not image_path.exists():
            boxes_by_filename[filename] = None
            continue

        results = model.predict(
            str(image_path),
            conf=0.01,
            imgsz=640,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            boxes_by_filename[filename] = None
            continue

        boxes = results[0].boxes
        best_index = int(boxes.conf.argmax().item())
        xyxy = boxes.xyxy[best_index].cpu().numpy().tolist()
        boxes_by_filename[filename] = tuple(float(value) for value in xyxy)

    return boxes_by_filename


def draw_crop_rectangle(ax, row: pd.Series, image: np.ndarray):
    height, width = image.shape[:2]
    x = row["crop_x"] / width
    y = 1 - (row["crop_y"] + row["crop_h"]) / height
    w = row["crop_w"] / width
    h = row["crop_h"] / height
    ax.add_patch(
        patches.Rectangle(
            (x, y),
            w,
            h,
            transform=ax.transAxes,
            fill=False,
            linewidth=1.8,
            edgecolor="#1a8f5a",
        )
    )


def build_figure() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_csv(PHASE3_SOURCE_CSV)
    refined_df = pd.read_csv(PHASE3_REFINED_CSV)
    refined_by_filename = refined_df.set_index("filename", drop=False)
    case_df = case_dataframe(source_df, refined_df)
    base_row = case_df[case_df["filename"] == BASE_FRAME].iloc[0]

    raw_image = read_rgb(RAW_IMAGES_DIR / BASE_FRAME)
    cropped_image = crop_from_row(raw_image, base_row)

    fig = plt.figure(figsize=(7.2, 3.05), facecolor="white")

    stage_x = [0.075, 0.255, 0.445, 0.635, 0.845]
    stage_w = 0.135
    top = 0.48

    # 1. Datos clinicos originales
    ax = add_thumbnail_axes(fig, stage_x[0] - 0.055, top + 0.035, 0.11, 0.25)
    add_image(ax, raw_image)
    add_stage_text(fig, stage_x[0], "Imagen cl\u00ednica original")

    # 2. Normalizacion
    ax = add_thumbnail_axes(fig, stage_x[1] - 0.055, top + 0.035, 0.11, 0.25)
    add_image(ax, cropped_image)
    add_stage_text(fig, stage_x[1], "Normalizaci\u00f3n", "Recorte de regiones no informativas")

    # 3. Ampliacion mediante video
    video_files = [
        "20250227_120901_R1_F0_S4_0bc7a3aa84f8e046_2.jpg",
        "20250227_120901_R1_F0_S4_0bc7a3aa84f8e046_3.jpg",
        "20250227_120901_R1_F0_S4_0bc7a3aa84f8e046_4.jpg",
        "20250227_120901_R1_F0_S4_0bc7a3aa84f8e046_5.jpg",
        "20250227_120904_R1_F0_S5_0bc7a3aa84f8e046_1.jpg",
    ]
    refine_files = [
        ("20250227_120904_R1_F0_S5_0bc7a3aa84f8e046_1.jpg", False),
        ("20250227_120904_R1_F0_S5_0bc7a3aa84f8e046_2.jpg", False),
        ("20250227_120904_R1_F0_S5_0bc7a3aa84f8e046_3.jpg", False),
        ("20250227_120904_R1_F0_S5_0bc7a3aa84f8e046_5.jpg", True),
        ("20250227_120905_R1_F0_S6_0bc7a3aa84f8e046_5.jpg", True),
    ]
    result_examples = class_examples(refined_df)
    plotted_filenames = [
        *video_files,
        *(filename for filename, _ in refine_files),
        *(filename for _, filename in result_examples),
    ]
    detected_boxes = detect_best_boxes(plotted_filenames)

    for i, filename in enumerate(video_files):
        ax = add_thumbnail_axes(fig, stage_x[2] - 0.070 + i * 0.035, top + 0.085, 0.033, 0.155)
        add_image(ax, read_rgb(FRAMES_DIR / filename), border="#4b5563", linewidth=0.7)
        add_pixel_box(ax, detected_boxes.get(filename))
    add_stage_text(fig, stage_x[2], "Ampliaci\u00f3n mediante v\u00eddeo", "Selecci\u00f3n de fotogramas pr\u00f3ximos a la lesi\u00f3n")

    # 4. Refinamiento
    for i, (filename, selected) in enumerate(refine_files):
        ax = add_thumbnail_axes(fig, stage_x[3] - 0.070 + i * 0.035, top + 0.085, 0.033, 0.155)
        image = read_rgb(FRAMES_DIR / filename) if selected else softened(FRAMES_DIR / filename)
        add_image(ax, image, border="#1f7a4d" if selected else "#b53434", linewidth=1.1)
        add_pixel_box(ax, detected_boxes.get(filename))
        if not selected:
            add_cross(ax)
    add_stage_text(fig, stage_x[3], "Refinamiento", "Reducci\u00f3n de redundancia temporal")

    # 5. Conjunto resultante
    labels = {
        "Adenoma": "Adenoma",
        "Sessile_serrated_adenoma": "SSA",
        "Hyperplastic": "Hiperpl\u00e1sico",
        "Adenocarcinoma": "Adenocarcinoma",
    }
    for i, (histology, filename) in enumerate(result_examples):
        row = i // 2
        col = i % 2
        ax = add_thumbnail_axes(
            fig,
            stage_x[4] - 0.078 + col * 0.103,
            top + 0.180 - row * 0.152,
            0.065,
            0.108,
        )
        add_image(ax, read_rgb(FRAMES_DIR / filename), border="#4b5563", linewidth=0.7)
        add_pixel_box(ax, detected_boxes.get(filename))
        ax.text(
            0.5,
            -0.14,
            labels[histology],
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=4.6,
            color="#374151",
            clip_on=False,
        )
    add_stage_text(fig, stage_x[4], "Conjunto de entrenamiento resultante")

    # Flow arrows
    for left, right in zip(stage_x[:4], stage_x[1:5]):
        add_arrow(fig, left + stage_w / 2, right - stage_w / 2)

    fig.text(
        0.5,
        0.935,
        "Flujo de procesamiento propuesto sobre los datos cl\u00ednicos",
        ha="center",
        va="center",
        fontsize=8.7,
        fontweight="semibold",
        color="#111827",
    )

    output_pdf = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
    output_png = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_pdf


if __name__ == "__main__":
    print(build_figure())
