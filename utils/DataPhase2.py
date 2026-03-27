from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


VIDEO_FILENAME_RE = re.compile(
    r"^(?P<day>\d{8})_(?P<hour>\d{6})_(?P<R>R\d+)_(?P<uid>[^.]+)\.(?P<ext>mp4|avi|mov)$",
    re.IGNORECASE,
)


def parse_video_filename(filename: str) -> dict[str, str] | None:
    match = VIDEO_FILENAME_RE.match(filename)
    if match is None:
        return None

    parts = match.groupdict()
    return {
        "day": parts["day"],
        "hour": parts["hour"],
        "video_filename": filename,
    }


def load_videos(inventory_path: str | Path) -> pd.DataFrame:
    inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    records: list[dict[str, str]] = []

    for patient in inventory:
        patient_id = str(patient["patient_id"])

        for file_data in patient.get("files", []):
            if not file_data.get("is_video", False):
                continue

            parsed_video = parse_video_filename(file_data["name"])
            if parsed_video is None:
                continue

            records.append(
                {
                    "patient_id": patient_id,
                    **parsed_video,
                }
            )

    if not records:
        return pd.DataFrame(
            columns=["patient_id", "day", "hour", "video_filename", "video_timestamp"]
        )

    videos_df = pd.DataFrame(records, dtype=str)
    videos_df["video_timestamp"] = pd.to_datetime(
        videos_df["day"] + videos_df["hour"],
        format="%Y%m%d%H%M%S",
    )

    return videos_df.sort_values(["patient_id", "video_timestamp", "video_filename"]).reset_index(
        drop=True
    )


def match_videos_to_images(baseline_df: pd.DataFrame, videos_df: pd.DataFrame) -> pd.DataFrame:
    matched_groups: list[pd.DataFrame] = []

    for patient_id, image_group_df in baseline_df.groupby("patient_id", sort=False):
        image_group_df = image_group_df.sort_values(["image_timestamp", "row_id"]).copy()
        patient_videos_df = videos_df.loc[videos_df["patient_id"] == patient_id].copy()

        if patient_videos_df.empty:
            image_group_df["video_filename"] = ""
            image_group_df["video_timestamp"] = pd.NaT
            matched_groups.append(image_group_df)
            continue

        patient_videos_df = patient_videos_df.sort_values(["video_timestamp", "video_filename"])

        matched_group_df = pd.merge_asof(
            image_group_df,
            patient_videos_df[["video_timestamp", "video_filename"]],
            left_on="image_timestamp",
            right_on="video_timestamp",
            direction="backward",
            allow_exact_matches=True,
        )
        matched_groups.append(matched_group_df)

    if not matched_groups:
        return baseline_df.copy()

    return pd.concat(matched_groups, ignore_index=True).sort_values("row_id").reset_index(drop=True)


def generate_phase2_baseline_with_video(
    baseline_csv_path: str | Path | None = None,
    inventory_path: str | Path | None = None,
    output_csv_path: str | Path | None = None,
) -> dict[str, int | str]:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    phase2_dir = data_dir / "phase2"

    baseline_csv_path = (
        Path(baseline_csv_path) if baseline_csv_path else data_dir / "unified_data_baseline.csv"
    )
    inventory_path = (
        Path(inventory_path) if inventory_path else data_dir / "dataset_inventory.json"
    )
    output_csv_path = (
        Path(output_csv_path)
        if output_csv_path
        else phase2_dir / "unified_data_baseline_phase2.csv"
    )

    baseline_df = pd.read_csv(
        baseline_csv_path,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    baseline_df["row_id"] = range(len(baseline_df))
    baseline_df["image_timestamp"] = pd.to_datetime(
        baseline_df["day"] + baseline_df["hour"],
        format="%Y%m%d%H%M%S",
    )

    videos_df = load_videos(inventory_path)
    result_df = match_videos_to_images(baseline_df=baseline_df, videos_df=videos_df)

    time_deltas = result_df["image_timestamp"] - result_df["video_timestamp"]

    result_df["elapsed_seconds"] = (
        time_deltas.dt.total_seconds()
        .astype(int)
        .astype(str)
    )


    output_columns = [
        *baseline_df.columns.drop(["row_id", "image_timestamp"]),
        "elapsed_seconds",
        "video_filename",
    ]
    output_df = result_df[output_columns]

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv_path, index=False, encoding="utf-8")

    return True


if __name__ == "__main__":
    summary = generate_phase2_baseline_with_video()
    print(f"Rows processed: {summary['rows_total']}")
    print(f"Output CSV:     {summary['output_csv_path']}")
