import pandas as pd
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PATIENT_CONTENT_DIR = DATA_DIR / "patient_content"
UNIFIED_METADATA_CSV = DATA_DIR / "unified_data_baseline.csv"
UNIFIED_IMAGES_DIR = DATA_DIR / "unified_images"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VALID_HISTOLOGIES = {
    "Adenoma",
    "Sessile_serrated_adenoma",
    "Hyperplastic",
    "Adenocarcinoma",
}


def _read_session_csv(csv_file: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_file,
        sep=None,
        engine="python",
        dtype=str,
        keep_default_na=False,
    )


def _find_column(df: pd.DataFrame, column_name: str) -> str | None:
    target = column_name.strip().lower()
    for column in df.columns:
        if str(column).strip().lower() == target:
            return column
    return None


def _normalize_histology(histology: pd.Series) -> pd.Series:
    normalized = histology.astype(str).str.strip()
    normalized = normalized.replace(
        {
            "Sessile serrated adenoma (with displasia)": "Sessile_serrated_adenoma",
            "Sessile serrated adenoma": "Sessile_serrated_adenoma",
        }
    )
    return normalized


def _build_metadata_from_session_csv(csv_file: Path, patient_id: str) -> pd.DataFrame:
    df = _read_session_csv(csv_file)
    if df.empty:
        return pd.DataFrame()

    filename_column = df.columns[0]
    histology_column = _find_column(df, "assumed histology")
    if histology_column is None:
        raise ValueError("missing 'assumed histology' column")

    filenames = df[filename_column].astype(str).str.strip()
    valid_filename_mask = filenames.map(
        lambda value: Path(value).suffix.lower() in IMAGE_EXTENSIONS
    )
    df = df.loc[valid_filename_mask].copy()
    filenames = filenames.loc[valid_filename_mask]
    if df.empty:
        return pd.DataFrame()

    filename_parts = filenames.str.extract(
        r"^(?P<day>\d{8})_(?P<hour>\d{6})_(?P<R>R\d+)_(?P<F>F\d+)_"
    )
    valid_pattern_mask = filename_parts.notna().all(axis=1)
    if not valid_pattern_mask.all():
        invalid_count = int((~valid_pattern_mask).sum())
        raise ValueError(f"{invalid_count} filenames do not match expected pattern")

    output_df = filename_parts.copy()
    output_df.insert(0, "patient_id", str(patient_id))
    output_df["histology"] = _normalize_histology(df[histology_column])
    output_df["filename"] = filenames
    return output_df


def unifyExcels(
    base_path: str | Path = PATIENT_CONTENT_DIR,
    output_path: str | Path = UNIFIED_METADATA_CSV,
) -> pd.DataFrame | None:
    base_path = Path(base_path)
    output_path = Path(output_path)
    all_dfs = []
    
    for patient_folder in (f for f in base_path.iterdir() if f.is_dir()):
        for csv_file in patient_folder.glob('*.csv'):
            try:
                temp_df = _build_metadata_from_session_csv(csv_file, patient_folder.name)
                if not temp_df.empty:
                    all_dfs.append(temp_df)
                
            except Exception as e:
                print(f'Error processing {csv_file.name}: {e}')

    if not all_dfs:
        print("No data found to unify.")
        return

    fulldf = pd.concat(all_dfs, ignore_index=True)

    fulldf = fulldf[fulldf["histology"].isin(VALID_HISTOLOGIES)]
    fulldf = fulldf.sort_values(
        ["patient_id", "day", "hour", "R", "F", "filename"]
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fulldf.to_csv(output_path, index=False)
    print(f'CSV unificado en: {output_path}')
    return fulldf

def unifyImages(
    src_base: str | Path = PATIENT_CONTENT_DIR,
    dst_dir: str | Path = UNIFIED_IMAGES_DIR,
) -> None:
    src_base = Path(src_base)
    dst_dir = Path(dst_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for patient_folder in (f for f in src_base.iterdir() if f.is_dir()):
        for img_path in patient_folder.iterdir():
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            dst_path = dst_dir / img_path.name
            try:
                shutil.copy2(img_path, dst_path)
                count += 1
            except Exception as e:
                print(f'Error copying {img_path.name}: {e}')
    
    print(f'Process completed. {count} files copied to {dst_dir}')

if __name__ == '__main__':
    unifyExcels()
    unifyImages()
