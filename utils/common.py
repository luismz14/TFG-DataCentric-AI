from pathlib import Path

import pandas as pd


def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)

    if path.suffix.lower() != ".csv":
        raise ValueError(f"`path` must point to a CSV file. Got: {path}")

    return pd.read_csv(path, **kwargs)


def write_csv(
    dataframe: pd.DataFrame,
    path: str | Path,
    index: bool = False,
    encoding: str = "utf-8",
    **kwargs,
) -> Path:
    path = ensure_parent_dir(path)
    dataframe.to_csv(path, index=index, encoding=encoding, **kwargs)
    return path


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    context: str = "dataframe",
) -> None:
    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if not missing_columns:
        return

    raise ValueError(
        f"Missing required columns in {context}: {', '.join(missing_columns)}"
    )
