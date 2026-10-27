from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from cmapss_xai.constants import RAW_COLUMNS, SUBSETS
from cmapss_xai.paths import ensure_dir


@dataclass(frozen=True)
class RawSubset:
    subset: str
    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.DataFrame
    raw_dir: Path


def extract_archive(zip_path: Path, raw_dir: Path) -> Path:
    extract_dir = ensure_dir(raw_dir / "CMAPSSData")
    with ZipFile(zip_path) as archive:
        needed = {name for name in archive.namelist() if name.endswith(".txt")}
        missing_after = not all((extract_dir / Path(name).name).exists() for name in needed)
        if missing_after:
            for member in archive.namelist():
                if member.endswith(".txt"):
                    (extract_dir / Path(member).name).write_bytes(archive.read(member))
    return extract_dir


def read_cmapss_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLUMNS, engine="python")
    return frame.astype({"unit_id": "int64", "cycle": "int64"})


def read_rul_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=["final_rul"], engine="python")
    frame.insert(0, "unit_id", range(1, len(frame) + 1))
    return frame.astype({"unit_id": "int64", "final_rul": "int64"})


def subset_paths(raw_dir: Path, subset: str) -> tuple[Path, Path, Path]:
    normalized = subset.upper()
    if normalized not in SUBSETS:
        raise ValueError(f"Unsupported subset {subset}. Expected one of {', '.join(SUBSETS)}")
    base = raw_dir / "CMAPSSData"
    return (
        base / f"train_{normalized}.txt",
        base / f"test_{normalized}.txt",
        base / f"RUL_{normalized}.txt",
    )


def load_raw_subset(raw_dir: Path, subset: str) -> RawSubset:
    train_path, test_path, rul_path = subset_paths(raw_dir, subset)
    normalized = subset.upper()
    return RawSubset(
        subset=normalized,
        train=read_cmapss_table(train_path),
        test=read_cmapss_table(test_path),
        test_rul=read_rul_table(rul_path),
        raw_dir=raw_dir,
    )
