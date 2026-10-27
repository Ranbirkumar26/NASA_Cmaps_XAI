from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from cmapss_xai.config import DatasetConfig
from cmapss_xai.paths import ensure_dir


def download_cmapss(config: DatasetConfig, force: bool = False) -> Path:
    raw_dir = ensure_dir(config.raw_dir)
    target = raw_dir / "CMAPSSData.zip"
    if target.exists() and not force:
        return target

    request = urllib.request.Request(
        config.source_url,
        headers={"User-Agent": "cmapss-xai/0.1"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header else None
        with target.open("wb") as handle:
            with tqdm(total=total, unit="B", unit_scale=True, desc=target.name) as progress:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    progress.update(len(chunk))
    if not zipfile.is_zipfile(target):
        target.unlink(missing_ok=True)
        raise ValueError(f"Downloaded file is not a valid zip archive: {target}")
    return target
