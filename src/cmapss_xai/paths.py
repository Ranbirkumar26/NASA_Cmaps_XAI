from __future__ import annotations

import os
from pathlib import Path


def _discover_project_root() -> Path:
    env_root = os.environ.get("CMAPSS_XAI_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    starts = [Path.cwd(), Path(__file__).resolve()]
    for start in starts:
        candidates = [start, *start.parents]
        for candidate in candidates:
            if (candidate / "configs" / "dataset.yaml").exists():
                return candidate
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _discover_project_root()


def project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def ensure_dir(path: str | Path) -> Path:
    resolved = project_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
