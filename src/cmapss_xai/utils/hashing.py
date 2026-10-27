from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: str(value)):
        if not path.exists():
            continue
        digest.update(str(path.name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def write_dataset_hash(paths: list[Path], output_path: Path) -> str:
    digest = sha256_files(paths)
    output_path.write_text(f"{digest}\n", encoding="utf-8")
    return digest
