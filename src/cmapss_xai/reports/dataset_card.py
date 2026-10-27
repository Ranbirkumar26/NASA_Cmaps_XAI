from __future__ import annotations

from pathlib import Path
from typing import Any


def make_dataset_card(metadata: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    card = output_dir / "dataset_card.md"
    lines = [
        "# NASA C-MAPSS dataset card",
        "",
        f"Dataset: {metadata['dataset_name']}",
        f"Subset: {metadata['subset']}",
        f"Dataset version: {metadata['dataset_version']}",
        f"Source: {metadata['source_page']}",
        f"Archive: {metadata['source_url']}",
        f"Rows: train {metadata['row_counts']['train']}, test {metadata['row_counts']['test']}",
        f"Units: train {metadata['unit_counts']['train']}, test {metadata['unit_counts']['test']}",
        f"RUL cap: {metadata['rul_cap']}",
        f"Window lengths: {', '.join(str(v) for v in metadata['window_lengths'])}",
        f"Dataset hash: {metadata.get('dataset_hash', 'not generated')}",
        "",
        "## Citation",
        "",
        metadata["citation"],
        "",
    ]
    card.write_text("\n".join(lines), encoding="utf-8")
    return card
