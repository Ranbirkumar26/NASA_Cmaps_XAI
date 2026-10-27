from __future__ import annotations

from pathlib import Path

from cmapss_xai.paths import project_path
from cmapss_xai.utils.hashing import sha256_files, write_dataset_hash


def test_project_path_resolves_workspace_config() -> None:
    assert project_path("configs/dataset.yaml").exists()


def test_hashing_stable_over_same_files(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")
    digest_a = sha256_files([second, first])
    digest_b = sha256_files([first, second])
    assert digest_a == digest_b


def test_write_dataset_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    output = tmp_path / "dataset_hash.txt"
    source.write_text("value", encoding="utf-8")
    digest = write_dataset_hash([source], output)
    assert output.read_text(encoding="utf-8").strip() == digest
