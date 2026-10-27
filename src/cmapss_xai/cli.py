from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console

from cmapss_xai.config import (
    load_dataset_config,
    load_preprocessing_config,
    override_subset,
)
from cmapss_xai.data.download import download_cmapss
from cmapss_xai.data.raw import extract_archive
from cmapss_xai.logging_utils import configure_logging
from cmapss_xai.pipeline import fingerprint_subset, prepare_subset, validate_subset
from cmapss_xai.xai.project import XaiRunConfig, run_xai_project


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dataset_config = override_subset(
        load_dataset_config(args.dataset_config), getattr(args, "subset", None)
    )
    preprocessing_config = load_preprocessing_config(args.preprocessing_config)
    configure_logging(args.command, dataset_config.reports_dir)
    console = Console()

    if args.command == "download":
        zip_path = download_cmapss(dataset_config, force=args.force)
        extract_archive(zip_path, dataset_config.raw_dir)
        console.print(f"Downloaded and extracted: {zip_path}")
        return 0
    if args.command == "prepare":
        prepare_subset(dataset_config, preprocessing_config)
        console.print(f"Prepared subset: {dataset_config.subset}")
        return 0
    if args.command == "validate":
        console.print("\n".join(validate_subset(dataset_config)))
        return 0
    if args.command == "fingerprint":
        console.print(fingerprint_subset(dataset_config))
        return 0
    if args.command == "run-xai":
        subsets = args.subsets or [dataset_config.subset]
        all_outputs = {}
        for subset in subsets:
            active_config = override_subset(dataset_config, subset)
            outputs = run_xai_project(
                active_config,
                preprocessing_config,
                XaiRunConfig(
                    subset=active_config.subset,
                    window_length=args.window_length,
                    max_train_windows=args.max_train_windows,
                    max_eval_windows=args.max_eval_windows,
                    xai_samples=args.xai_samples,
                    random_state=args.seed,
                    include_deep_models=args.include_deep_models,
                    deep_epochs=args.deep_epochs,
                    deep_batch_size=args.deep_batch_size,
                    deep_hidden_size=args.deep_hidden_size,
                    deep_learning_rate=args.deep_learning_rate,
                    deep_dropout=args.deep_dropout,
                    exact_xai=args.exact_xai,
                    exact_xai_background=args.exact_xai_background,
                    exact_shap_samples=args.exact_shap_samples,
                    exact_lime_samples=args.exact_lime_samples,
                    exact_dice_samples=args.exact_dice_samples,
                    integrated_gradients=args.integrated_gradients,
                    benchmark_explainers=args.benchmark_explainers,
                    prediction_intervals=args.prediction_intervals,
                    tuning_grid=args.tuning_grid,
                ),
            )
            all_outputs[active_config.subset] = outputs
            for name, path in outputs.items():
                console.print(f"{active_config.subset} {name}: {path}")
        if len(all_outputs) > 1:
            cross_subset = write_cross_subset_benchmark(dataset_config.results_dir, all_outputs)
            console.print(f"cross_subset_benchmark: {cross_subset}")
        return 0
    raise ValueError(f"Unknown command: {args.command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cmapss-xai")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--preprocessing-config", default="configs/preprocessing.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download")
    download.add_argument("--force", action="store_true")

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--subset", default=None)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--subset", default=None)

    fingerprint = subparsers.add_parser("fingerprint")
    fingerprint.add_argument("--subset", default=None)

    run_xai = subparsers.add_parser("run-xai")
    run_xai.add_argument("--subset", default=None)
    run_xai.add_argument("--subsets", nargs="+", default=None)
    run_xai.add_argument("--window-length", type=int, default=30)
    run_xai.add_argument("--max-train-windows", type=int, default=6000)
    run_xai.add_argument("--max-eval-windows", type=int, default=3000)
    run_xai.add_argument("--xai-samples", type=int, default=40)
    run_xai.add_argument("--seed", type=int, default=42)
    run_xai.add_argument("--include-deep-models", action="store_true")
    run_xai.add_argument("--deep-epochs", type=int, default=4)
    run_xai.add_argument("--deep-batch-size", type=int, default=128)
    run_xai.add_argument("--deep-hidden-size", type=int, default=48)
    run_xai.add_argument("--deep-learning-rate", type=float, default=1e-3)
    run_xai.add_argument("--deep-dropout", type=float, default=0.1)
    run_xai.add_argument("--exact-xai", action="store_true")
    run_xai.add_argument("--exact-xai-background", type=int, default=64)
    run_xai.add_argument("--exact-shap-samples", type=int, default=4)
    run_xai.add_argument("--exact-lime-samples", type=int, default=8)
    run_xai.add_argument("--exact-dice-samples", type=int, default=2)
    run_xai.add_argument("--integrated-gradients", action="store_true")
    run_xai.add_argument("--benchmark-explainers", action="store_true")
    run_xai.add_argument("--prediction-intervals", action="store_true")
    run_xai.add_argument("--tuning-grid", action="store_true")

    return parser


def write_cross_subset_benchmark(
    results_dir: Path,
    all_outputs: dict[str, dict[str, Path]],
) -> Path:
    rows = []
    for subset, outputs in all_outputs.items():
        metrics = pd.read_csv(outputs["model_metrics"])
        best = metrics.sort_values("validation_rmse").iloc[0].to_dict()
        best["subset"] = subset
        rows.append(best)
    frame = pd.DataFrame(rows)
    output_dir = results_dir / "xai"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cross_subset_benchmark.csv"
    frame.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
