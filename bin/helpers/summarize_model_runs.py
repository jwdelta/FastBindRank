#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path


ADAM_RE = re.compile(r"^hidims(?P<hidden>.+)_dropout(?P<dropout>[^_]+)_adam_lr(?P<lr>.+)$")
ADAMW_RE = re.compile(
    r"^hidims(?P<hidden>.+)_dropout(?P<dropout>[^_]+)_adamw_lr(?P<lr>.+)_weight_decay(?P<wd>.+)$"
)
SGD_RE = re.compile(
    r"^hidims(?P<hidden>.+)_dropout(?P<dropout>[^_]+)_sgd_lr(?P<lr>.+)_weight_decay(?P<wd>.+)_momentum(?P<momentum>.+)$"
)


def parse_run_name(name: str) -> dict[str, str]:
    for optimizer, pattern in (("adam", ADAM_RE), ("adamw", ADAMW_RE), ("sgd", SGD_RE)):
        match = pattern.match(name)
        if match:
            data = match.groupdict()
            data["optimizer"] = optimizer
            data.setdefault("wd", "")
            data.setdefault("momentum", "")
            return data
    return {"hidden": "", "dropout": "", "optimizer": "", "lr": "", "wd": "", "momentum": ""}


def parse_metrics_file(path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    if not path.exists():
        return metrics
    with path.open() as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                metrics[row[0]] = row[1]
    return metrics


def first_existing(run_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def metric_value(run_dir: Path, names: list[str], key: str) -> str:
    metrics_path = first_existing(run_dir, names)
    if metrics_path is None:
        return "NA"
    metrics = parse_metrics_file(metrics_path)
    return metrics.get(key, "NA")


def float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_tree_if_exists(src: Path, dst: Path) -> None:
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summary_dir = results_dir / "model_selection"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / "soft_label_model_summary.tsv"

    rows: list[dict[str, str]] = []
    iteration_dirs = sorted(
        [path for path in results_dir.glob("iteration_*") if path.is_dir()],
        key=lambda path: int(path.name.split("_")[-1]),
    )

    for iteration_dir in iteration_dirs:
        iteration = iteration_dir.name.split("_")[-1]
        model_root = iteration_dir / "model" / "output_soft_labels"
        if not model_root.exists():
            continue
        for run_dir in sorted([path for path in model_root.iterdir() if path.is_dir()]):
            params = parse_run_name(run_dir.name)
            row = {
                "iteration": iteration,
                "run_name": run_dir.name,
                "hidden_dims": params["hidden"],
                "dropout": params["dropout"],
                "optimizer": params["optimizer"],
                "learning_rate": params["lr"],
                "weight_decay": params["wd"],
                "momentum": params["momentum"],
                "training_soft_pr_auc": metric_value(run_dir, ["training_split_metrics.txt", "train_metrics.txt"], "soft_pr_auc"),
                "tuning_soft_pr_auc": metric_value(run_dir, ["tuning_split_metrics.txt", "val_metrics.txt"], "soft_pr_auc"),
                "test_soft_pr_auc": metric_value(run_dir, ["test_split_metrics.txt", "test_metrics.txt"], "soft_pr_auc"),
                "run_dir": str(run_dir),
            }
            rows.append(row)

    with summary_file.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iteration",
                "run_name",
                "hidden_dims",
                "dropout",
                "optimizer",
                "learning_rate",
                "weight_decay",
                "momentum",
                "training_soft_pr_auc",
                "tuning_soft_pr_auc",
                "test_soft_pr_auc",
                "run_dir",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    if not iteration_dirs:
        print(f"Summary written to: {summary_file}")
        print("No iteration directories were found.")
        return

    last_iteration = iteration_dirs[-1].name.split("_")[-1]
    last_iteration_rows = [row for row in rows if row["iteration"] == last_iteration]

    best_row = None
    best_score = None
    for row in last_iteration_rows:
        score = float_or_none(row["tuning_soft_pr_auc"])
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_row = row

    print(f"Summary written to: {summary_file}")

    if best_row is None:
        print(f"No valid tuning soft_pr_auc was found for iteration_{last_iteration}.")
        return

    selected_dir = results_dir / "best_model" / f"iteration_{last_iteration}_best_tuning_soft_pr_auc"
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    selected_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(best_row["run_dir"])
    copy_tree_if_exists(run_dir, selected_dir / "model_artifacts")

    iteration_dir = results_dir / f"iteration_{last_iteration}"
    first_iteration_dir = results_dir / "iteration_1"

    copy_if_exists(iteration_dir / "fingerprints" / "training_split_morgan.csv", selected_dir / "model_inputs" / "training_split_morgan.csv")
    copy_if_exists(first_iteration_dir / "fingerprints" / "tuning_split_morgan.csv", selected_dir / "model_inputs" / "tuning_split_morgan.csv")
    copy_if_exists(first_iteration_dir / "fingerprints" / "test_split_morgan.csv", selected_dir / "model_inputs" / "test_split_morgan.csv")
    copy_if_exists(iteration_dir / "affinity_prediction" / "training_split_affinity.tsv", selected_dir / "affinity_prediction" / "training_split_affinity.tsv")
    copy_if_exists(first_iteration_dir / "affinity_prediction" / "tuning_split_affinity.tsv", selected_dir / "affinity_prediction" / "tuning_split_affinity.tsv")
    copy_if_exists(first_iteration_dir / "affinity_prediction" / "test_split_affinity.tsv", selected_dir / "affinity_prediction" / "test_split_affinity.tsv")
    copy_if_exists(iteration_dir / "affinity_prediction" / "training_split_affinity.tsv", selected_dir / "model_inputs" / "training_split_affinity.tsv")
    copy_if_exists(first_iteration_dir / "affinity_prediction" / "tuning_split_affinity.tsv", selected_dir / "model_inputs" / "tuning_split_affinity.tsv")
    copy_if_exists(first_iteration_dir / "affinity_prediction" / "test_split_affinity.tsv", selected_dir / "model_inputs" / "test_split_affinity.tsv")

    copy_if_exists(iteration_dir / "splits" / "training_split.txt", selected_dir / "splits" / "training_split.txt")
    copy_if_exists(first_iteration_dir / "splits" / "tuning_split.txt", selected_dir / "splits" / "tuning_split.txt")
    copy_if_exists(first_iteration_dir / "splits" / "test_split.txt", selected_dir / "splits" / "test_split.txt")

    manifest = selected_dir / "best_model_summary.tsv"
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["field", "value"])
        for key in [
            "iteration",
            "run_name",
            "hidden_dims",
            "dropout",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "momentum",
            "training_soft_pr_auc",
            "tuning_soft_pr_auc",
            "test_soft_pr_auc",
            "run_dir",
        ]:
            writer.writerow([key, best_row[key]])

    print(f"Best model for iteration_{last_iteration}: {best_row['run_name']}")
    print(f"Best tuning soft_pr_auc: {best_row['tuning_soft_pr_auc']}")
    print(f"Selected model bundle: {selected_dir}")


if __name__ == "__main__":
    main()
