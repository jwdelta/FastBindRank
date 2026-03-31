#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load_target_rows(ranked_file: Path, top_k: int) -> list[dict[str, str]]:
    selected_rows: list[dict[str, str]] = []
    with ranked_file.open() as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if index >= top_k:
                break
            selected_rows.append(
                {
                    "rank": str(index + 1),
                    "CID": row["ligand_id"],
                    "FastBindRank_pred_prob": row["prob"],
                    "SMILES": "",
                }
            )
    return selected_rows


def load_smiles_mapping(smiles_file: Path, target_ids: set[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with smiles_file.open() as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            compound_id = parts[-1]
            if compound_id not in target_ids:
                continue
            smiles = " ".join(parts[:-1])
            mapping[compound_id] = smiles
            if len(mapping) == len(target_ids):
                break
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranked-predictions", required=True)
    parser.add_argument("--smiles-file", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--output-table", required=True)
    parser.add_argument("--output-smiles", required=True)
    args = parser.parse_args()

    ranked_path = Path(args.ranked_predictions)
    smiles_path = Path(args.smiles_file)
    output_table = Path(args.output_table)
    output_smiles = Path(args.output_smiles)

    print(f"[top-hits select] reading top {args.top_k} ranked compounds", file=sys.stderr, flush=True)
    selected_rows = load_target_rows(ranked_path, args.top_k)
    target_ids = {row["CID"] for row in selected_rows}
    print(
        f"[top-hits select] scanning smiles file for {len(target_ids)} selected CIDs",
        file=sys.stderr,
        flush=True,
    )
    smiles_mapping = load_smiles_mapping(smiles_path, target_ids)

    for row in selected_rows:
        row["SMILES"] = smiles_mapping.get(row["CID"], "")

    output_table.parent.mkdir(parents=True, exist_ok=True)
    with output_table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "CID", "FastBindRank_pred_prob", "SMILES"])
        writer.writeheader()
        writer.writerows(selected_rows)

    with output_smiles.open("w") as handle:
        for row in selected_rows:
            if row["SMILES"]:
                handle.write(f"{row['SMILES']}\t{row['CID']}\n")


if __name__ == "__main__":
    main()
