#!/usr/bin/env python3

import argparse
import glob
import os
from multiprocessing import Pool, cpu_count


def load_ids(file_path: str) -> set[str]:
    with open(file_path, "r") as handle:
        return set(line.strip().split()[-1] for line in handle if line.strip())


def extract_file(file_path: str, selected_ids: set[str]):
    results = []
    seen = set()
    with open(file_path, "r") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            smiles, compound_id = parts
            if compound_id in selected_ids and compound_id not in seen:
                results.append(f"{smiles}\t{compound_id}\n")
                seen.add(compound_id)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles-library", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--workers", type=int, default=cpu_count())
    args = parser.parse_args()

    selected_ids = load_ids(args.split_file)
    files = glob.glob(os.path.join(args.smiles_library, "*.txt"))

    with Pool(processes=min(max(len(files), 1), args.workers)) as pool:
        results = pool.starmap(extract_file, [(file_path, selected_ids) for file_path in files])

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as handle:
        for chunk in results:
            handle.writelines(chunk)


if __name__ == "__main__":
    main()
