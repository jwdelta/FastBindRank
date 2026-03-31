#!/usr/bin/env python3

import argparse
import os
import random


def load_id_file(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, "r") as handle:
        ids = set()
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            ids.add(parts[-1])
        return ids


def load_used_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, "r") as handle:
        return set(line.strip().split("\t")[1] for line in handle if line.strip())


def write_used_ids(records, path: str, iteration: int) -> None:
    with open(path, "a") as handle:
        for compound_id, _ in records:
            handle.write(f"{iteration}\t{compound_id}\n")


def parse_record(line: str):
    parts = line.strip().split()
    if len(parts) < 2:
        return None
    compound_id = parts[-1]
    return compound_id, line


def reservoir_sample(input_file: str, excluded_ids: set[str], sample_size: int, seed: int):
    rng = random.Random(seed)
    reservoir = []
    seen_count = 0

    with open(input_file, "r") as handle:
        for line in handle:
            record = parse_record(line)
            if record is None:
                continue
            compound_id, raw_line = record
            if compound_id in excluded_ids:
                continue

            seen_count += 1
            if len(reservoir) < sample_size:
                reservoir.append((compound_id, raw_line))
            else:
                replacement_index = rng.randint(1, seen_count)
                if replacement_index <= sample_size:
                    reservoir[replacement_index - 1] = (compound_id, raw_line)

    if seen_count < sample_size:
        raise ValueError(f"Not enough compounds: needed {sample_size}, found {seen_count}.")

    return reservoir


def save_lines(records, output_file: str) -> None:
    with open(output_file, "w") as handle:
        for _, line in records:
            handle.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--training-size", type=int, required=True)
    parser.add_argument("--tuning-size", type=int, required=True)
    parser.add_argument("--test-size", type=int, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--exclude-ids-file", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results_dir = args.results_dir
    used_ids_path = os.path.join(results_dir, "used_compound_ids.tsv")
    used_ids = load_used_ids(used_ids_path)
    if args.exclude_ids_file:
        used_ids |= load_id_file(args.exclude_ids_file)

    training_file = os.path.join(args.output_dir, "training_split.txt")
    tuning_file = os.path.join(args.output_dir, "tuning_split.txt")
    test_file = os.path.join(args.output_dir, "test_split.txt")

    if args.iteration == 1:
        total_needed = args.training_size + args.tuning_size + args.test_size
        sampled = reservoir_sample(args.input_file, used_ids, total_needed, args.seed)

        training_records = sampled[: args.training_size]
        tuning_records = sampled[args.training_size : args.training_size + args.tuning_size]
        test_records = sampled[args.training_size + args.tuning_size :]

        save_lines(training_records, training_file)
        save_lines(tuning_records, tuning_file)
        save_lines(test_records, test_file)

        write_used_ids(sampled, used_ids_path, args.iteration)
    else:
        first_iteration_dir = os.path.join(results_dir, "iteration_1")
        first_split_dir = os.path.join(first_iteration_dir, "splits")
        first_tuning = os.path.join(first_split_dir, "tuning_split.txt")
        first_test = os.path.join(first_split_dir, "test_split.txt")
        if not os.path.exists(first_tuning) or not os.path.exists(first_test):
            raise FileNotFoundError("iteration_1 tuning/test split files are missing.")

        training_records = reservoir_sample(args.input_file, used_ids, args.training_size, args.seed)
        save_lines(training_records, training_file)

        with open(first_tuning, "r") as src, open(tuning_file, "w") as dst:
            dst.write(src.read())
        with open(first_test, "r") as src, open(test_file, "w") as dst:
            dst.write(src.read())

        write_used_ids(training_records, used_ids_path, args.iteration)


if __name__ == "__main__":
    main()
