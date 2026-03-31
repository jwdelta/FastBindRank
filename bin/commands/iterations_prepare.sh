#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  fastbindrank iterations prepare --workspace PATH
                                  --iteration N
                                  [--results-dir PATH]
                                  [--exclude-ids-file PATH]
                                  [--python PATH]
                                  [--workers N]
                                  [--training-split N]
                                  [--tuning-split N]
                                  [--test-split N]
                                  [--seed N]

Description:
  Prepare one iteration by:
  1. sampling split files
  2. extracting split-specific SMILES
  3. extracting split-specific Morgan fingerprints

Important behavior:
  - iteration 1 samples training, tuning, and test splits
  - iterations > 1 sample only the training split
  - iterations > 1 reuse tuning and test splits from iteration 1

Requirements:
  The prepared library must already exist under:
    <workspace>/library/smiles.txt
    <workspace>/library/smiles_chunks/
    <workspace>/library/morgan_fingerprints/

Output location:
  Iteration outputs are written under:
    <results-dir>/iteration_<N>/
  If --results-dir is not given, the default is:
    <workspace>/results/
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bin_dir="$(cd "$script_dir/.." && pwd)"
workspace=""
iteration=""
results_dir=""
exclude_ids_file=""
python_bin="python3"
workers="10"
training_split="250000"
tuning_split="100000"
test_split="250000"
seed="42"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) workspace="$2"; shift 2 ;;
    --iteration) iteration="$2"; shift 2 ;;
    --results-dir) results_dir="$2"; shift 2 ;;
    --exclude-ids-file) exclude_ids_file="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    --workers) workers="$2"; shift 2 ;;
    --training-split) training_split="$2"; shift 2 ;;
    --tuning-split) tuning_split="$2"; shift 2 ;;
    --test-split) test_split="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$workspace" || -z "$iteration" ]]; then
  usage >&2
  exit 1
fi

library_dir="$workspace/library"
smiles_table="$library_dir/smiles.txt"
smiles_chunks_dir="$library_dir/smiles_chunks"
morgan_library_dir="$library_dir/morgan_fingerprints"
project_dir="${results_dir:-$workspace/results}"
iteration_dir="$project_dir/iteration_${iteration}"
split_dir="$iteration_dir/splits"
smile_dir="$iteration_dir/smiles"
fingerprint_dir="$iteration_dir/fingerprints"
first_iteration_dir="$project_dir/iteration_1"
first_smile_dir="$first_iteration_dir/smiles"
first_fingerprint_dir="$first_iteration_dir/fingerprints"

if [[ ! -f "$smiles_table" || ! -d "$smiles_chunks_dir" || ! -d "$morgan_library_dir" ]]; then
  echo "Error: library assets are missing under $library_dir" >&2
  exit 1
fi

sample_helper="$bin_dir/helpers/sample_splits.py"
extract_smiles_helper="$bin_dir/helpers/extract_split_smiles.py"
extract_morgan_helper="$bin_dir/helpers/extract_split_morgan.py"

mkdir -p "$split_dir" "$smile_dir" "$fingerprint_dir"

sample_cmd=(
  "$python_bin" "$sample_helper"
  --input-file "$smiles_table"
  --output-dir "$split_dir"
  --results-dir "$project_dir"
  --training-size "$training_split"
  --tuning-size "$tuning_split"
  --test-size "$test_split"
  --iteration "$iteration"
  --seed "$seed"
  --workers "$workers"
)

if [[ -n "$exclude_ids_file" ]]; then
  sample_cmd+=(--exclude-ids-file "$exclude_ids_file")
fi

"${sample_cmd[@]}"

"$python_bin" "$extract_smiles_helper" --smiles-library "$smiles_chunks_dir" --split-file "$split_dir/training_split.txt" --output-file "$smile_dir/training_split.smi" --workers "$workers"
"$python_bin" "$extract_morgan_helper" --morgan-library "$morgan_library_dir" --split-file "$split_dir/training_split.txt" --output-file "$fingerprint_dir/training_split_morgan.csv" --workers "$workers"

if [[ "$iteration" == "1" ]]; then
  "$python_bin" "$extract_smiles_helper" --smiles-library "$smiles_chunks_dir" --split-file "$split_dir/tuning_split.txt" --output-file "$smile_dir/tuning_split.smi" --workers "$workers"
  "$python_bin" "$extract_smiles_helper" --smiles-library "$smiles_chunks_dir" --split-file "$split_dir/test_split.txt" --output-file "$smile_dir/test_split.smi" --workers "$workers"

  "$python_bin" "$extract_morgan_helper" --morgan-library "$morgan_library_dir" --split-file "$split_dir/tuning_split.txt" --output-file "$fingerprint_dir/tuning_split_morgan.csv" --workers "$workers"
  "$python_bin" "$extract_morgan_helper" --morgan-library "$morgan_library_dir" --split-file "$split_dir/test_split.txt" --output-file "$fingerprint_dir/test_split_morgan.csv" --workers "$workers"
else
  for required_file in \
    "$first_smile_dir/tuning_split.smi" \
    "$first_smile_dir/test_split.smi" \
    "$first_fingerprint_dir/tuning_split_morgan.csv" \
    "$first_fingerprint_dir/test_split_morgan.csv"
  do
    if [[ ! -f "$required_file" ]]; then
      echo "Error: required iteration_1 file not found: $required_file" >&2
      exit 1
    fi
  done

  cp "$first_smile_dir/tuning_split.smi" "$smile_dir/tuning_split.smi"
  cp "$first_smile_dir/test_split.smi" "$smile_dir/test_split.smi"
  cp "$first_fingerprint_dir/tuning_split_morgan.csv" "$fingerprint_dir/tuning_split_morgan.csv"
  cp "$first_fingerprint_dir/test_split_morgan.csv" "$fingerprint_dir/test_split_morgan.csv"
fi

echo "Prepared iteration: $iteration"
echo "Split directory:    $split_dir"
echo "SMILES directory:   $smile_dir"
echo "Fingerprint dir:    $fingerprint_dir"
