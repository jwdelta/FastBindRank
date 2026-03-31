#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  fastbindrank results predict-all-best --workspace PATH
                                        [--results-dir PATH]
                                        [--selection-dir PATH]
                                        [--python PATH]

Description:
  Use the selected best model to predict binding probabilities for every Morgan
  fingerprint chunk in the prepared library, then merge and sort the outputs.

Inputs:
  - library Morgan fingerprints under:
      <workspace>/library/morgan_fingerprints/
  - selected best model bundle produced by:
      fastbindrank results summarize-models

Outputs:
  <selection-dir>/library_predictions/
    chunks/
    library_predictions.csv
    library_predictions_ranked.csv
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bin_dir="$(cd "$script_dir/.." && pwd)"
workspace=""
results_dir=""
selection_dir=""
python_bin="python3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) workspace="$2"; shift 2 ;;
    --results-dir) results_dir="$2"; shift 2 ;;
    --selection-dir) selection_dir="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
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

if [[ -z "$workspace" ]]; then
  usage >&2
  exit 1
fi

resolved_results_dir="${results_dir:-$workspace/results}"
if [[ -z "$selection_dir" ]]; then
  selection_dir="$(find "$resolved_results_dir/best_model" -maxdepth 1 -mindepth 1 -type d | sort -V | tail -n 1)"
fi

if [[ -z "${selection_dir:-}" || ! -d "$selection_dir" ]]; then
  echo "Error: selected model directory not found. Run summarize-models first or pass --selection-dir." >&2
  exit 1
fi

train_helper="$bin_dir/helpers/train_dnn_morgan.py"
model_run_dir="$selection_dir/model_artifacts"
library_morgan_dir="$workspace/library/morgan_fingerprints"
prediction_root="$selection_dir/library_predictions"
prediction_chunks_dir="$prediction_root/chunks"
merged_csv="$prediction_root/library_predictions.csv"
sorted_csv="$prediction_root/library_predictions_ranked.csv"

train_fp="$selection_dir/model_inputs/training_split_morgan.csv"
train_labels="$selection_dir/model_inputs/training_split_affinity.tsv"
val_fp="$selection_dir/model_inputs/tuning_split_morgan.csv"
val_labels="$selection_dir/model_inputs/tuning_split_affinity.tsv"

for required_path in "$train_helper" "$model_run_dir" "$library_morgan_dir" "$train_fp" "$train_labels" "$val_fp" "$val_labels"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Error: required path not found: $required_path" >&2
    exit 1
  fi
done

rm -rf "$prediction_root"
mkdir -p "$prediction_chunks_dir"

find "$library_morgan_dir" -maxdepth 1 -type f | sort -V | while read -r chunk_file; do
  chunk_name="$(basename "$chunk_file")"
  chunk_prefix="${chunk_name%.*}"

  "$python_bin" "$train_helper" \
    --train_fp "$train_fp" \
    --train_labels "$train_labels" \
    --val_fp "$val_fp" \
    --val_labels "$val_labels" \
    --test_fp "$chunk_file" \
    --output_dir "$model_run_dir" \
    --mode test \
    --test_out_prefix "$chunk_prefix" \
    --save_probabilities

  generated_csv="$model_run_dir/${chunk_prefix}_pred_out.csv"
  if [[ -f "$generated_csv" ]]; then
    mv "$generated_csv" "$prediction_chunks_dir/"
  fi
done

{
  printf "ligand_id,prob\n"
  find "$prediction_chunks_dir" -maxdepth 1 -type f -name "*_pred_out.csv" | sort -V | while read -r chunk_csv; do
    sed -n '2,$p' "$chunk_csv"
  done
} > "$merged_csv"

{
  head -n 1 "$merged_csv"
  tail -n +2 "$merged_csv" | sort -t "," -k2 -r
} > "$sorted_csv"

echo "All-library chunk predictions: $prediction_chunks_dir"
echo "Merged predictions:            $merged_csv"
echo "Sorted predictions:            $sorted_csv"
