#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  fastbindrank top-hits select --workspace PATH
                               [--results-dir PATH]
                               [--selection-dir PATH]
                               --top-k N
                               [--python PATH]

Description:
  Select the top N compounds from the ranked full-library predictions and
  prepare a SMILES table for Boltz-2 rescoring.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bin_dir="$(cd "$script_dir/.." && pwd)"
workspace=""
results_dir=""
selection_dir=""
top_k=""
python_bin="python3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) workspace="$2"; shift 2 ;;
    --results-dir) results_dir="$2"; shift 2 ;;
    --selection-dir) selection_dir="$2"; shift 2 ;;
    --top-k) top_k="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$workspace" || -z "$top_k" ]]; then
  usage >&2
  exit 1
fi

resolved_results_dir="${results_dir:-$workspace/results}"
if [[ -z "$selection_dir" ]]; then
  selection_dir="$(find "$resolved_results_dir/best_model" -maxdepth 1 -mindepth 1 -type d | sort -V | tail -n 1)"
fi

ranked_predictions="$selection_dir/library_predictions/library_predictions_ranked.csv"
smiles_file="$workspace/library/smiles.txt"
top_hits_dir="$selection_dir/top_hits"

mkdir -p "$top_hits_dir"

"$python_bin" "$bin_dir/helpers/select_top_hits.py" \
  --ranked-predictions "$ranked_predictions" \
  --smiles-file "$smiles_file" \
  --top-k "$top_k" \
  --output-table "$top_hits_dir/top_hits.csv" \
  --output-smiles "$top_hits_dir/top_hits_smiles.txt"

echo "Top-hits table:  $top_hits_dir/top_hits.csv"
echo "Top-hits smiles: $top_hits_dir/top_hits_smiles.txt"
