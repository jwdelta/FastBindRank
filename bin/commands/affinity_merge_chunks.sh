#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  fastbindrank affinity merge-chunks --workspace PATH
                                     --protein-name NAME
                                     --iteration N
                                     --split NAME
                                     [--results-dir PATH]

Description:
  Merge chunk-level affinity prediction outputs for one iteration split into:
    <results-dir>/iteration_<N>/affinity_prediction/<split>_affinity.tsv

Default results location:
  If --results-dir is not given, the default is:
    <workspace>/results/
EOF
}

workspace=""
protein_name=""
iteration=""
split_name=""
results_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) workspace="$2"; shift 2 ;;
    --protein-name) protein_name="$2"; shift 2 ;;
    --iteration) iteration="$2"; shift 2 ;;
    --split) split_name="$2"; shift 2 ;;
    --results-dir) results_dir="$2"; shift 2 ;;
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

if [[ -z "$workspace" || -z "$protein_name" || -z "$iteration" || -z "$split_name" ]]; then
  usage >&2
  exit 1
fi

project_dir="${results_dir:-$workspace/results}"
iteration_dir="$project_dir/iteration_${iteration}"
chunks_root="$iteration_dir/affinity_prediction/chunks"
output_dir="$iteration_dir/affinity_prediction"
merged_file="$output_dir/${split_name}_affinity.tsv"

mkdir -p "$output_dir"

{
  printf "CID\tconfidence_score\tptm\tiptm\tligand_iptm\taffinity_pred_log10_IC50\taffinity_probability_binary\truntime_seconds\n"
  find "$chunks_root" -maxdepth 1 -type f -name "${split_name}_*.tsv" | sort -V | while read -r chunk_file; do
    if [[ -f "$chunk_file" ]]; then
      sed -n '2,$p' "$chunk_file"
    fi
  done
} > "$merged_file"

echo "Merged affinity file written to: $merged_file"
