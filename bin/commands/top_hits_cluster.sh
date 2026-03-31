#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  fastbindrank top-hits cluster --workspace PATH
                                [--results-dir PATH]
                                [--selection-dir PATH]
                                [--python PATH]
                                [--boltz-prob-threshold FLOAT]
                                [--boltz-logic50-threshold FLOAT]
                                [--drug-like VALUE]
                                [--similarity-threshold FLOAT]

Description:
  Merge top-hit Boltz-2 rescoring chunks, annotate molecular properties,
  filter hits by the requested thresholds, and run structural diversity clustering.

Outputs:
  <selection-dir>/top_hits/
    boltz_rescoring/top_hits_boltz_rescored.tsv
    top_hits_annotated.csv
    top_hits_clustered.csv
    top_hits_clustered.summary.log
    top_hits_clustered.pca.pdf
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bin_dir="$(cd "$script_dir/.." && pwd)"
workspace=""
results_dir=""
selection_dir=""
python_bin="python3"
boltz_prob_threshold="0.65"
boltz_logic50_threshold="-0.5"
drug_like="Yes"
similarity_threshold="0.7"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) workspace="$2"; shift 2 ;;
    --results-dir) results_dir="$2"; shift 2 ;;
    --selection-dir) selection_dir="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    --boltz-prob-threshold) boltz_prob_threshold="$2"; shift 2 ;;
    --boltz-logic50-threshold) boltz_logic50_threshold="$2"; shift 2 ;;
    --drug-like) drug_like="$2"; shift 2 ;;
    --similarity-threshold) similarity_threshold="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$workspace" ]]; then
  usage >&2
  exit 1
fi

if [[ -z "$selection_dir" ]]; then
  resolved_results_dir="${results_dir:-$workspace/results}"
  selection_dir="$(find "$resolved_results_dir/best_model" -maxdepth 1 -mindepth 1 -type d | sort -V | tail -n 1)"
fi

top_hits_dir="$selection_dir/top_hits"
chunks_root="$top_hits_dir/boltz_rescoring/chunks"
merged_boltz="$top_hits_dir/boltz_rescoring/top_hits_boltz_rescored.tsv"
annotated_csv="$top_hits_dir/top_hits_annotated.csv"
clustered_csv="$top_hits_dir/top_hits_clustered.csv"
cluster_log="$top_hits_dir/top_hits_clustered.summary.log"
pca_pdf="$top_hits_dir/top_hits_clustered.pca.pdf"

mkdir -p "$top_hits_dir/boltz_rescoring"

{
  printf "CID\tconfidence_score\tptm\tiptm\tligand_iptm\taffinity_pred_log10_IC50\taffinity_probability_binary\truntime_seconds\n"
  find "$chunks_root" -maxdepth 2 -type f -name "boltz_rescore_*.tsv" | sort -V | while read -r chunk_file; do
    sed -n '2,$p' "$chunk_file"
  done
} > "$merged_boltz"

"$python_bin" "$bin_dir/helpers/annotate_top_hits.py" \
  --top-hits-table "$top_hits_dir/top_hits.csv" \
  --boltz-table "$merged_boltz" \
  --output-file "$annotated_csv"

"$python_bin" "$bin_dir/helpers/cluster_top_hits.py" \
  --input-file "$annotated_csv" \
  --output-file "$clustered_csv" \
  --summary-log "$cluster_log" \
  --pca-figure "$pca_pdf" \
  --prob-threshold "$boltz_prob_threshold" \
  --logic50-threshold "$boltz_logic50_threshold" \
  --drug-like "$drug_like" \
  --similarity-threshold "$similarity_threshold"

echo "Merged Boltz rescoring: $merged_boltz"
echo "Annotated top hits:     $annotated_csv"
echo "Clustered top hits:     $clustered_csv"
echo "Cluster summary log:    $cluster_log"
echo "Cluster PCA figure:     $pca_pdf"
