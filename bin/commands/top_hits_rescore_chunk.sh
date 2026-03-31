#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  fastbindrank top-hits rescore-chunk --workspace PATH
                                      --protein-name NAME
                                      --protein-yaml-template PATH
                                      --start-line N
                                      --end-line N
                                      [--results-dir PATH]
                                      [--selection-dir PATH]
                                      [--boltz PATH]
                                      [--boltz-threads N]
                                      [--accelerator NAME]

Description:
  Run Boltz-2 rescoring for one chunk of selected top-hits.
  The YAML template should already contain the protein definition(s),
  ligand block, and affinity property block. Use __LIGAND_SMILES__ as the
  placeholder that will be replaced for each compound.
EOF
}

workspace=""
protein_name=""
protein_yaml_template=""
start_line=""
end_line=""
results_dir=""
selection_dir=""
boltz_bin="boltz"
boltz_threads="1"
accelerator="gpu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) workspace="$2"; shift 2 ;;
    --protein-name) protein_name="$2"; shift 2 ;;
    --protein-yaml-template) protein_yaml_template="$2"; shift 2 ;;
    --start-line) start_line="$2"; shift 2 ;;
    --end-line) end_line="$2"; shift 2 ;;
    --results-dir) results_dir="$2"; shift 2 ;;
    --selection-dir) selection_dir="$2"; shift 2 ;;
    --boltz) boltz_bin="$2"; shift 2 ;;
    --boltz-threads) boltz_threads="$2"; shift 2 ;;
    --accelerator) accelerator="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$workspace" || -z "$protein_name" || -z "$protein_yaml_template" || -z "$start_line" || -z "$end_line" ]]; then
  usage >&2
  exit 1
fi

if [[ -z "$selection_dir" ]]; then
  resolved_results_dir="${results_dir:-$workspace/results}"
  selection_dir="$(find "$resolved_results_dir/best_model" -maxdepth 1 -mindepth 1 -type d | sort -V | tail -n 1)"
fi

top_hits_smiles="$selection_dir/top_hits/top_hits_smiles.txt"
chunks_root="$selection_dir/top_hits/boltz_rescoring/chunks"
chunk_name="${start_line}_${end_line}"
chunk_dir="$chunks_root/${chunk_name}"
chunk_output="$chunks_root/boltz_rescore_${start_line}_${end_line}.tsv"

if [[ ! -f "$protein_yaml_template" ]]; then
  echo "Error: protein YAML template not found: $protein_yaml_template" >&2
  exit 1
fi

mkdir -p "$chunk_dir"

chunk_total=$(( end_line - start_line + 1 ))

{
  printf "CID\tconfidence_score\tptm\tiptm\tligand_iptm\taffinity_pred_log10_IC50\taffinity_probability_binary\truntime_seconds\n"
  chunk_index=0
  sed -n "${start_line},${end_line}p" "$top_hits_smiles" | while read -r smiles compound_id; do
    [[ -z "${smiles:-}" || -z "${compound_id:-}" ]] && continue
    chunk_index=$((chunk_index + 1))
    ligand_start_time="$(date +%s)"

    compound_tmp_dir="$chunk_dir/tmp_${compound_id}"
    compound_input_dir="$compound_tmp_dir/input"
    compound_output_dir="$compound_tmp_dir/output"
    mkdir -p "$compound_input_dir" "$compound_output_dir"

    yaml_file="$compound_input_dir/${protein_name}_${compound_id}.yaml"
    python3 - "$protein_yaml_template" "$yaml_file" "$smiles" "$compound_id" <<'EOF'
from pathlib import Path
import sys

template_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
smiles = sys.argv[3]
compound_id = sys.argv[4]

template = template_path.read_text()
if "__LIGAND_SMILES__" not in template:
    raise SystemExit("Protein YAML template must contain __LIGAND_SMILES__.")

escaped_smiles = smiles.replace("'", "''")
rendered = template.replace("__LIGAND_SMILES__", escaped_smiles)
rendered = rendered.replace("__LIGAND_ID__", compound_id)
output_path.write_text(rendered if rendered.endswith("\n") else rendered + "\n")
EOF

    echo "[top-hits rescore-chunk] ${start_line}-${end_line}: ${chunk_index}/${chunk_total} CID=${compound_id}" >&2
    "$boltz_bin" predict "$yaml_file" --use_msa_server --accelerator "$accelerator" --out_dir "$compound_output_dir" --preprocessing-threads "$boltz_threads" >&2

    confidence_file="$compound_output_dir/boltz_results_${protein_name}_${compound_id}/predictions/${protein_name}_${compound_id}/confidence_${protein_name}_${compound_id}_model_0.json"
    affinity_file="$compound_output_dir/boltz_results_${protein_name}_${compound_id}/predictions/${protein_name}_${compound_id}/affinity_${protein_name}_${compound_id}.json"

    confidence_score=""
    ptm=""
    iptm=""
    ligand_iptm=""
    affinity_pred_log10_ic50=""
    affinity_probability_binary=""

    if [[ -f "$confidence_file" ]]; then
      confidence_score="$(grep -o '"confidence_score"[[:space:]]*:[[:space:]]*[0-9.e+-]*' "$confidence_file" | head -n1 | awk -F: '{print $2}' | tr -d ' ')"
      ptm="$(grep -o '"ptm"[[:space:]]*:[[:space:]]*[0-9.e+-]*' "$confidence_file" | head -n1 | awk -F: '{print $2}' | tr -d ' ')"
      iptm="$(grep -o '"iptm"[[:space:]]*:[[:space:]]*[0-9.e+-]*' "$confidence_file" | head -n1 | awk -F: '{print $2}' | tr -d ' ')"
      ligand_iptm="$(grep -o '"ligand_iptm"[[:space:]]*:[[:space:]]*[0-9.e+-]*' "$confidence_file" | head -n1 | awk -F: '{print $2}' | tr -d ' ')"
    fi
    if [[ -f "$affinity_file" ]]; then
      affinity_pred_log10_ic50="$(grep -o '"affinity_pred_value"[[:space:]]*:[[:space:]]*[0-9.e+-]*' "$affinity_file" | head -n1 | awk -F: '{print $2}' | tr -d ' ')"
      affinity_probability_binary="$(grep -o '"affinity_probability_binary"[[:space:]]*:[[:space:]]*[0-9.e+-]*' "$affinity_file" | head -n1 | awk -F: '{print $2}' | tr -d ' ')"
    fi

    ligand_end_time="$(date +%s)"
    ligand_elapsed=$((ligand_end_time - ligand_start_time))

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$compound_id" "$confidence_score" "$ptm" "$iptm" "$ligand_iptm" "$affinity_pred_log10_ic50" "$affinity_probability_binary" "$ligand_elapsed"

    echo "[top-hits rescore-chunk] ${start_line}-${end_line}: ${chunk_index}/${chunk_total} CID=${compound_id} finished in ${ligand_elapsed}s" >&2

    rm -rf "$compound_tmp_dir"
  done
} > "$chunk_output"

rm -rf "$chunk_dir"

echo "Top-hit chunk rescoring written to: $chunk_output"
