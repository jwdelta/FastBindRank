#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  fastbindrank results summarize-models --workspace PATH
                                        [--results-dir PATH]
                                        [--python PATH]

Description:
  Summarize all soft-label model runs across iterations, print the best model
  from the last iteration ranked by tuning split soft_pr_auc, and copy that
  model run plus key prediction inputs into a dedicated selection folder.

Default results location:
  If --results-dir is not given, the default is:
    <workspace>/results/

Outputs:
  <results-dir>/model_selection/soft_label_model_summary.tsv
  <results-dir>/best_model/iteration_<N>_best_tuning_soft_pr_auc/
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bin_dir="$(cd "$script_dir/.." && pwd)"
workspace=""
results_dir=""
python_bin="python3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) workspace="$2"; shift 2 ;;
    --results-dir) results_dir="$2"; shift 2 ;;
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
helper="$bin_dir/helpers/summarize_model_runs.py"

if [[ ! -f "$helper" ]]; then
  echo "Error: helper not found: $helper" >&2
  exit 1
fi

"$python_bin" "$helper" --results-dir "$resolved_results_dir"
