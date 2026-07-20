#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  fastbindrank iterations train --workspace PATH
                                --iteration N
                                [--results-dir PATH]
                                [--python PATH]
                                [--epochs N]
                                [--batch-size N]
                                [--train-workers N]
                                [--early-stop-patience N]
                                [--grid-shard-index N]
                                [--grid-shard-count N]

Description:
  Train one iteration after affinity prediction files are ready.
  This command runs the full predefined hyperparameter sweep used by FastBindRank.
  hard_label is fixed to false.

Required affinity files:
  <results-dir>/iteration_<N>/affinity_prediction/training_split_affinity.tsv
  <results-dir>/iteration_1/affinity_prediction/tuning_split_affinity.tsv
  <results-dir>/iteration_1/affinity_prediction/test_split_affinity.tsv

Important behavior:
  - the training split always comes from the requested iteration
  - tuning and test always come from iteration 1
  - if iteration > 1 and a previous checkpoint exists, training finetunes from it
  - --grid-shard-index and --grid-shard-count split the hyperparameter grid
    across multiple jobs; by default, one job runs the full grid

Default results location:
  If --results-dir is not given, the default is:
    <workspace>/results/
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bin_dir="$(cd "$script_dir/.." && pwd)"
workspace=""
iteration=""
results_dir=""
python_bin="python3"
epochs="1000"
batch_size="512"
train_workers="1"
early_stop_patience="3"
grid_shard_index="0"
grid_shard_count="1"

# Full predefined search space for FastBindRank model training.
hard_label="false"
hidden_list=("1024,512,256" "2048,1024,512" "2048,1024,512,256" "4096,2048,1024,512")
dropouts=("0" "0.1" "0.2" "0.3")
optimizers=("adam" "adamw" "sgd")
lrs=("1e-6" "1e-5" "1e-4" "1e-3")
wds=("1e-6" "1e-5" "1e-4" "1e-3" "1e-2")
momentums=("0.9")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) workspace="$2"; shift 2 ;;
    --iteration) iteration="$2"; shift 2 ;;
    --results-dir) results_dir="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --train-workers) train_workers="$2"; shift 2 ;;
    --early-stop-patience) early_stop_patience="$2"; shift 2 ;;
    --grid-shard-index) grid_shard_index="$2"; shift 2 ;;
    --grid-shard-count) grid_shard_count="$2"; shift 2 ;;
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

if ! [[ "$grid_shard_index" =~ ^[0-9]+$ && "$grid_shard_count" =~ ^[0-9]+$ ]]; then
  echo "Error: --grid-shard-index and --grid-shard-count must be non-negative integers." >&2
  exit 1
fi
if [[ "$grid_shard_count" -lt 1 ]]; then
  echo "Error: --grid-shard-count must be at least 1." >&2
  exit 1
fi
if [[ "$grid_shard_index" -ge "$grid_shard_count" ]]; then
  echo "Error: --grid-shard-index must be smaller than --grid-shard-count." >&2
  exit 1
fi
grid_shard_index=$((10#$grid_shard_index))
grid_shard_count=$((10#$grid_shard_count))

project_dir="${results_dir:-$workspace/results}"
iteration_dir="$project_dir/iteration_${iteration}"
first_iteration_dir="$project_dir/iteration_1"
fingerprint_dir="$iteration_dir/fingerprints"
model_dir="$iteration_dir/model"
train_helper="$bin_dir/helpers/train_dnn_morgan.py"

train_fp="$fingerprint_dir/training_split_morgan.csv"
train_labels="$iteration_dir/affinity_prediction/training_split_affinity.tsv"
val_fp="$first_iteration_dir/fingerprints/tuning_split_morgan.csv"
val_labels="$first_iteration_dir/affinity_prediction/tuning_split_affinity.tsv"
test_fp="$first_iteration_dir/fingerprints/test_split_morgan.csv"
test_labels="$first_iteration_dir/affinity_prediction/test_split_affinity.tsv"

for required_file in "$train_fp" "$train_labels" "$val_fp" "$val_labels" "$test_fp" "$test_labels" "$train_helper"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Error: required file not found: $required_file" >&2
    exit 1
  fi
done

mkdir -p "$model_dir"

should_run_config() {
  local config_index="$1"
  (( config_index % grid_shard_count == grid_shard_index ))
}

config_index=0

# Run the full search space. Each configuration writes to its own output directory
# so the results remain separated and iteration-to-iteration finetuning can reuse
# the matching checkpoint from the previous iteration when available.
for hidden_dims in "${hidden_list[@]}"; do
  for dropout in "${dropouts[@]}"; do
    for optimizer in "${optimizers[@]}"; do
      case "$optimizer" in
        adam)
          for learning_rate in "${lrs[@]}"; do
            run_output_dir="$model_dir/output_soft_labels/hidims${hidden_dims}_dropout${dropout}_${optimizer}_lr${learning_rate}"
            current_config_index="$config_index"
            config_index=$((config_index + 1))
            if ! should_run_config "$current_config_index"; then
              continue
            fi
            echo "[iterations train] running config ${current_config_index}: hidden=${hidden_dims} dropout=${dropout} optimizer=${optimizer} lr=${learning_rate}" >&2
            train_cmd=(
              "$python_bin" "$train_helper"
              --train_fp "$train_fp"
              --train_labels "$train_labels"
              --val_fp "$val_fp"
              --val_labels "$val_labels"
              --test_fp "$test_fp"
              --test_labels "$test_labels"
              --output_dir "$run_output_dir"
              --hidden_dims "$hidden_dims"
              --dropout "$dropout"
              --batch_size "$batch_size"
              --epochs "$epochs"
              --optimizer "$optimizer"
              --lr "$learning_rate"
              --num_workers "$train_workers"
              --early_stop_patience "$early_stop_patience"
              --save_probabilities
              --mode train
            )

            if [[ "$iteration" -gt 1 ]]; then
              previous_ckpt="$project_dir/iteration_$((iteration - 1))/model/output_soft_labels/hidims${hidden_dims}_dropout${dropout}_${optimizer}_lr${learning_rate}/ckpts/best_model.pt"
              if [[ -f "$previous_ckpt" ]]; then
                train_cmd+=(--finetune_from "$previous_ckpt")
              fi
            fi

            "${train_cmd[@]}"

            for eval_name in training_split tuning_split test_split; do
              case "$eval_name" in
                training_split)
                  eval_fp="$train_fp"
                  eval_labels="$train_labels"
                  ;;
                tuning_split)
                  eval_fp="$val_fp"
                  eval_labels="$val_labels"
                  ;;
                test_split)
                  eval_fp="$test_fp"
                  eval_labels="$test_labels"
                  ;;
              esac

              test_cmd=(
                "$python_bin" "$train_helper"
                --train_fp "$train_fp"
                --train_labels "$train_labels"
                --val_fp "$val_fp"
                --val_labels "$val_labels"
                --test_fp "$eval_fp"
                --test_labels "$eval_labels"
                --output_dir "$run_output_dir"
                --hidden_dims "$hidden_dims"
                --dropout "$dropout"
                --batch_size "$batch_size"
                --epochs "$epochs"
                --optimizer "$optimizer"
                --lr "$learning_rate"
                --num_workers "$train_workers"
                --early_stop_patience "$early_stop_patience"
                --save_probabilities
                --mode test
                --test_out_prefix "$eval_name"
              )
              "${test_cmd[@]}"
            done
          done
          ;;
        adamw)
          for learning_rate in "${lrs[@]}"; do
            for weight_decay in "${wds[@]}"; do
              run_output_dir="$model_dir/output_soft_labels/hidims${hidden_dims}_dropout${dropout}_${optimizer}_lr${learning_rate}_weight_decay${weight_decay}"
              current_config_index="$config_index"
              config_index=$((config_index + 1))
              if ! should_run_config "$current_config_index"; then
                continue
              fi
              echo "[iterations train] running config ${current_config_index}: hidden=${hidden_dims} dropout=${dropout} optimizer=${optimizer} lr=${learning_rate} weight_decay=${weight_decay}" >&2
              train_cmd=(
                "$python_bin" "$train_helper"
                --train_fp "$train_fp"
                --train_labels "$train_labels"
                --val_fp "$val_fp"
                --val_labels "$val_labels"
                --test_fp "$test_fp"
                --test_labels "$test_labels"
                --output_dir "$run_output_dir"
                --hidden_dims "$hidden_dims"
                --dropout "$dropout"
                --batch_size "$batch_size"
                --epochs "$epochs"
                --optimizer "$optimizer"
                --lr "$learning_rate"
                --weight_decay "$weight_decay"
                --num_workers "$train_workers"
                --early_stop_patience "$early_stop_patience"
                --save_probabilities
                --mode train
              )

              if [[ "$iteration" -gt 1 ]]; then
                previous_ckpt="$project_dir/iteration_$((iteration - 1))/model/output_soft_labels/hidims${hidden_dims}_dropout${dropout}_${optimizer}_lr${learning_rate}_weight_decay${weight_decay}/ckpts/best_model.pt"
                if [[ -f "$previous_ckpt" ]]; then
                  train_cmd+=(--finetune_from "$previous_ckpt")
                fi
              fi

              "${train_cmd[@]}"

              for eval_name in training_split tuning_split test_split; do
                case "$eval_name" in
                  training_split)
                    eval_fp="$train_fp"
                    eval_labels="$train_labels"
                    ;;
                  tuning_split)
                    eval_fp="$val_fp"
                    eval_labels="$val_labels"
                    ;;
                  test_split)
                    eval_fp="$test_fp"
                    eval_labels="$test_labels"
                    ;;
                esac

                test_cmd=(
                  "$python_bin" "$train_helper"
                  --train_fp "$train_fp"
                  --train_labels "$train_labels"
                  --val_fp "$val_fp"
                  --val_labels "$val_labels"
                  --test_fp "$eval_fp"
                  --test_labels "$eval_labels"
                  --output_dir "$run_output_dir"
                  --hidden_dims "$hidden_dims"
                  --dropout "$dropout"
                  --batch_size "$batch_size"
                  --epochs "$epochs"
                  --optimizer "$optimizer"
                  --lr "$learning_rate"
                  --weight_decay "$weight_decay"
                  --num_workers "$train_workers"
                  --early_stop_patience "$early_stop_patience"
                  --save_probabilities
                  --mode test
                  --test_out_prefix "$eval_name"
                )
                "${test_cmd[@]}"
              done
            done
          done
          ;;
        sgd)
          for learning_rate in "${lrs[@]}"; do
            for weight_decay in "${wds[@]}"; do
              for momentum in "${momentums[@]}"; do
                run_output_dir="$model_dir/output_soft_labels/hidims${hidden_dims}_dropout${dropout}_${optimizer}_lr${learning_rate}_weight_decay${weight_decay}_momentum${momentum}"
                current_config_index="$config_index"
                config_index=$((config_index + 1))
                if ! should_run_config "$current_config_index"; then
                  continue
                fi
                echo "[iterations train] running config ${current_config_index}: hidden=${hidden_dims} dropout=${dropout} optimizer=${optimizer} lr=${learning_rate} weight_decay=${weight_decay} momentum=${momentum}" >&2
                train_cmd=(
                  "$python_bin" "$train_helper"
                  --train_fp "$train_fp"
                  --train_labels "$train_labels"
                  --val_fp "$val_fp"
                  --val_labels "$val_labels"
                  --test_fp "$test_fp"
                  --test_labels "$test_labels"
                  --output_dir "$run_output_dir"
                  --hidden_dims "$hidden_dims"
                  --dropout "$dropout"
                  --batch_size "$batch_size"
                  --epochs "$epochs"
                  --optimizer "$optimizer"
                  --lr "$learning_rate"
                  --weight_decay "$weight_decay"
                  --momentum "$momentum"
                  --num_workers "$train_workers"
                  --early_stop_patience "$early_stop_patience"
                  --save_probabilities
                  --mode train
                )

                if [[ "$iteration" -gt 1 ]]; then
                  previous_ckpt="$project_dir/iteration_$((iteration - 1))/model/output_soft_labels/hidims${hidden_dims}_dropout${dropout}_${optimizer}_lr${learning_rate}_weight_decay${weight_decay}_momentum${momentum}/ckpts/best_model.pt"
                  if [[ -f "$previous_ckpt" ]]; then
                    train_cmd+=(--finetune_from "$previous_ckpt")
                  fi
                fi

                "${train_cmd[@]}"

                for eval_name in training_split tuning_split test_split; do
                  case "$eval_name" in
                    training_split)
                      eval_fp="$train_fp"
                      eval_labels="$train_labels"
                      ;;
                    tuning_split)
                      eval_fp="$val_fp"
                      eval_labels="$val_labels"
                      ;;
                    test_split)
                      eval_fp="$test_fp"
                      eval_labels="$test_labels"
                      ;;
                  esac

                  test_cmd=(
                    "$python_bin" "$train_helper"
                    --train_fp "$train_fp"
                    --train_labels "$train_labels"
                    --val_fp "$val_fp"
                    --val_labels "$val_labels"
                    --test_fp "$eval_fp"
                    --test_labels "$eval_labels"
                    --output_dir "$run_output_dir"
                    --hidden_dims "$hidden_dims"
                    --dropout "$dropout"
                    --batch_size "$batch_size"
                    --epochs "$epochs"
                    --optimizer "$optimizer"
                    --lr "$learning_rate"
                    --weight_decay "$weight_decay"
                    --momentum "$momentum"
                    --num_workers "$train_workers"
                    --early_stop_patience "$early_stop_patience"
                    --save_probabilities
                    --mode test
                    --test_out_prefix "$eval_name"
                  )
                  "${test_cmd[@]}"
                done
              done
            done
          done
          ;;
        *)
          echo "Error: unknown optimizer $optimizer" >&2
          exit 1
          ;;
      esac
    done
  done
done

echo "Training finished for iteration: $iteration"
echo "Model directory:                $model_dir"
