#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/scripts"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_get_probs_choice.sh [test] [all_models|model <name>] [all_data|data <id>]

Runs scripts/get_probs_choice.py, which scores each pair via a forced
1-or-2 multiple-choice prompt ("Which sentence is more plausible?") instead
of comparing raw sentence logprobs. Works for both base and
instruction-tuned models. Results go to ../results_choice/.

Examples:
  bash scripts/run_get_probs_choice.sh all_models all_data
  bash scripts/run_get_probs_choice.sh test all_models data 3
  bash scripts/run_get_probs_choice.sh model Llama-3.2-1B-it all_data
EOF
}

list_models() {
  python - <<'PY'
from utils.models import BASE_MODELS, INSTRUCT_MODELS

for name in list(BASE_MODELS.keys()) + list(INSTRUCT_MODELS.keys()):
    print(name)
PY
}

list_data() {
  python - <<'PY'
from utils.data_map import DATA_MAP

for key in DATA_MAP.keys():
    print(key)
PY
}

run_case() {
  local model_name="$1"
  local data_name="$2"
  local -a cmd=(python get_probs_choice.py --model "$model_name" --data "$data_name")

  if [[ "$RUN_TEST" -eq 1 ]]; then
    cmd+=(--test)
  fi

  if [[ "$RUN_TEST" -eq 1 ]]; then
    echo "Running model=${model_name} data=${data_name} test"
  else
    echo "Running model=${model_name} data=${data_name}"
  fi

  if "${cmd[@]}"; then
    return 0
  fi

  echo "Warning: get_probs_choice.py failed for model=${model_name} data=${data_name}; continuing." >&2
  return 0
}

contains_item() {
  local needle="$1"
  shift

  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done

  return 1
}

RUN_TEST=0
MODEL_MODE="all"
DATA_MODE="all"
MODEL_NAME=""
DATA_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    test|--test)
      RUN_TEST=1
      shift
      ;;
    all_models|all-models)
      MODEL_MODE="all"
      MODEL_NAME=""
      shift
      ;;
    all_data|all-data)
      DATA_MODE="all"
      DATA_NAME=""
      shift
      ;;
    model)
      if [[ $# -lt 2 ]]; then
        echo "Missing model name after 'model'." >&2
        usage
        exit 1
      fi
      MODEL_MODE="single"
      MODEL_NAME="$2"
      shift 2
      ;;
    data)
      if [[ $# -lt 2 ]]; then
        echo "Missing data id after 'data'." >&2
        usage
        exit 1
      fi
      DATA_MODE="single"
      DATA_NAME="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

cd "$SCRIPT_DIR"

mapfile -t AVAILABLE_MODELS < <(list_models)
mapfile -t AVAILABLE_DATASETS < <(list_data)

if [[ "$MODEL_MODE" == "all" ]]; then
  MODELS=("${AVAILABLE_MODELS[@]}")
else
  if ! contains_item "$MODEL_NAME" "${AVAILABLE_MODELS[@]}"; then
    echo "Unknown model: $MODEL_NAME" >&2
    exit 1
  fi
  MODELS=("$MODEL_NAME")
fi

if [[ "$DATA_MODE" == "all" ]]; then
  DATASETS=("${AVAILABLE_DATASETS[@]}")
else
  if ! contains_item "$DATA_NAME" "${AVAILABLE_DATASETS[@]}"; then
    echo "Unknown dataset: $DATA_NAME" >&2
    exit 1
  fi
  DATASETS=("$DATA_NAME")
fi

for model_name in "${MODELS[@]}"; do
  for data_name in "${DATASETS[@]}"; do
    run_case "$model_name" "$data_name"
  done
done
