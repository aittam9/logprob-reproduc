#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/scripts"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_get_probs.sh [test] [all_models|model <name>] [all_data|data <id>] [remote]

Pass "remote" to run model traces on NDIF instead of locally (requires the
NDIF_API_KEY environment variable; see get_probs.py --help and the README).

Examples:
  bash scripts/run_get_probs.sh all_models all_data
  bash scripts/run_get_probs.sh test all_models data 3
  bash scripts/run_get_probs.sh model Llama-3.2-1B all_data
  bash scripts/run_get_probs.sh all_models data 3
  bash scripts/run_get_probs.sh model Llama-3.2-1B all_data remote
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
  local -a cmd=(python get_probs.py --model "$model_name" --data "$data_name")

  if [[ "$RUN_TEST" -eq 1 ]]; then
    cmd+=(--test)
  fi

  if [[ "$RUN_REMOTE" -eq 1 ]]; then
    cmd+=(--remote)
  fi

  if [[ "$RUN_TEST" -eq 1 ]]; then
    echo "Running model=${model_name} data=${data_name} test"
  else
    echo "Running model=${model_name} data=${data_name}"
  fi

  if "${cmd[@]}"; then
    return 0
  fi

  echo "Warning: get_probs.py failed for model=${model_name} data=${data_name}; continuing." >&2
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
RUN_REMOTE=0
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
    remote|--remote)
      RUN_REMOTE=1
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