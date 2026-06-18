#!/usr/bin/env bash

PAPER_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_REPO_ROOT="$(cd "${PAPER_COMMON_DIR}/../.." && pwd)"

PAPER_PROTOCOL_ID="ogsod_hbb_mosaic100_clean_a1b_probea_20260618"
PAPER_PROTOCOL="mosaic100"
PAPER_DATASET="OGSOD-1.0"
PAPER_TASK="hbb"
PAPER_IMGSZ="256"
PAPER_EPOCHS="800"
PAPER_A1_EPOCHS="10"
PAPER_B_EPOCHS="800"
PAPER_MOSAIC="1.0"
PAPER_CLOSE_MOSAIC="700"
PAPER_A1_MOSAIC="1.0"
PAPER_A1_CLOSE_MOSAIC="0"
PAPER_B_MOSAIC="1.0"
PAPER_B_CLOSE_MOSAIC="700"
PAPER_MIXUP="0.0"
PAPER_CUTMIX="0.0"
PAPER_DEGREES="0.0"
PAPER_PERSPECTIVE="0.0"
PAPER_TRANSLATE="0.1"
PAPER_SCALE="0.5"
PAPER_FLIPLR="0.5"
PAPER_FLIPUD="0.0"
PAPER_HSV_H="0.0"
PAPER_HSV_S="0.0"
PAPER_HSV_V="0.0"
PAPER_ERASING="0.0"
PAPER_OPTIMIZER="auto"
PAPER_LR0="0.01"
PAPER_LRF="0.01"
PAPER_WARMUP_EPOCHS="3.0"
PAPER_WARMUP_BIAS_LR="0.1"
PAPER_WORKERS="${PAPER_WORKERS:-8}"
PAPER_SAVE_PERIOD="${PAPER_SAVE_PERIOD:-100}"
PAPER_RUN_ROOT="runs_public/paper/ogsod_hbb_mosaic100"
PAPER_LOG_ROOT="logs/paper/ogsod_hbb_mosaic100"
PAPER_SAR_DATA_CFG="configs/paper/datasets/ogsod_hbb_sar.yaml"
PAPER_RGB_DATA_CFG="configs/paper/datasets/ogsod_hbb_rgb.yaml"

paper_die() {
  echo "$*" >&2
  exit 1
}

paper_require_size() {
  [[ "${1:-}" =~ ^(n|s|m|l|x)$ ]] || paper_die "Expected model size n|s|m|l|x, got: ${1:-<empty>}"
}

paper_require_seed() {
  case "${1:-}" in
    0|42|123) ;;
    *) [[ "${ALLOW_NONPAPER_SEED:-0}" == "1" ]] || paper_die "Paper seeds are restricted to 0, 42, 123; got: ${1:-<empty>}" ;;
  esac
}

paper_batch_for_size() {
  case "$1" in
    n|s) printf '64\n' ;;
    m|l) printf '32\n' ;;
    x) printf '16\n' ;;
    *) paper_die "Unknown model size: $1" ;;
  esac
}

paper_git_commit() {
  git -C "$PAPER_REPO_ROOT" rev-parse HEAD 2>/dev/null || printf 'unknown\n'
}

paper_git_dirty() {
  if git -C "$PAPER_REPO_ROOT" diff --quiet --ignore-submodules -- && git -C "$PAPER_REPO_ROOT" diff --cached --quiet --ignore-submodules --; then
    printf 'false\n'
  else
    printf 'true\n'
  fi
}

paper_check_strict_git() {
  local dirty
  dirty="$(paper_git_dirty)"
  if [[ "${PAPER_STRICT_GIT:-0}" == "1" && "$dirty" != "false" ]]; then
    paper_die "PAPER_STRICT_GIT=1 but tracked git diff is dirty."
  fi
}

paper_require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    paper_die "Missing ${label}: ${path}"
  fi
}

paper_latest_weight() {
  local pattern="$1"
  local -a candidates=()
  local candidate
  shopt -s nullglob
  for candidate in $pattern; do
    candidates+=("$candidate")
  done
  shopt -u nullglob
  if (( ${#candidates[@]} == 0 )); then
    return 1
  fi
  ls -t "${candidates[@]}" 2>/dev/null | head -n 1
}

paper_baseline_pattern() {
  local modality="$1"
  local size="$2"
  local seed="$3"
  local batch
  batch="$(paper_batch_for_size "$size")"
  printf '%s/baselines/%s/yolo11%s/seed%s/paper_ogsod_hbb_mosaic100_%s_yolo11%s_e800_b%s_s%s*/weights/best.pt\n' \
    "$PAPER_RUN_ROOT" "$modality" "$size" "$seed" "$modality" "$size" "$batch" "$seed"
}

paper_find_baseline() {
  local modality="$1"
  local size="$2"
  local seed="$3"
  paper_latest_weight "$(paper_baseline_pattern "$modality" "$size" "$seed")"
}

paper_print_command() {
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'
}

paper_write_meta_common() {
  local path="$1"
  local method="$2"
  local method_label="$3"
  local size="$4"
  local seed="$5"
  local gpu_id="$6"
  local batch="$7"
  local run_tag="$8"
  local project_dir="$9"
  local run_dir="${10}"
  local command_text="${11}"

  mkdir -p "$(dirname "$path")"
  {
    printf 'paper_run=1\n'
    printf 'paper_protocol_id=%q\n' "$PAPER_PROTOCOL_ID"
    printf 'paper_protocol=%q\n' "$PAPER_PROTOCOL"
    printf 'protocol_id=%q\n' "$PAPER_PROTOCOL_ID"
    printf 'protocol=%q\n' "$PAPER_PROTOCOL"
    printf 'method=%q\n' "$method"
    printf 'method_label=%q\n' "$method_label"
    printf 'dataset=%q\n' "$PAPER_DATASET"
    printf 'task=%q\n' "$PAPER_TASK"
    printf 'model_size=%q\n' "$size"
    printf 'seed=%q\n' "$seed"
    printf 'gpu_id=%q\n' "$gpu_id"
    printf 'batch=%q\n' "$batch"
    printf 'imgsz=%q\n' "$PAPER_IMGSZ"
    printf 'epochs=%q\n' "$PAPER_EPOCHS"
    printf 'mosaic=%q\n' "$PAPER_MOSAIC"
    printf 'close_mosaic=%q\n' "$PAPER_CLOSE_MOSAIC"
    printf 'data_cfg=%q\n' "$PAPER_SAR_DATA_CFG"
    printf 'teacher_data_cfg=%q\n' "$PAPER_RGB_DATA_CFG"
    printf 'run_tag=%q\n' "$run_tag"
    printf 'project_dir=%q\n' "$project_dir"
    printf 'run_dir=%q\n' "$run_dir"
    printf 'git_commit=%q\n' "$(paper_git_commit)"
    printf 'git_dirty=%q\n' "$(paper_git_dirty)"
    printf 'launcher=%q\n' "${BASH_SOURCE[1]}"
    printf 'created_at=%q\n' "$(date '+%F %T')"
    printf 'command=%q\n' "$command_text"
  } > "$path"
}

paper_command_text() {
  printf '%q ' "$@"
}
