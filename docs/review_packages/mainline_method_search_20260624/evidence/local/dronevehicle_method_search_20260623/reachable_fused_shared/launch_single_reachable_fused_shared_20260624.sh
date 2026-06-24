#!/usr/bin/env bash
set -euo pipefail

# Launch exactly one reachable_fused_shared variant.
# Usage:
#   GPU_ID=0 VARIANT=c0_nofusion_splitrec bash launch_single_reachable_fused_shared_20260624.sh
#   GPU_ID=0 VARIANT=sum_mlp_cap2 bash launch_single_reachable_fused_shared_20260624.sh
#   GPU_ID=0 VARIANT=concat_mlp_cap2 bash launch_single_reachable_fused_shared_20260624.sh
#   GPU_ID=0 VARIANT=sum_mlp_cap2_ema bash launch_single_reachable_fused_shared_20260624.sh
#   GPU_ID=0 VARIANT=c0_yoloinit_nofusion bash launch_single_reachable_fused_shared_20260624.sh
#   GPU_ID=1 VARIANT=sum_mlp_cap2_ema_yoloinit bash launch_single_reachable_fused_shared_20260624.sh
#   GPU_ID=0 VARIANT=c0_yoloinit_std bash launch_single_reachable_fused_shared_20260624.sh
#   GPU_ID=1 VARIANT=sum_mlp_cap2_ema_yoloinit_std bash launch_single_reachable_fused_shared_20260624.sh
#
# This intentionally replaces the old continuous queue for night search. It
# avoids launching c0/sum/concat back-to-back on a still-busy GPU.

cd /root/shared-nvme/LADD_public

GPU_ID="${GPU_ID:?set GPU_ID}"
VARIANT="${VARIANT:?set VARIANT=c0_nofusion_splitrec|sum_mlp_cap2|concat_mlp_cap2|sum_mlp_cap2_ema|c0_yoloinit_nofusion|sum_mlp_cap2_ema_yoloinit|c0_yoloinit_std|sum_mlp_cap2_ema_yoloinit_std}"
PRECHECK_MAX_USED_MB="${PRECHECK_MAX_USED_MB:-12000}"
TEACHER_TARGET_MODE=static
TEACHER_EMA_MOMENTUM="${TEACHER_EMA_MOMENTUM:-0.99}"
SCHEDULE_TAG="lowlr1e3_nowarmup"
LR0_VALUE=0.001
LRF_VALUE=0.1
OPTIMIZER_VALUE=SGD
WARMUP_EPOCHS_VALUE=0.0
WARMUP_BIAS_LR_VALUE=0.0
WARMUP_MOMENTUM_VALUE=0.937

STUDENT="runs_public/cross_dataset/cclkd_yolo11n/dronevehicle_sub2k_seed0/baselines/student_rgb/dronevehicle_sub2k_student_rgb_yolo11n_cclkdproto_e200_b64_img512_mosaic0p0_close0_mixup0p1_s0_20260623_221620/weights/best.pt"
TEACHER="runs_public/cross_dataset/cclkd_yolo11n/dronevehicle_sub2k_seed0/baselines/teacher_ir/dronevehicle_sub2k_teacher_ir_yolo11n_cclkdproto_e200_b64_img512_mosaic0p0_close0_mixup0p1_s0_gpu0_20260623_221936/weights/best.pt"
DECOMP="runs_public/dronevehicle_method_search/sub2k_seed0_fullval/oldsplit_90_hbb/ir_to_rgb/oldsplit90_hbb_cclkdproto_ir2rgb_from_rgbbase_P1_20260623_2313_a2/weights/last.pt"
YOLO_INIT="${YOLO_INIT:-yolo11n.pt}"
RGB_YAML="comparison/cmdistill/native_reproduction/data/processed/DroneVehicle_cclkd_hbb_sub2k_seed0/configs/dronevehicle_rgb_hbb.yaml"
IR_YAML="comparison/cmdistill/native_reproduction/data/processed/DroneVehicle_cclkd_hbb_sub2k_seed0/configs/dronevehicle_ir_hbb.yaml"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 2
  fi
}

require_file "$STUDENT" "RGB baseline checkpoint"
require_file "$TEACHER" "IR teacher checkpoint"
require_file "$DECOMP" "A2 decomposition checkpoint"
require_file "$YOLO_INIT" "YOLO init checkpoint"
require_file "$RGB_YAML" "RGB data yaml"
require_file "$IR_YAML" "IR data yaml"

used_mb="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v gpu="$GPU_ID" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1 == gpu) print $2}')"
if [[ -z "$used_mb" ]]; then
  echo "Could not read GPU ${GPU_ID} memory" >&2
  exit 3
fi
if (( used_mb > PRECHECK_MAX_USED_MB )); then
  echo "GPU ${GPU_ID} already uses ${used_mb} MB > PRECHECK_MAX_USED_MB=${PRECHECK_MAX_USED_MB}; refusing to launch" >&2
  exit 4
fi

DETECTOR_SOURCE="$STUDENT"
B_RESET_STUDENT_FROM_SCRATCH=0
B_LOAD_STUDENT_SPLIT=1
B_LOAD_STUDENT_REACHABILITY=0

case "$VARIANT" in
  c0_nofusion_splitrec)
    FUSED_SHARED_MODE=none
    FUSED_SHARED_ALIGN_WEIGHT=0.0
    FUSED_SHARED_REACH_WEIGHT=0.0
    FUSED_SHARED_KD_WEIGHT=0.0
    FUSED_SHARED_TASK_WEIGHT=0.0
    ALPHA_KD=0.0
    LAMBDA_REC=0.0
    LAMBDA_TASKL=0.0
    PROJECT_SUFFIX="c0_nofusion_splitrec"
    INIT_TYPE="reachable_fused_shared_c0_nofusion_splitrec_dronevehicle"
    PROTOCOL_ID="dronevehicle_sub2k_lowlr_nowarmup_reachable_fused_shared_c0_nofusion_splitrec"
    ;;
  c0_yoloinit_nofusion)
    FUSED_SHARED_MODE=none
    FUSED_SHARED_ALIGN_WEIGHT=0.0
    FUSED_SHARED_REACH_WEIGHT=0.0
    FUSED_SHARED_KD_WEIGHT=0.0
    FUSED_SHARED_TASK_WEIGHT=0.0
    ALPHA_KD=0.0
    LAMBDA_REC=0.0
    LAMBDA_TASKL=0.0
    PROJECT_SUFFIX="c0_yoloinit_nofusion"
    INIT_TYPE="reachable_fused_shared_c0_yoloinit_nofusion_dronevehicle"
    PROTOCOL_ID="dronevehicle_sub2k_lowlr_nowarmup_reachable_fused_shared_c0_yoloinit_nofusion"
    DETECTOR_SOURCE="$YOLO_INIT"
    B_RESET_STUDENT_FROM_SCRATCH=1
    B_LOAD_STUDENT_SPLIT=0
    ;;
  c0_yoloinit_std)
    FUSED_SHARED_MODE=none
    FUSED_SHARED_ALIGN_WEIGHT=0.0
    FUSED_SHARED_REACH_WEIGHT=0.0
    FUSED_SHARED_KD_WEIGHT=0.0
    FUSED_SHARED_TASK_WEIGHT=0.0
    ALPHA_KD=0.0
    LAMBDA_REC=0.0
    LAMBDA_TASKL=0.0
    PROJECT_SUFFIX="c0_yoloinit_std"
    INIT_TYPE="reachable_fused_shared_c0_yoloinit_std_dronevehicle"
    PROTOCOL_ID="dronevehicle_sub2k_stdlr_warmup_reachable_fused_shared_c0_yoloinit_std"
    SCHEDULE_TAG="stdlr0p01_warmup3"
    LR0_VALUE=0.01
    LRF_VALUE=0.01
    OPTIMIZER_VALUE=auto
    WARMUP_EPOCHS_VALUE=3.0
    WARMUP_BIAS_LR_VALUE=0.1
    WARMUP_MOMENTUM_VALUE=0.8
    DETECTOR_SOURCE="$YOLO_INIT"
    B_RESET_STUDENT_FROM_SCRATCH=1
    B_LOAD_STUDENT_SPLIT=0
    ;;
  sum_mlp_cap2)
    FUSED_SHARED_MODE=sum
    FUSED_SHARED_ALIGN_WEIGHT=0.25
    FUSED_SHARED_REACH_WEIGHT=0.50
    FUSED_SHARED_KD_WEIGHT=1.0
    FUSED_SHARED_TASK_WEIGHT=0.25
    ALPHA_KD=1.0
    LAMBDA_REC=0.0
    LAMBDA_TASKL=0.0
    PROJECT_SUFFIX="sum_mlp_cap2"
    INIT_TYPE="reachable_fused_shared_sum_mlp_cap2_dronevehicle"
    PROTOCOL_ID="dronevehicle_sub2k_lowlr_nowarmup_reachable_fused_shared_sum_mlp_cap2"
    ;;
  sum_mlp_cap2_ema)
    FUSED_SHARED_MODE=sum
    FUSED_SHARED_ALIGN_WEIGHT=0.25
    FUSED_SHARED_REACH_WEIGHT=0.50
    FUSED_SHARED_KD_WEIGHT=1.0
    FUSED_SHARED_TASK_WEIGHT=0.25
    ALPHA_KD=1.0
    LAMBDA_REC=0.0
    LAMBDA_TASKL=0.0
    PROJECT_SUFFIX="sum_mlp_cap2_ema"
    INIT_TYPE="reachable_fused_shared_sum_mlp_cap2_ema_dronevehicle"
    PROTOCOL_ID="dronevehicle_sub2k_lowlr_nowarmup_reachable_fused_shared_sum_mlp_cap2_ema"
    TEACHER_TARGET_MODE=ema
    ;;
  sum_mlp_cap2_ema_yoloinit)
    FUSED_SHARED_MODE=sum
    FUSED_SHARED_ALIGN_WEIGHT=0.25
    FUSED_SHARED_REACH_WEIGHT=0.50
    FUSED_SHARED_KD_WEIGHT=1.0
    FUSED_SHARED_TASK_WEIGHT=0.25
    ALPHA_KD=1.0
    LAMBDA_REC=0.0
    LAMBDA_TASKL=0.0
    PROJECT_SUFFIX="sum_mlp_cap2_ema_yoloinit"
    INIT_TYPE="reachable_fused_shared_sum_mlp_cap2_ema_yoloinit_dronevehicle"
    PROTOCOL_ID="dronevehicle_sub2k_lowlr_nowarmup_reachable_fused_shared_sum_mlp_cap2_ema_yoloinit"
    TEACHER_TARGET_MODE=ema
    DETECTOR_SOURCE="$YOLO_INIT"
    B_RESET_STUDENT_FROM_SCRATCH=1
    B_LOAD_STUDENT_SPLIT=0
    ;;
  sum_mlp_cap2_ema_yoloinit_std)
    FUSED_SHARED_MODE=sum
    FUSED_SHARED_ALIGN_WEIGHT=0.25
    FUSED_SHARED_REACH_WEIGHT=0.50
    FUSED_SHARED_KD_WEIGHT=1.0
    FUSED_SHARED_TASK_WEIGHT=0.25
    ALPHA_KD=1.0
    LAMBDA_REC=0.0
    LAMBDA_TASKL=0.0
    PROJECT_SUFFIX="sum_mlp_cap2_ema_yoloinit_std"
    INIT_TYPE="reachable_fused_shared_sum_mlp_cap2_ema_yoloinit_std_dronevehicle"
    PROTOCOL_ID="dronevehicle_sub2k_stdlr_warmup_reachable_fused_shared_sum_mlp_cap2_ema_yoloinit_std"
    SCHEDULE_TAG="stdlr0p01_warmup3"
    LR0_VALUE=0.01
    LRF_VALUE=0.01
    OPTIMIZER_VALUE=auto
    WARMUP_EPOCHS_VALUE=3.0
    WARMUP_BIAS_LR_VALUE=0.1
    WARMUP_MOMENTUM_VALUE=0.8
    TEACHER_TARGET_MODE=ema
    DETECTOR_SOURCE="$YOLO_INIT"
    B_RESET_STUDENT_FROM_SCRATCH=1
    B_LOAD_STUDENT_SPLIT=0
    ;;
  concat_mlp_cap2)
    FUSED_SHARED_MODE=concat
    FUSED_SHARED_ALIGN_WEIGHT=0.25
    FUSED_SHARED_REACH_WEIGHT=0.50
    FUSED_SHARED_KD_WEIGHT=1.0
    FUSED_SHARED_TASK_WEIGHT=0.25
    ALPHA_KD=1.0
    LAMBDA_REC=0.0
    LAMBDA_TASKL=0.0
    PROJECT_SUFFIX="concat_mlp_cap2"
    INIT_TYPE="reachable_fused_shared_concat_mlp_cap2_dronevehicle"
    PROTOCOL_ID="dronevehicle_sub2k_lowlr_nowarmup_reachable_fused_shared_concat_mlp_cap2"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}" >&2
    exit 5
    ;;
esac

stamp="$(date +%Y%m%d_%H%M%S)"
run_tag="reachable_fused_${VARIANT}_${SCHEDULE_TAG}_ir2rgb_yolo11n_e200_b64_img512_s0_${stamp}"
project_dir="runs_public/dronevehicle_method_search/sub2k_seed0_fullval/reachable_fused_shared/${PROJECT_SUFFIX}"
log_root="logs/dronevehicle_method_search/sub2k_seed0_fullval/reachable_fused_shared/${PROJECT_SUFFIX}"
phase_log_dir="${log_root}/${run_tag}_gpu${GPU_ID}"
outer_log="${log_root}/${run_tag}_gpu${GPU_ID}.outer.log"
pid_path="${log_root}/${run_tag}_gpu${GPU_ID}.pid"
run_name="${run_tag}_b"

mkdir -p "$project_dir" "$phase_log_dir" "$log_root"

echo "[$(date '+%F %T')] launching ${VARIANT} gpu=${GPU_ID} used_mb=${used_mb} run=${run_tag}"

nohup env \
  PYTHON_BIN=/root/shared-nvme/venvs/ladd312/bin/python \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  MODEL="$STUDENT" SAR_BASELINE="$STUDENT" RGB_TEACHER="$TEACHER" \
  INIT_TYPE="$INIT_TYPE" \
  PAPER_RUN=0 PAPER_PROTOCOL_ID="$PROTOCOL_ID" \
  DISABLE_OGSOD_PROTOCOL_GUARD=1 \
  DATA_CFG="$RGB_YAML" TEACHER_DATA_CFG="$IR_YAML" \
  GPU_ID="$GPU_ID" SEED=0 BATCH_SIZE=64 STRICT_BATCH_SIZE=1 WORKERS=8 IMGSZ=512 \
  EPOCHS=200 PATIENCE=200 PHASE_MIN_EPOCHS=200 \
  PROJECT_DIR="$project_dir" LOG_DIR="$phase_log_dir" RUN_NAME="$run_name" SERVER_TAG=ladd4090zw1 \
  B_DETECTOR_SOURCE="$DETECTOR_SOURCE" B_DECOMP_SOURCE="$DECOMP" B_RESET_STUDENT_FROM_SCRATCH="$B_RESET_STUDENT_FROM_SCRATCH" B_LOAD_STUDENT_SPLIT="$B_LOAD_STUDENT_SPLIT" B_LOAD_STUDENT_REACHABILITY="$B_LOAD_STUDENT_REACHABILITY" \
  COMPARISON_KD_PROFILE=none PROFILE_KD_WEIGHT=0.0 PROFILE_KD_REPLACE_BASE=1 \
  STUDENT_BRANCH_MODE=split STUDENT_DETECT_MODE=raw TEACHER_FEATURE_MODE=decomposed UNLEARNABLE_HIDDEN_RATIO=1.0 \
  USE_MASK=0 USE_FG_MASK_FOR_REACH=1 USE_FG_MASK_FOR_REC=0 TASK_LOSS_FG_ONLY=1 \
  ALPHA_KD="$ALPHA_KD" LAMBDA_REACH=0.0 LAMBDA_REC="$LAMBDA_REC" LAMBDA_TASKL="$LAMBDA_TASKL" ALPHA_S_REC=0.05 KD_CALIBRATION_MODE=affine \
  FUSED_SHARED_MODE="$FUSED_SHARED_MODE" FUSED_SHARED_ALIGN_WEIGHT="$FUSED_SHARED_ALIGN_WEIGHT" FUSED_SHARED_REACH_WEIGHT="$FUSED_SHARED_REACH_WEIGHT" FUSED_SHARED_KD_WEIGHT="$FUSED_SHARED_KD_WEIGHT" FUSED_SHARED_TASK_WEIGHT="$FUSED_SHARED_TASK_WEIGHT" \
  TEACHER_TARGET_MODE="$TEACHER_TARGET_MODE" TEACHER_EMA_MOMENTUM="$TEACHER_EMA_MOMENTUM" \
  COS_LR=1 LR0="$LR0_VALUE" LRF="$LRF_VALUE" MOMENTUM=0.937 WEIGHT_DECAY=0.0005 OPTIMIZER="$OPTIMIZER_VALUE" \
  WARMUP_EPOCHS="$WARMUP_EPOCHS_VALUE" WARMUP_BIAS_LR="$WARMUP_BIAS_LR_VALUE" WARMUP_MOMENTUM="$WARMUP_MOMENTUM_VALUE" \
  MOSAIC=0.0 CLOSE_MOSAIC=0 CLOSE_AT_EPOCH= MIXUP=0.1 CUTMIX=0.0 DEGREES=0.0 PERSPECTIVE=0.0 TRANSLATE=0.1 SCALE=0.5 FLIPLR=0.5 FLIPUD=0.0 HSV_H=0.0 HSV_S=0.0 HSV_V=0.0 ERASING=0.0 \
  SAVE_PERIOD=25 EXIST_OK=0 \
  bash ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh hbb b "$run_tag" \
  > "$outer_log" 2>&1 &

echo "$!" > "$pid_path"
echo "[$(date '+%F %T')] pid=$(cat "$pid_path") outer=${outer_log}"
echo "project=${project_dir}"
echo "run_name=${run_name}"
