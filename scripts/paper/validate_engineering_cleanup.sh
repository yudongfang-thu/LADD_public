#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 -m py_compile \
  baseline/code/train_ogsod_baseline.py \
  ladd/code/train_ladd_hbb.py \
  ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py \
  ladd/code/src/teacher_student_decomposition_kd_hbb/trainer.py \
  ladd/code/src/teacher_student_decomposition_kd_hbb/model.py \
  ladd/code_versions/current_hbb/tools/train_ladd_hbb.py \
  tools/paper_collect_results.py \
  tools/paper_validate_main_table.py \
  tools/check_current_hbb_sync.py

bash -n scripts/paper/paper_common.sh
bash -n scripts/paper/run_paper_baseline.sh
bash -n scripts/paper/run_paper_ladd_probea.sh
bash -n scripts/paper/run_paper_comparison_kd.sh
bash -n scripts/paper/run_paper_hallucidet.sh
bash -n scripts/paper/run_paper_cclkd_online.sh
bash -n scripts/paper/dry_run_all_paper_entrypoints.sh
bash -n scripts/paper/prepare_paper_dataset_yamls.sh
bash -n baseline/scripts/run_formal_baseline.sh
bash -n ladd/scripts/launch_ladd_clean_a1b_job.sh
bash -n comparison/code/launch_formal_transfer_kd_job.sh
bash -n comparison/code/launch_formal_from_yolo_kd_job.sh
bash -n comparison/code/launch_formal_online_cclkd_job.sh

python3 tools/check_current_hbb_sync.py

dry_log="$(mktemp)"
valid_csv="$(mktemp)"
invalid_csv="$(mktemp)"
ladd_negative_log="$(mktemp)"
comparison_negative_log="$(mktemp)"
validator_negative_log="$(mktemp)"
trap 'rm -f "$dry_log" "$valid_csv" "$invalid_csv" "$ladd_negative_log" "$comparison_negative_log" "$validator_negative_log"' EXIT
DRY_RUN=1 bash scripts/paper/dry_run_all_paper_entrypoints.sh | tee "$dry_log"
DRY_RUN=1 PROTOCOL=mosaic100 bash comparison/code/launch_formal_transfer_kd_job.sh ld n 0 0 >/dev/null
DRY_RUN=1 PROTOCOL=mosaic100 bash comparison/code/launch_formal_from_yolo_kd_job.sh ld n 0 0 >/dev/null
DRY_RUN=1 PROTOCOL=mosaic100 bash comparison/code/launch_formal_online_cclkd_job.sh n 0 0 >/dev/null

grep -q "mosaic=1.0" "$dry_log"
grep -q "close_mosaic=700" "$dry_log"
grep -q "LADD_A1B_MODE=dynamic_probe" "$dry_log"
grep -q "KD_CALIBRATION_MODE=affine" "$dry_log"
grep -q "PAPER_PROTOCOL_ID=ogsod_hbb_mosaic100_clean_a1b_probea_20260618" "$dry_log"

if grep -E "formal_nomosaic|no-mosaic|nomosaic" "$dry_log"; then
  echo "Dry-run output references formal/no-mosaic paths." >&2
  exit 1
fi

if grep -E " a2 |phase a2|<a2|_a2_" "$dry_log"; then
  echo "Dry-run output appears to reference A2." >&2
  exit 1
fi

set +e
PROTOCOL=nomosaic DRY_RUN=1 bash scripts/paper/run_paper_baseline.sh sar n 0 0 >"$comparison_negative_log" 2>&1
status=$?
if [[ "$status" -eq 0 ]]; then
  echo "Expected paper baseline nomosaic failure, but command succeeded." >&2
  exit 1
fi

PAPER_RUN=1 LADD_A1B_MODE=static DRY_RUN=1 bash ladd/scripts/launch_ladd_clean_a1b_job.sh n 0 0 >"$ladd_negative_log" 2>&1
status=$?
if [[ "$status" -eq 0 ]]; then
  echo "Expected LADD static PAPER_RUN failure, but command succeeded." >&2
  exit 1
fi

PAPER_RUN=1 PROTOCOL=nomosaic DRY_RUN=1 bash comparison/code/launch_formal_transfer_kd_job.sh ld n 0 0 >"$comparison_negative_log" 2>&1
status=$?
if [[ "$status" -eq 0 ]]; then
  echo "Expected comparison nomosaic PAPER_RUN failure, but command succeeded." >&2
  exit 1
fi

PAPER_RUN=1 PROTOCOL=nomosaic DRY_RUN=1 bash comparison/code/launch_formal_online_cclkd_job.sh n 0 0 >"$comparison_negative_log" 2>&1
status=$?
if [[ "$status" -eq 0 ]]; then
  echo "Expected CCLKD online nomosaic PAPER_RUN failure, but command succeeded." >&2
  exit 1
fi
set -e

cat > "$valid_csv" <<'EOF'
dataset,task,protocol_id,method,method_display,model_size,seed,init_type,student_modality,teacher_modality,inference_modality,imgsz,epochs,batch,mosaic,close_mosaic,phase_chain,ladd_mode,run_tag,project_dir,results_csv,args_yaml,manifest,git_commit,best_ap50_95,best_ap50,final_ap50_95,final_ap50,best_epoch,status,claim_usable,notes
OGSOD-1.0,hbb,ogsod_hbb_mosaic100_clean_a1b_probea_20260618,ladd_probea,LADD Probe-A,n,0,sar_baseline,SAR,RGB,SAR,256,800,64,1.0,700,A1->B,dynamic_probe,paper_clean_a1b_dynprobe_mosaic100_yolo11n_s0,runs_public/paper/demo,results.csv,args.yaml,paper_run_meta.env,deadbeef,0.5,0.8,0.49,0.79,100,verified,yes,
EOF
python3 tools/paper_validate_main_table.py "$valid_csv"

cat > "$invalid_csv" <<'EOF'
dataset,task,protocol_id,method,method_display,model_size,seed,init_type,student_modality,teacher_modality,inference_modality,imgsz,epochs,batch,mosaic,close_mosaic,phase_chain,ladd_mode,run_tag,project_dir,results_csv,args_yaml,manifest,git_commit,best_ap50_95,best_ap50,final_ap50_95,final_ap50,best_epoch,status,claim_usable,notes
OGSOD-1.0,hbb,wrong,ladd_probea,LADD Probe-A,n,0,sar_baseline,SAR,RGB,SAR,256,800,64,0.0,0,A1-A2-B,static,old_a1a2b_nomosaic,runs_public/old,results.csv,args.yaml,manifest.txt,deadbeef,0.5,0.8,0.49,0.79,100,verified,yes,A2 BN-freeze no-mosaic
EOF
set +e
python3 tools/paper_validate_main_table.py "$invalid_csv" >"$validator_negative_log" 2>&1
status=$?
set -e
if [[ "$status" -eq 0 ]]; then
  echo "Expected paper validator negative fixture to fail, but it passed." >&2
  exit 1
fi

git diff --check

echo "OK: paper engineering cleanup validation passed."
