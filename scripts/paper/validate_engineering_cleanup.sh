#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

REQUIRED_FILES=(
  docs/paper/PAPER_PROTOCOL_CN.md
  docs/paper/METHOD_NAME_WHITELIST_CN.md
  configs/paper/ogsod_hbb_mosaic100.yaml
  configs/paper/datasets/ogsod_hbb_sar.yaml
  configs/paper/datasets/ogsod_hbb_rgb.yaml
  scripts/paper/paper_common.sh
  scripts/paper/dry_run_all_paper_entrypoints.sh
  scripts/paper/run_paper_baseline.sh
  scripts/paper/run_paper_ladd_probea.sh
  scripts/paper/run_paper_comparison_kd.sh
  scripts/paper/run_paper_cclkd_online.sh
  scripts/paper/run_paper_hallucidet.sh
  scripts/paper/prepare_paper_dataset_yamls.sh
  scripts/paper/validate_engineering_cleanup.sh
  paper_results/README_CN.md
  paper_results/main_table_schema.csv
  paper_results/example_valid.csv
  paper_results/example_invalid.csv
  paper_results/validator_fixtures/valid_probea_name_variants.csv
  paper_results/validator_fixtures/invalid_legacy_variants.csv
  paper_results/validator_fixtures/invalid_wrong_yaml.csv
  paper_results/validator_fixtures/invalid_missing_metrics.csv
  tools/paper_collect_results.py
  tools/paper_validate_main_table.py
)

for path in "${REQUIRED_FILES[@]}"; do
  test -f "$path" || { echo "missing required paper file: $path" >&2; exit 1; }
done

grep -q "LADD Probe-A / LADD-clean A1B" README.md
grep -q "clean_a1b_dynprobe" README.md
grep -q "Paper mainline gate" PACKAGE_AUDIT_CN.md
grep -q "tools/paper_validate_main_table.py" PACKAGE_AUDIT_CN.md

python3 -m py_compile \
  tools/build_experiment_registry.py \
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
transfer_log="$(mktemp)"
from_yolo_log="$(mktemp)"
collector_dir="$(mktemp -d)"
collector_csv="$(mktemp)"
collector_bad_root="$(mktemp -d)"
collector_bad_csv="$(mktemp)"
trap 'rm -f "$dry_log" "$valid_csv" "$invalid_csv" "$ladd_negative_log" "$comparison_negative_log" "$validator_negative_log" "$transfer_log" "$from_yolo_log" "$collector_csv" "$collector_bad_csv"; rm -rf "$collector_dir" "$collector_bad_root"' EXIT
DRY_RUN=1 bash scripts/paper/dry_run_all_paper_entrypoints.sh | tee "$dry_log"
DRY_RUN=1 PROTOCOL=mosaic100 bash comparison/code/launch_formal_transfer_kd_job.sh ld n 0 0 >"$transfer_log"
grep -q "baseline_controls/mosaic_baselines_20260615" "$transfer_log"
grep -q "MOSAIC=1.0" "$transfer_log"
grep -q "CLOSE_MOSAIC=700" "$transfer_log"
grep -q "CLOSE_AT_EPOCH=" "$transfer_log"
grep -q "INIT_TYPE=transferred_kd" "$transfer_log"
if grep -q "formal_nomosaic_20260528" "$transfer_log"; then
  echo "mosaic100 transfer dry-run still references formal_nomosaic_20260528" >&2
  exit 1
fi

DRY_RUN=1 PROTOCOL=mosaic100 bash comparison/code/launch_formal_from_yolo_kd_job.sh ld n 0 0 >"$from_yolo_log"
grep -q "baseline_controls/mosaic_baselines_20260615" "$from_yolo_log"
grep -q "MOSAIC=1.0" "$from_yolo_log"
grep -q "CLOSE_MOSAIC=700" "$from_yolo_log"
grep -q "CLOSE_AT_EPOCH=" "$from_yolo_log"
grep -q "INIT_TYPE=from_yolo_pretrain" "$from_yolo_log"
if grep -q "formal_nomosaic_20260528" "$from_yolo_log"; then
  echo "mosaic100 from-YOLO dry-run still references formal_nomosaic_20260528" >&2
  exit 1
fi
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

PAPER_RUN=1 PROTOCOL=nomosaic DRY_RUN=1 bash baseline/scripts/run_formal_baseline.sh sar n 0 0 >"$comparison_negative_log" 2>&1
status=$?
if [[ "$status" -eq 0 ]]; then
  echo "Expected lower-level baseline nomosaic PAPER_RUN failure, but command succeeded." >&2
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

python3 tools/paper_validate_main_table.py paper_results/example_valid.csv
python3 tools/paper_validate_main_table.py paper_results/validator_fixtures/valid_probea_name_variants.csv
set +e
python3 tools/paper_validate_main_table.py paper_results/example_invalid.csv >"$validator_negative_log" 2>&1
status=$?
set -e
if [[ "$status" -eq 0 ]]; then
  echo "Expected paper_results/example_invalid.csv to fail, but it passed." >&2
  exit 1
fi
set +e
python3 tools/paper_validate_main_table.py paper_results/validator_fixtures/invalid_legacy_variants.csv >"$validator_negative_log" 2>&1
status=$?
set -e
if [[ "$status" -eq 0 ]]; then
  echo "Expected invalid_legacy_variants.csv to fail, but it passed." >&2
  exit 1
fi
set +e
python3 tools/paper_validate_main_table.py paper_results/validator_fixtures/invalid_missing_metrics.csv >"$validator_negative_log" 2>&1
status=$?
set -e
if [[ "$status" -eq 0 ]]; then
  echo "Expected invalid_missing_metrics.csv to fail, but it passed." >&2
  exit 1
fi
set +e
python3 tools/paper_validate_main_table.py --check-yaml paper_results/validator_fixtures/invalid_wrong_yaml.csv >"$validator_negative_log" 2>&1
status=$?
set -e
if [[ "$status" -eq 0 ]]; then
  echo "Expected invalid_wrong_yaml.csv --check-yaml to fail, but it passed." >&2
  exit 1
fi

write_minimal_collector_run() {
  local run_dir="$1"
  mkdir -p "$run_dir"
  cat > "${run_dir}/results.csv" <<'EOF'
epoch,metrics/mAP50-95(B),metrics/mAP50(B)
0,0.40,0.70
1,0.50,0.80
EOF
  cat > "${run_dir}/args.yaml" <<'EOF'
imgsz: 256
epochs: 800
batch: 64
mosaic: 1.0
close_mosaic: 700
EOF
}

assert_no_claim_usable_yes() {
  local csv_path="$1"
  python3 - "$csv_path" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
bad = [row for row in rows if row.get("claim_usable") == "yes"]
if bad:
    raise SystemExit(f"collector unexpectedly emitted claim_usable=yes: {bad}")
PY
}

cat > "${collector_dir}/results.csv" <<'EOF'
epoch,metrics/mAP50-95(B),metrics/mAP50(B)
0,0.40,0.70
1,0.50,0.80
EOF
cat > "${collector_dir}/args.yaml" <<'EOF'
imgsz: 256
epochs: 800
batch: 64
mosaic: 1.0
close_mosaic: 700
EOF
cat > "${collector_dir}/paper_run_meta.env" <<'EOF'
paper_protocol_id=ogsod_hbb_mosaic100_clean_a1b_probea_20260618
protocol_id=ogsod_hbb_mosaic100_clean_a1b_probea_20260618
paper_run=1
dataset=OGSOD-1.0
task=hbb
method=ladd_probea
method_label=LADD Probe-A
model_size=n
seed=0
phase_chain=A1->B
ladd_a1b_mode=dynamic_probe
run_tag=paper_clean_a1b_dynprobe_mosaic100_yolo11n_s0
project_dir=runs_public/paper/example
student_modality=SAR
teacher_modality=RGB
inference_modality=SAR
git_commit=deadbeef
status=verified
EOF
python3 tools/paper_collect_results.py --input "$collector_dir" --out "$collector_csv"
python3 tools/paper_validate_main_table.py "$collector_csv"

realistic_run_dir="${collector_bad_root}/runs_public/paper/ogsod_hbb_mosaic100/ladd_probea/yolo11n/seed0/RUN"
realistic_log_dir="${collector_bad_root}/logs/paper/ogsod_hbb_mosaic100/ladd_probea/yolo11n/seed0/RUN"
realistic_registry="${collector_bad_root}/registry.csv"
write_minimal_collector_run "$realistic_run_dir"
mkdir -p "$realistic_log_dir"
cat > "${realistic_log_dir}/paper_run_meta.env" <<'EOF'
paper_protocol_id=ogsod_hbb_mosaic100_clean_a1b_probea_20260618
protocol_id=ogsod_hbb_mosaic100_clean_a1b_probea_20260618
paper_run=1
dataset=OGSOD-1.0
task=hbb
method=ladd_probea
method_label=LADD Probe-A
model_size=n
seed=0
phase_chain=A1->B
ladd_a1b_mode=dynamic_probe
run_tag=paper_clean_a1b_dynprobe_mosaic100_yolo11n_s0
project_dir=runs_public/paper/example
student_modality=SAR
teacher_modality=RGB
inference_modality=SAR
data_cfg=configs/paper/datasets/ogsod_hbb_sar.yaml
teacher_data_cfg=configs/paper/datasets/ogsod_hbb_rgb.yaml
git_commit=deadbeef
status=verified
EOF
cat > "$realistic_registry" <<EOF
results_csv,args_yaml,manifest
${realistic_run_dir}/results.csv,${realistic_run_dir}/args.yaml,${realistic_log_dir}/paper_run_meta.env
EOF
python3 tools/paper_collect_results.py --input "$realistic_registry" --out "$collector_csv"
python3 tools/paper_validate_main_table.py --check-yaml "$collector_csv"

archive_collector_dir="${collector_bad_root}/archive_legacy_ladd_20260618/clean_a1b_dynprobe"
write_minimal_collector_run "$archive_collector_dir"
cat > "${archive_collector_dir}/paper_run_meta.env" <<'EOF'
paper_protocol_id=ogsod_hbb_mosaic100_clean_a1b_probea_20260618
protocol_id=ogsod_hbb_mosaic100_clean_a1b_probea_20260618
paper_run=1
dataset=OGSOD-1.0
task=hbb
method=ladd_probea
method_label=LADD Probe-A
model_size=n
seed=0
phase_chain=A1->B
ladd_a1b_mode=dynamic_probe
run_tag=paper_clean_a1b_dynprobe_mosaic100_yolo11n_s0
project_dir=runs_public/paper/example
student_modality=SAR
teacher_modality=RGB
inference_modality=SAR
git_commit=deadbeef
status=verified
EOF
python3 tools/paper_collect_results.py --input "$archive_collector_dir" --out "$collector_bad_csv"
assert_no_claim_usable_yes "$collector_bad_csv"

missing_git_dir="${collector_bad_root}/missing_git_clean_a1b_dynprobe"
write_minimal_collector_run "$missing_git_dir"
cat > "${missing_git_dir}/paper_run_meta.env" <<'EOF'
paper_protocol_id=ogsod_hbb_mosaic100_clean_a1b_probea_20260618
protocol_id=ogsod_hbb_mosaic100_clean_a1b_probea_20260618
paper_run=1
dataset=OGSOD-1.0
task=hbb
method=ladd_probea
method_label=LADD Probe-A
model_size=n
seed=0
phase_chain=A1->B
ladd_a1b_mode=dynamic_probe
run_tag=paper_clean_a1b_dynprobe_mosaic100_yolo11n_s0
project_dir=runs_public/paper/example
student_modality=SAR
teacher_modality=RGB
inference_modality=SAR
status=verified
EOF
python3 tools/paper_collect_results.py --input "$missing_git_dir" --out "$collector_bad_csv"
assert_no_claim_usable_yes "$collector_bad_csv"

missing_manifest_dir="${collector_bad_root}/missing_manifest_clean_a1b_dynprobe"
write_minimal_collector_run "$missing_manifest_dir"
python3 tools/paper_collect_results.py --input "$missing_manifest_dir" --out "$collector_bad_csv"
assert_no_claim_usable_yes "$collector_bad_csv"

status_all="$(git status --short --untracked-files=all)"
if printf '%s\n' "$status_all" \
  | grep -v '^?? comparison/cmdistill/native_reproduction/' \
  | grep -v '^?? tmp/' \
  | grep -E '(\.pt|\.pth|events\.out|wandb|\.onnx|\.engine|runs_public/.*/weights|data/raw|native_reproduction/data)'; then
  echo "Forbidden artifact appears in git status." >&2
  exit 1
fi
if printf '%s\n' "$status_all" | grep -E '^\?\? (comparison/cmdistill/native_reproduction|tmp)/' >/dev/null; then
  echo "WARNING: local-only untracked workspace exists: comparison/cmdistill/native_reproduction/ or tmp/."
fi
local_only_artifacts="$(find comparison/cmdistill/native_reproduction tmp -type f \( -name '*.pt' -o -name '*.pth' -o -name '*events.out*' -o -name '*.onnx' -o -name '*.engine' \) 2>/dev/null || true)"
if [[ -n "$local_only_artifacts" ]]; then
  echo "WARNING: local-only workspace contains large training artifacts; ensure they remain untracked." >&2
  printf '%s\n' "$local_only_artifacts" | head -n 20 >&2
fi

git diff --check

echo "OK: paper engineering cleanup validation passed."
