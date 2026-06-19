# VEDAI / DroneVehicle Cross-Dataset Records

This directory is reserved for optional cross-dataset validation. These rows are not OGSOD main-table candidates.

Current active extension:

```text
protocol_id: cclkd_yolo11n_cross_dataset_20260619
method under test: LADD Probe-A / clean_a1b_dynprobe
reference protocol: CCLKD YOLO11n extension tables
datasets: VEDAI, DroneVehicle
```

Authoritative protocol and launchers:

```text
docs/paper/CCLKD_YOLO11N_CROSS_DATASET_PROTOCOL_CN.md
configs/paper/cclkd_yolo11n_cross_dataset.yaml
scripts/paper/run_paper_cclkd_yolo11n_cross_baseline.sh
scripts/paper/run_paper_ladd_cclkd_yolo11n_cross_dataset.sh
```

Reporting rules:

- VEDAI direction: RGB / visible teacher -> IR student.
- DroneVehicle direction: IR teacher -> RGB / visible student.
- Our rows should include student baseline, teacher baseline, and LADD Probe-A.
- CCLKD / CMDistill YOLO11n rows copied from the CCLKD paper must be marked as reported results.
- Do not merge these rows into the OGSOD mosaic100 main table.
