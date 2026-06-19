# LADD Mosaic100 Results

最后更新：2026-06-18

本文只记录符合 paper gate 的 LADD / `clean_a1b_dynprobe` / mosaic100 结果。旧 A1-A2-B、no-mosaic、BN-freeze、close@100、400ep、partial、smoke 和 diagnostic 结果只放链接或归档说明，不在本文中形成主表结论。

## Validity Rule

进入本文主结果区的行必须同时满足：

```text
protocol = mosaic100
imgsz = 256
epochs_B = 800
mosaic = 1.0
close_mosaic = 700
phase_chain = A -> B
method = LADD
LADD_A1B_MODE = dynamic_probe
no A2
same model size
same seed
same SAR baseline checkpoint
same RGB teacher checkpoint
same dataset yaml / same class order
has results.csv
has args.yaml
has manifest/meta with git commit
status = verified
usable_for_main_table = yes
```

Canonical CSV 来源：

- `paper_results/ogsod_mosaic100/ladd.csv`
- `paper_results/ogsod_mosaic100/main_table_candidate.csv`

## YOLO11n

| seed | SAR | RGB | LADD | gain | remaining gap | best epoch | final | status | source |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending paper gate | `paper_results/` |

## YOLO11s

| seed | SAR | RGB | LADD | gain | remaining gap | best epoch | final | status | source |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending paper gate | `paper_results/` |

## YOLO11m

| seed | SAR | RGB | LADD | gain | remaining gap | best epoch | final | status | source |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending paper gate | `paper_results/` |

## Invalid / Archived Rows

Only links are allowed here; no paper main conclusion should be drawn from these rows.

- Historical LADD archive: `ladd/results/archive_legacy_ladd_20260618/`
- Historical experiment archive: `docs/experiments/archive_legacy_ladd_20260618/`
- Current curve snapshots before paper gate finalization: `docs/experiments/ladd_mosaic100_mainline_curves_20260618/`
