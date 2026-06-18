# LADD initialization-source comparison notes

Date: 2026-06-18

This note consolidates earlier records about whether LADD/B-stage experiments started from a converged SAR baseline checkpoint or from YOLO initial weights. It does not introduce new training results; it indexes existing lightweight `results.csv` summaries.

## Main takeaways

1. We did run YOLO-init diagnostics. Under the no-mosaic B800 restart batch, YOLO-init detector runs were far below converged-SAR-baseline continuation at the available snapshots.
2. The strongest short entrance evidence for split-load LADD used a converged SAR baseline detector plus A2 teacher decomposition, not YOLO init.
3. The current mosaic100 clean A1B mainline uses a converged SAR baseline checkpoint as the A1 starting detector, then B starts from A1 best. It is therefore a SAR-baseline-init A1->B protocol, not a YOLO-init protocol.
4. A fully fair init-source ablation would require the same mosaic100 clean A1B code path with only the initial checkpoint changed. We do not yet have that clean paired ablation; the historical YOLO-init evidence is directional and negative.

## Summary CSV

- `init_source_comparison_summary.csv`

## no-mosaic B800 restart

| model | schedule | run | init source | rows | best AP @ epoch | last AP | status | caveat |
|---|---|---|---|---:|---:|---:|---|---|
| YOLO11n | B800 cosine | N0 YOLO-init det-only B800sched | YOLO init detector, det-only control | 332 | 0.45155 @332 | 0.45155 | running snapshot | YOLO initial detector; no LADD loss. |
| YOLO11n | B800 cosine | N1 SAR-best det-only B800sched | converged SAR baseline best checkpoint, det-only control | 332 | 0.57521 @324 | 0.57494 | running snapshot | SAR baseline best checkpoint continued with detection-only B. |
| YOLO11n | B800 cosine | N1 SAR-last det-only B800sched | converged SAR baseline last checkpoint, det-only control | 338 | 0.57687 @337 | 0.57661 | running snapshot | SAR baseline last checkpoint continued with detection-only B. |
| YOLO11n | B800 cosine | N2 A2-best full LADD B800sched | A2 best full checkpoint, full LADD B | 229 | 0.55681 @214 | 0.54271 | stopped, NaN | Continues from A2 best checkpoint; crashed after NaN recovery failed. |
| YOLO11n | B800 cosine | N2 A2-last full LADD B800sched | A2 last full checkpoint, full LADD B | 319 | 0.56073 @271 | 0.46290 | stopped, NaN | Continues from A2 last checkpoint; crashed after NaN recovery failed. |
| YOLO11n | B800 cosine | N3 YOLO-init + A2 decomp B800sched | YOLO init detector + A2 teacher decomposition | 525 | 0.48742 @525 | 0.48742 | running snapshot | YOLO initial detector plus A2 decomposition split-load. |
| YOLO11n | B800 cosine | N4 YOLO-init + decomp KD-warmup B800sched | YOLO init detector + A2 teacher decomposition + KD warmup | 512 | 0.46180 @512 | 0.46180 | running snapshot | YOLO initial detector plus A2 decomposition and KD-only warmup. |

## no-mosaic B entrance compressed

| model | schedule | run | init source | rows | best AP @ epoch | last AP | status | caveat |
|---|---|---|---|---:|---:|---:|---|---|
| YOLO11n | B100 compressed | N1 current: SAR baseline cont. B100 | converged SAR baseline checkpoint, det-only B | 100 | 0.56615 @99 | 0.56594 | completed compressed entrance | Short compressed schedule; useful for entrance trend, not final B800 claim. |
| YOLO11n | B100 compressed | N2 current: A2-best cont. B100 | A2 best checkpoint | 100 | 0.55872 @100 | 0.55872 | completed compressed entrance | Short compressed schedule; useful for entrance trend, not final B800 claim. |
| YOLO11n | B100 compressed | N3 current: SAR-base + A2-last decomp B100 | converged SAR baseline detector + A2 teacher decomposition | 100 | 0.55722 @100 | 0.55722 | completed compressed entrance | Short compressed schedule; useful for entrance trend, not final B800 claim. |
| YOLO11n | B120 compressed | N4 current: N3 + KD ramp B120 | converged SAR baseline detector + A2 teacher decomposition + KD ramp | 120 | 0.56379 @113 | 0.56311 | completed compressed entrance | Short compressed schedule; useful for entrance trend, not final B800 claim. |
| YOLO11s | B100 compressed | S1 current: SAR baseline cont. B100 | converged SAR baseline checkpoint, det-only B | 100 | 0.62493 @62 | 0.62238 | completed compressed entrance | Short compressed schedule; useful for entrance trend, not final B800 claim. |
| YOLO11s | B100 compressed | S2 current: A2-best cont. B100 | A2 best checkpoint | 100 | 0.62599 @54 | 0.62174 | completed compressed entrance | Short compressed schedule; useful for entrance trend, not final B800 claim. |
| YOLO11s | B100 compressed | S3 current: SAR-base + A2-last decomp B100 | converged SAR baseline detector + A2 teacher decomposition | 100 | 0.62553 @65 | 0.62262 | completed compressed entrance | Short compressed schedule; useful for entrance trend, not final B800 claim. |
| YOLO11s | B120 compressed | S4 current: S3 + KD ramp B120 | converged SAR baseline detector + A2 teacher decomposition + KD ramp | 120 | 0.62521 @62 | 0.62111 | completed compressed entrance | Short compressed schedule; useful for entrance trend, not final B800 claim. |

## protocol split yolo-init check

| model | schedule | run | init source | rows | best AP @ epoch | last AP | status | caveat |
|---|---|---|---|---:|---:|---:|---|---|
| YOLO11n | mixed historical/active | SAR baseline n no-mosaic | same-protocol SAR baseline | 800 | 0.55654 @734 | 0.55127 | historical/diagnostic | Some LADD-like rows include sep/aux contamination; use as directional evidence only. |
| YOLO11n | mixed historical/active | Current diagnostic n yolo-init A1->B_A2core (sep/aux) | YOLO init A1/B diagnostic | 684 | 0.54678 @615 | 0.54206 | historical/diagnostic | Some LADD-like rows include sep/aux contamination; use as directional evidence only. |
| YOLO11s | mixed historical/active | SAR baseline s no-mosaic | same-protocol SAR baseline | 800 | 0.62897 @729 | 0.62233 | historical/diagnostic | Some LADD-like rows include sep/aux contamination; use as directional evidence only. |
| YOLO11s | mixed historical/active | LADD-like s yolo-init (sep/aux) | YOLO init A1/B diagnostic | 177 | 0.28862 @177 | 0.28862 | historical/diagnostic | Some LADD-like rows include sep/aux contamination; use as directional evidence only. |

## current mosaic100 clean A1B

| model | schedule | run | init source | rows | best AP @ epoch | last AP | status | caveat |
|---|---|---|---|---:|---:|---:|---|---|
| YOLO11n | A1 -> B 800 | YOLO11n Static | converged SAR baseline checkpoint -> A1 best -> B | 800 | 0.57113 @758 | 0.56836 | running/completed current mainline | Current preferred protocol; not a YOLO-init comparison. |
| YOLO11n | A1 -> B 800 | YOLO11n Dynamic | converged SAR baseline checkpoint -> A1 best -> B | 792 | 0.57544 @749 | 0.57102 | running/completed current mainline | Current preferred protocol; not a YOLO-init comparison. |
| YOLO11s | A1 -> B 800 | YOLO11s Static | converged SAR baseline checkpoint -> A1 best -> B | 725 | 0.62600 @725 | 0.62600 | running/completed current mainline | Current preferred protocol; not a YOLO-init comparison. |
| YOLO11s | A1 -> B 800 | YOLO11s Dynamic | converged SAR baseline checkpoint -> A1 best -> B | 567 | 0.62303 @567 | 0.62303 | running/completed current mainline | Current preferred protocol; not a YOLO-init comparison. |
| YOLO11s | A1 -> B 800 | YOLO11s Probe-A | converged SAR baseline checkpoint -> A1 best -> B | 450 | 0.60582 @450 | 0.60582 | running/completed current mainline | Current preferred protocol; not a YOLO-init comparison. |

## Interpretation

- For current paper-facing LADD mainline work, converged SAR-baseline initialization is the more defensible default because it is stable, matches the current launcher, and gives clean detector capacity before decomposition/distillation.
- YOLO-init remains useful as an ablation only if we explicitly want to claim the method can learn from raw YOLO pretrained weights without first training a SAR detector. The existing evidence does not support prioritizing that path.
- Do not mix no-mosaic B800 restart, compressed B100/B120 entrance runs, and current mosaic100 clean A1B runs as one final performance comparison. They answer different questions.

## Source files

- `ladd/results/b800_restart_20260614/summary/ladd_b800_restart_curve_summary_20260614.csv`
- `docs/experiments/figures/ladd_b_stage_historical_compare_20260614/b_stage_historical_compare_summary_20260614.csv`
- `docs/experiments/figures/ladd_capacity_protocol_split_20260617/capacity_protocol_split_summary_20260617.csv`
- `docs/experiments/ladd_mosaic100_mainline_curves_20260618/summary.csv`
