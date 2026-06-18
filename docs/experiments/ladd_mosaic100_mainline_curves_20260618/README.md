# LADD Mosaic100 Mainline Curves Snapshot

Snapshot time: 2026-06-18 00:43 +08.

This folder compares the current mosaic100 LADD mainline candidates by model
capacity:

- YOLO11n: Static, Dynamic, Probe-A snapshot.
- YOLO11s: Static, Dynamic snapshot, Probe-A.

The performance panels include same-capacity SAR and RGB baseline curves, and
also mark their best AP/AP50 as horizontal reference lines. In this report,
"dual-modal baseline" is interpreted as the pair of modality baselines used by
the LADD setting: SAR-only detector baseline and RGB teacher/reference baseline.

## Caveat

The 4090 server SSH authentication failed during this refresh. For YOLO11s,
Static/Dynamic curves are therefore taken from the latest local 4090 sync, and
star markers indicate later status snapshots beyond the synced curves. For
YOLO11n Probe-A, only the latest snapshot marker is available until 4090 access
is restored.

Older 90-server Static/skip-A2 curves are intentionally not used as substitutes
in the cleaned mainline figures, because they are not the same evidence line as
the current clean A1->B runs.

## Outputs

- `summary.csv`: latest/best numbers and data provenance.
- `performance_table.md`: readable performance table with delta vs same-capacity
  SAR baseline and gap to same-capacity RGB baseline.
- `figures/ladd_mosaic100_yolo11n_mainline_curves.png`
- `figures/ladd_mosaic100_yolo11n_mainline_curves.pdf`
- `figures/ladd_mosaic100_yolo11s_mainline_curves.png`
- `figures/ladd_mosaic100_yolo11s_mainline_curves.pdf`
