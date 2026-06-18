# Comparison Curves 2026-06-17

Curves compare LD, FGD, CMDistill, HalluciDet, LADD reference, SAR baseline, and RGB teacher.
All plotted comparison methods use formal no-mosaic, 800 epoch, YOLO11n/s, batch 64 where applicable.

No valid local `results.csv` was found for a same-protocol bimodal/fusion baseline; it is therefore not plotted.
Add its `results.csv` to `RUNS` in `plot_comparison_curves.py` and rerun this script when available.

## Outputs

- `comparison_curves_yolo11n_main.png/pdf`
- `comparison_curves_yolo11s_main.png/pdf`
- `comparison_curve_summary.csv`
