# Active Runs by Model with Baseline References

Snapshot time: 2026-06-16 14:22 +08

SAR baseline is the directly comparable single-modality baseline. RGB teacher is plotted as a reference upper line.

| run | epoch | AP50 | AP | SAR base AP same epoch | delta AP same epoch | SAR base best AP | delta AP vs best | SAR base final AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n LADD | 376 | 0.79155 | 0.51903 | 0.47946 | 0.03957 | 0.55654@734 | -0.03751 | 0.55127 |
| YOLO11n CMDistill | 479 | 0.78874 | 0.51973 | 0.51022 | 0.00951 | 0.55654@734 | -0.03681 | 0.55127 |
| YOLO11s LADD | 89 | 0.38128 | 0.18571 | 0.41199 | -0.22628 | 0.62897@729 | -0.44326 | 0.62233 |
| YOLO11s CMDistill | 480 | 0.88597 | 0.60151 | 0.59755 | 0.00396 | 0.62897@729 | -0.02746 | 0.62233 |
| YOLO11m LADD | 38 | 0.44117 | 0.19295 | 0.32312 | -0.13017 | 0.65580@704 | -0.46285 | 0.64903 |
