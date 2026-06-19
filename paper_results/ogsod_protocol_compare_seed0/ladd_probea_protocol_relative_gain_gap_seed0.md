# Relative Protocol Comparison: LADD Probe-A seed0

This table intentionally compares only same-protocol relative quantities. Main formulas:

- `Gain vs SAR best = LADD_best - SAR_best` within the same protocol/model/seed.
- `Gap closed = (LADD_best - SAR_best) / (RGB_best - SAR_best)`.
- Same-epoch columns use SAR/RGB AP at the LADD best epoch and are useful for partial mosaic snapshots.

## Per-Protocol Relative Results

| Protocol   | Model   | LADD status   | Ref status   |   Rows |   Gain vs SAR best | Gap closed vs best gap   |   LADD best epoch |   Gain vs SAR same epoch | Gap closed same epoch   |
|:-----------|:--------|:--------------|:-------------|-------:|-------------------:|:-------------------------|------------------:|-------------------------:|:------------------------|
| no-mosaic  | YOLO11n | complete      | complete     |    800 |            0.01779 | 24.2%                    |               697 |                  0.02067 | 27.8%                   |
| no-mosaic  | YOLO11s | complete      | complete     |    800 |            0.01176 | 41.0%                    |               640 |                  0.02054 | 56.6%                   |
| no-mosaic  | YOLO11m | complete      | complete     |    800 |            0.01402 | 60.2%                    |               603 |                  0.02023 | 72.8%                   |
| mosaic100  | YOLO11n | partial       | complete     |    347 |           -0.04313 | -57.4%                   |               347 |                  0.04559 | 48.6%                   |
| mosaic100  | YOLO11s | complete      | partial_ref  |    800 |            0.01515 | 37.3%                    |               708 |                  0.01619 | 40.7%                   |
| mosaic100  | YOLO11m | partial       | partial_ref  |    493 |            0.00528 | 23.5%                    |               493 |                  0.0261  | 76.1%                   |

## Protocol Delta on Relative Gain

| Model   |   No-mosaic gain |   Mosaic100 gain |   Mosaic - no gain | No-mosaic gap closed   | Mosaic100 gap closed   | Mosaic - no gap closed   | No status   | Mosaic status   |
|:--------|-----------------:|-----------------:|-------------------:|:-----------------------|:-----------------------|:-------------------------|:------------|:----------------|
| YOLO11n |          0.01779 |         -0.04313 |           -0.06092 | +24.2%                 | -57.4%                 | -81.5%                   | complete    | partial         |
| YOLO11s |          0.01176 |          0.01515 |            0.00339 | +41.0%                 | +37.3%                 | -3.6%                    | complete    | complete        |
| YOLO11m |          0.01402 |          0.00528 |           -0.00874 | +60.2%                 | +23.5%                 | -36.7%                   | complete    | partial         |
