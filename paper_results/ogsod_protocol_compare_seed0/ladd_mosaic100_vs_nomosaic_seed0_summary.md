# LADD Protocol Comparison: mosaic100 vs no-mosaic

Metric: AP50-95. `Gap closed = (LADD - SAR) / (RGB - SAR)`. Rows marked `partial` are snapshots, not completed 800-epoch evidence.

## Per-Protocol Results

| Protocol   | Model   | LADD status   | Ref status   |   LADD rows |   SAR rows |   RGB rows |   LADD best |   SAR best |   RGB best |   Gain vs SAR | Gap closed   |   Best epoch |   Latest/final LADD |   Latest epoch |
|:-----------|:--------|:--------------|:-------------|------------:|-----------:|-----------:|------------:|-----------:|-----------:|--------------:|:-------------|-------------:|--------------------:|---------------:|
| no-mosaic  | YOLO11n | complete      | complete     |         800 |        800 |        800 |     0.57433 |    0.55654 |    0.63018 |       0.01779 | 24.2%        |          697 |             0.56415 |            800 |
| no-mosaic  | YOLO11s | complete      | complete     |         800 |        800 |        800 |     0.64073 |    0.62897 |    0.65768 |       0.01176 | 41.0%        |          640 |             0.62741 |            800 |
| no-mosaic  | YOLO11m | complete      | complete     |         800 |        800 |        800 |     0.66982 |    0.6558  |    0.67909 |       0.01402 | 60.2%        |          603 |             0.65275 |            800 |
| mosaic100  | YOLO11n | partial       | complete     |         347 |        800 |        800 |     0.49778 |    0.54091 |    0.6161  |      -0.04313 | -57.4%       |          347 |             0.49778 |            347 |
| mosaic100  | YOLO11s | complete      | partial_ref  |         800 |        800 |        759 |     0.63487 |    0.61972 |    0.66029 |       0.01515 | 37.3%        |          708 |             0.62764 |            800 |
| mosaic100  | YOLO11m | partial       | partial_ref  |         493 |        793 |        680 |     0.6562  |    0.65092 |    0.6734  |       0.00528 | 23.5%        |          493 |             0.6562  |            493 |

## Protocol Gain Difference

| Model   |   No-mosaic gain |   Mosaic100 gain |   Mosaic - no | No status   | Mosaic status   | Interpretation                       |
|:--------|-----------------:|-----------------:|--------------:|:------------|:----------------|:-------------------------------------|
| YOLO11n |          0.01779 |         -0.04313 |      -0.06092 | complete    | partial         | no-mosaic higher / mosaic incomplete |
| YOLO11s |          0.01176 |          0.01515 |       0.00339 | complete    | complete        | mosaic higher                        |
| YOLO11m |          0.01402 |          0.00528 |      -0.00874 | complete    | partial         | no-mosaic higher / mosaic incomplete |
