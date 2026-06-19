# LADD Probe-A Nomosaic Seed0 Mainline Summary

Metric: AP50-95. Gap closed = (LADD - SAR baseline) / (RGB teacher - SAR baseline).

| Model   |   SAR best |   RGB best |   LADD best |   Gain vs SAR | Gap closed   | Remaining gap   |   LADD best epoch |   LADD final |   Final gain | Final gap closed   |
|:--------|-----------:|-----------:|------------:|--------------:|:-------------|:----------------|------------------:|-------------:|-------------:|:-------------------|
| YOLO11n |    0.55654 |    0.63018 |     0.57433 |       0.01779 | 24.2%        | 75.8%           |               697 |      0.56415 |      0.01288 | 16.9%              |
| YOLO11s |    0.62897 |    0.65768 |     0.64073 |       0.01176 | 41.0%        | 59.0%           |               640 |      0.62741 |      0.00508 | 18.6%              |
| YOLO11m |    0.6558  |    0.67909 |     0.66982 |       0.01402 | 60.2%        | 39.8%           |               603 |      0.65275 |      0.00372 | 16.5%              |
