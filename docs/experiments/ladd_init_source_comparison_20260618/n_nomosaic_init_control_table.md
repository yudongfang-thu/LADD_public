# YOLO11n no-mosaic init-source control table

SAR baseline best AP is `0.55654@734`; SAR baseline best AP50 is `0.82568@751`.

| Run | Role | Epochs | First AP | Best AP @ epoch | Gap AP vs SAR best | Last AP | Last AP gap vs SAR best | Best AP50 @ epoch | Gap AP50 vs SAR best |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO-init det-only | init-source control | 332 | 0.01768 | 0.45155 @332 | -0.10499 | 0.45155 | -0.10499 | 0.70449 @332 | -0.12119 |
| SAR-last det-only | init-source control | 338 | 0.53600 | 0.57687 @337 | +0.02033 | 0.57661 | +0.02007 | 0.84907 @336 | +0.02339 |
| SAR baseline | baseline reference | 800 | 0.05211 | 0.55654 @734 | +0.00000 | 0.55127 | -0.00527 | 0.82568 @751 | +0.00000 |
| RGB baseline | baseline reference | 800 | 0.13342 | 0.63018 @723 | +0.07364 | 0.62737 | +0.07083 | 0.93028 @729 | +0.10460 |

Interpretation: `YOLO-init det-only` and `SAR-last det-only` are the cleanest paired controls here because both use the B800 schedule and turn off LADD losses; their main difference is detector initialization.
The SAR/RGB baseline curves are reference curves from the same formal no-mosaic protocol, not B-stage continuation runs.
