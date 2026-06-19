# Mosaic100 LADD Mainline Performance Table

Delta columns are computed against the same-capacity SAR baseline best value. `gap_to_RGB` is best AP minus same-capacity RGB baseline best AP.

| Model | Method | Epochs | Last AP | Best AP @ epoch | Delta AP vs SAR | Gap AP to RGB | Last AP50 | Best AP50 @ epoch | Delta AP50 vs SAR | Gap AP50 to RGB | Note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| YOLO11n | Dynamic | 800 | 0.57030 | 0.57544 @749 | 0.03453 | -0.04066 | 0.84849 | 0.85286 @750 | 0.03345 | -0.06763 |  |
| YOLO11n | LADD | 347 | 0.49778 | 0.49778 @347 | -0.04313 | -0.11832 | 0.77260 | 0.77260 @347 | -0.04681 | -0.14789 |  |
| YOLO11n | Static | 800 | 0.56836 | 0.57113 @758 | 0.03022 | -0.04497 | 0.84721 | 0.84949 @770 | 0.03008 | -0.07100 |  |
| YOLO11s | Dynamic | 662 | 0.63003 | 0.63647 @656 | 0.01675 | -0.02382 | 0.91917 | 0.92123 @649 | 0.01286 | -0.02579 |  |
| YOLO11s | LADD | 800 | 0.62764 | 0.63487 @708 | 0.01515 | -0.02542 | 0.91364 | 0.91981 @714 | 0.01144 | -0.02721 |  |
| YOLO11s | Static | 800 | 0.62425 | 0.62716 @739 | 0.00744 | -0.03313 | 0.91334 | 0.91334 @800 | 0.00497 | -0.03368 |  |
