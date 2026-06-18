# Current Running Snapshot 2026-06-16

| host | GPU | run | epoch | AP50 | AP | best AP | status |
|---|---:|---|---:|---:|---:|---:|---|
| autodl | 0 | LADD n no-mosaic A1+B(A2core) | 445/800 | 0.81016 | 0.53035 | 0.53096@432 | running |
| autodl | 0 | LADD s no-mosaic A1+B(A2core) | 154/800 | 0.47114 | 0.25784 | 0.25784@154 | running |
| autodl | 0 | CMDistill n no-mosaic | 569/800 | 0.81317 | 0.54191 | 0.54191@569 | running |
| autodl | 0 | CMDistill s no-mosaic | 571/800 | 0.90470 | 0.61637 | 0.61673@569 | running |
| ladd90 | 0 | RGB yolo11m mosaic baseline | 346/800 | 0.94100 | 0.64676 | 0.64678@344 | running |
| ladd90 | 0 | SAR yolo11m mosaic baseline | 345/800 | 0.88309 | 0.60048 | 0.60048@345 | running |
| ladd90 | 1 | LADD n mosaic skipA2 s42 | 771/800 | 0.84608 | 0.56819 | 0.56835@746 | running |
| ladd90 | 1 | LADD n mosaic A2last s0 | 765/800 | 0.84732 | 0.56555 | 0.56563@734 | running |
| ladd90 | 3 | LADD n mosaic A2last s123 | 259/800 | 0.74411 | 0.47196 | 0.47196@259 | running |
| ladd90 | 5 | LADD m no-mosaic A1+B(A2core) | 59/800 | 0.50709 | 0.24601 | 0.24601@59 | running |

Figures:
- `figures/current_running_ap_curves_20260616.png`
- `figures/current_running_progress_20260616.png`
