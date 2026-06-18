# YOLO11n no-mosaic LADD-like init-source comparison

SAR baseline best AP reference: `0.55654`.

| Run | Init source | Schedule | Epochs | First AP | Best AP @ epoch | Gap vs SAR best | Last AP | KD first | KD last | KD max | Last KD/det | s_rec first -> last |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAR-base + A2 decomp B100 | SAR baseline detector | B100 compressed | 100 | 0.51435 | 0.55722 @100 | +0.00068 | 0.55722 | 3.99304 | 0.23837 | 3.99304 | 0.10881 | 0.05096 -> 0.00418 |
| SAR-base + A2 decomp KD-ramp B120 | SAR baseline detector | B120 compressed | 120 | 0.53806 | 0.56379 @113 | +0.00725 | 0.56311 | 0.00000 | 0.33480 | 0.40587 | 0.15559 | 0.04451 -> 0.00337 |
| YOLO-init + A2 decomp B800 | YOLO init detector | B800 snapshot | 525 | 0.05308 | 0.48742 @525 | -0.06912 | 0.48742 | 3.60590 | 0.10773 | 3.60590 | 0.04343 | 0.04516 -> 0.00095 |
| YOLO-init + A2 decomp KD-warm B800 | YOLO init detector | B800 snapshot | 512 | 0.02484 | 0.46180 @512 | -0.09474 | 0.46180 | 0.00000 | 0.21310 | 1.05440 | 0.07952 | 0.03990 -> 0.00149 |

Caveat: the SAR-baseline-init split-load runs are B100/B120 compressed entrance diagnostics, while YOLO-init split-load runs are B800 running snapshots. They compare entrance behavior and loss scale, not final converged B800 capacity.
