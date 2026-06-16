# Running Status 2026-06-16 10:02 CST

## AutoDL GPU

- GPU: NVIDIA GeForce RTX 4090 D
- Memory: 14393 MiB used / 9699 MiB free
- Utilization: 99%

Active compute processes:

| PID | job | GPU memory |
|---:|---|---:|
| 66242 | Current LADD cap2 YOLO11n B800 | 4438 MiB |
| 67298 | CMDistill YOLO11n formal800 | 3832 MiB |
| 67304 | CMDistill YOLO11s formal800 | 6084 MiB |

## Main Runs

| run | status | latest completed epoch | AP50 | AP | diagnostics |
|---|---|---:|---:|---:|---|
| CMDistill YOLO11n formal800 | running | 304 | 0.73608 | 0.46990 | finite |
| CMDistill YOLO11s formal800 | running | 304 | 0.84889 | 0.56709 | finite |
| Current LADD cap2 YOLO11n B800 | running | 247 | 0.74891 | 0.48544 | finite |

CMDistill latest diagnostics from the pulled snapshot:

| run | cmd total | PCC | SLRD | IBCLD | candidate ratio | nonfinite |
|---|---:|---:|---:|---:|---:|---:|
| CMDistill YOLO11n | 1.09174 | 0.42254 | 0.09578 | 0.57342 | 0.01181 | 0 |
| CMDistill YOLO11s | 0.95852 | 0.28643 | 0.04211 | 0.62998 | 0.01116 | 0 |

## Stopped Runs

The two `cclkd_table2_noextraaug_20260616` YOLO11n 400 epoch baselines were stopped because the protocol was not producing a useful baseline for CMDistill/CCLKD alignment.

| run | final completed epoch | latest AP50 | latest AP | best AP |
|---|---:|---:|---:|---:|
| SAR no-extra YOLO11n | 399 | 0.46086 | 0.25156 | 0.27494 |
| RGB no-extra YOLO11n | 401 | 0.75812 | 0.45467 | 0.46811 |

## Next Monitoring Point

- Continue monitoring CMDistill n/s formal800 and Current LADD cap2 n B800.
- Do not restart no-extra-aug baselines.
- Do not compare no-extra-aug runs against formal no-mosaic 800 runs except as negative protocol evidence.
