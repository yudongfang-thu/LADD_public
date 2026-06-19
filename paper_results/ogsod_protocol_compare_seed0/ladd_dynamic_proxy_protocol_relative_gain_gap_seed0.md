# LADD Protocol Relative Analysis With Dynamic Mosaic Proxy

更新：2026-06-19

口径：只比较同协议内相对 SAR baseline 的提升，以及相对 RGB teacher gap 的闭合比例；不比较跨协议绝对 AP。

注意：mosaic100 中 YOLO11n/YOLO11s 使用 `clean_a1b_dyn` dynamic LADD 作为临时 proxy，不是正式 LADD 主线；YOLO11m 未找到本地 dynamic proxy，暂保留已有 partial LADD。

| protocol   | model   | ladd_source                                                      | substitute_for_probea   | status   | ref_status   |   ladd_rows |   sar_best_ap50_95 |   rgb_best_ap50_95 |   ladd_best_ap50_95 |   ladd_best_epoch |   best_to_best_gain_ap50_95 |   best_to_best_gap_closed_pct |   same_epoch_gain_ap50_95 |   same_epoch_gap_closed_pct |   ladd_latest_ap50_95 | notes                                      |
|:-----------|:--------|:-----------------------------------------------------------------|:------------------------|:---------|:-------------|------------:|-------------------:|-------------------:|--------------------:|------------------:|----------------------------:|------------------------------:|--------------------------:|----------------------------:|----------------------:|:-------------------------------------------|
| no-mosaic  | YOLO11n | LADD clean_a1b_dynprobe                                       | no                      | complete | complete     |         800 |            0.55654 |            0.63018 |             0.57433 |               697 |                     0.01779 |                         24.16 |                   0.02067 |                       27.84 |               0.56415 | official complete no-mosaic LADD        |
| no-mosaic  | YOLO11s | LADD clean_a1b_dynprobe                                       | no                      | complete | complete     |         800 |            0.62897 |            0.65768 |             0.64073 |               640 |                     0.01176 |                         40.96 |                   0.02054 |                       56.63 |               0.62741 | official complete no-mosaic LADD        |
| no-mosaic  | YOLO11m | LADD clean_a1b_dynprobe                                       | no                      | complete | complete     |         800 |            0.6558  |            0.67909 |             0.66982 |               603 |                     0.01402 |                         60.2  |                   0.02023 |                       72.82 |               0.65275 | official complete no-mosaic LADD        |
| mosaic100  | YOLO11n | Dynamic LADD clean_a1b_dyn proxy                                 | yes                     | complete | complete     |         800 |            0.54091 |            0.6161  |             0.57544 |               749 |                     0.03453 |                         45.92 |                   0.03479 |                       46.47 |               0.5703  | dynamic proxy complete; not paper mainline |
| mosaic100  | YOLO11s | Dynamic LADD clean_a1b_dyn proxy                                 | yes                     | partial  | partial_ref  |         712 |            0.61972 |            0.66029 |             0.63647 |               656 |                     0.01675 |                         41.29 |                   0.02305 |                       50.6  |               0.60079 | dynamic proxy partial; not paper mainline  |
| mosaic100  | YOLO11m | LADD clean_a1b_dynprobe partial; no local dynamic proxy found | no_dynamic_available    | partial  | partial_ref  |         493 |            0.65092 |            0.6734  |             0.6562  |               493 |                     0.00528 |                         23.49 |                   0.0261  |                       76.14 |               0.6562  | fallback partial LADD; not complete     |


## Mosaic Proxy - No-mosaic Delta

| model   | mosaic_ladd_source                                               | mosaic_substitute_for_probea   | mosaic_status   |   nomosaic_gain |   mosaic_proxy_gain |   mosaic_proxy_minus_nomosaic_gain |   nomosaic_gap_closed_pct |   mosaic_proxy_gap_closed_pct |   mosaic_proxy_minus_nomosaic_gap_pct |
|:--------|:-----------------------------------------------------------------|:-------------------------------|:----------------|----------------:|--------------------:|-----------------------------------:|--------------------------:|------------------------------:|--------------------------------------:|
| YOLO11n | Dynamic LADD clean_a1b_dyn proxy                                 | yes                            | complete        |         0.01779 |             0.03453 |                            0.01674 |                     24.16 |                         45.92 |                                 21.76 |
| YOLO11s | Dynamic LADD clean_a1b_dyn proxy                                 | yes                            | partial         |         0.01176 |             0.01675 |                            0.00499 |                     40.96 |                         41.29 |                                  0.33 |
| YOLO11m | LADD clean_a1b_dynprobe partial; no local dynamic proxy found | no_dynamic_available           | partial         |         0.01402 |             0.00528 |                           -0.00874 |                     60.2  |                         23.49 |                                -36.71 |


## Interpretation

- `best_to_best_gain_ap50_95` 是正式比较口径：同模型、同 seed、同协议下 LADD best AP50-95 减 SAR baseline best AP50-95。

- `best_to_best_gap_closed_pct` 是 `(LADD_best - SAR_best) / (RGB_best - SAR_best)`；它描述 LADD 补上了多少 SAR-RGB teacher gap。

- mosaic100 dynamic proxy 只能用于判断趋势和是否值得补正式 LADD；不能直接进入论文主表。

- 当前 proxy 显示 mosaic100 的 n 相对提升明显大于 no-mosaic，s 基本接近，m 因没有 dynamic proxy 仍不能得出最终判断。
