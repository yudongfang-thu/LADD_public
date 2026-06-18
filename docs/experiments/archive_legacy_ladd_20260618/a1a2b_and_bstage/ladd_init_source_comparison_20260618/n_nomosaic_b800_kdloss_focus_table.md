# YOLO11n no-mosaic B800 KD-loss focus table

| Run | Epochs | Best AP @ epoch | Last AP | KD first | KD last | KD max | nonzero KD epochs | Last KD/det |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N0 YOLO-init det-only | 332 | 0.45155 @332 | 0.45155 | 0.00000 | 0.00000 | 0.00000 | 0 | 0.00000 |
| N1 SAR-last det-only | 338 | 0.57687 @337 | 0.57661 | 0.00000 | 0.00000 | 0.00000 | 0 | 0.00000 |
| N2 A2-best full LADD | 229 | 0.55681 @214 | 0.54271 | 4.10515 | 0.41855 | 4.10515 | 229 | 0.22070 |
| N2 A2-last full LADD | 319 | 0.56073 @271 | 0.46290 | 4.06005 | 0.24685 | 4.06005 | 319 |  |
| N3 YOLO-init + A2 decomp | 525 | 0.48742 @525 | 0.48742 | 3.60590 | 0.10773 | 3.60590 | 525 | 0.04343 |
| N4 YOLO-init + A2 decomp KD-warm | 512 | 0.46180 @512 | 0.46180 | 0.00000 | 0.21310 | 1.05440 | 462 | 0.07952 |

N0/N1 are det-only controls and therefore have zero KD loss by construction. They are useful initialization controls, not KD-behavior runs.
