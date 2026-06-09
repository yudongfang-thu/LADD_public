# YOLOv5 Sanity Summary

## Baseline Target Reminder

- YOLOv5 CSPDarkNet-X / YOLOv5x target: 86.23M params, AP50 80.9, AP 46.3.
- Loose pass threshold: AP50 >= 78 and AP >= 44.

## Best AP Ranking

| rank | run | best AP50 | best AP | pass |
|---:|---|---:|---:|---|

## Comparison Against Target

| run | delta AP50 | delta AP | params M | class order |
|---|---:|---:|---:|---|

## Per-Class AP50

| run | Oil Tank | Bridge | Harbor |
|---|---:|---:|---:|

## Failure Diagnosis Hints

- If class-specific AP is mismatched, especially Oil Tank, check class mapping and small-object distribution.
