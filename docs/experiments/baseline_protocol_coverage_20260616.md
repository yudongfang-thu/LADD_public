# OGSOD HBB Baseline Protocol Coverage - 2026-06-16

This note tracks the baseline/teacher coverage needed before launching clean LADD A1B runs.

## Protocols

| Protocol key | Training args | Meaning |
|---|---|---|
| `nomosaic` | `mosaic=0.0, close_mosaic=0, epochs=800` | Formal no-mosaic protocol. |
| `mosaic100` | `mosaic=1.0, close_mosaic=700, epochs=800` | 800 epochs with mosaic active for the first 100 epochs, then closed for the remaining 700 epochs. |

The standard launcher is:

```bash
PROTOCOL=nomosaic bash baseline/scripts/run_formal_baseline.sh <sar|rgb> <n|s|m> 0 <gpu_id>
PROTOCOL=mosaic100 bash baseline/scripts/run_formal_baseline.sh <sar|rgb> <n|s|m> 0 <gpu_id>
```

## Coverage Matrix

| Protocol | Size | SAR baseline | RGB teacher | Status |
|---|---|---:|---:|---|
| `nomosaic` | n | AP 0.55127, best 0.55654 | AP 0.62737, best 0.63018 | Complete. |
| `nomosaic` | s | AP 0.62233, best 0.62897 | AP 0.64958, best 0.65768 | Complete. |
| `nomosaic` | m | AP 0.64903, best 0.65580 | AP 0.67159, best 0.67909 | Complete in synced result snapshots; local compact checkpoint copy still needs confirmation before new LADD m no-mosaic jobs. |
| `mosaic100` | n | AP 0.53836, best 0.54091 | AP 0.61345, best 0.61610 | Complete, historical mosaic mainline baseline. |
| `mosaic100` | s | AP 0.61570, best 0.61972 | AP 0.65818, best 0.66029 | Complete. |
| `mosaic100` | m | epoch 511 AP 0.63459 | epoch 512 AP 0.66811 | Running on 90 server; healthy, not yet complete. |

## Immediate Scheduling Implication

1. Do not start new m-size clean LADD mosaic100 until the running m SAR/RGB mosaic100 baselines finish or at least produce accepted checkpoints.
2. It is safe to launch clean LADD A1B n/s under mosaic100 after syncing code to the remote servers, because both n/s SAR and RGB baselines are available.
3. For no-mosaic clean LADD A1B, n/s/m baseline results exist, but m checkpoint availability should be verified on the target server before launching m.

## Smoke Status

Local smoke on 2026-06-16:

- `python3 -m py_compile` passed for baseline and LADD HBB code paths.
- `bash -n` passed for baseline and LADD launch scripts.
- `DRY_RUN=1` passed for clean LADD A1B launcher.
- `DRY_RUN=1` passed for baseline launcher under both `nomosaic` and `mosaic100`.

No checkpoint weights are included in this repository update.
