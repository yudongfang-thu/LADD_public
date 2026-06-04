# LADD Code Versions

This directory stores the current public HBB LADD code snapshot and records how historical variants were configured.

## current_hbb

`current_hbb/` contains the latest code used for:

- LADD cap2 mainline.
- A2/B MuSGD stability settings.
- `--freeze-bn-stats` / `FREEZE_BN_STATS=1`.
- Corrected FGD/LD, CCLKD-style, and HalluciDet-style comparison profiles; removed methods remain for audit only.
- Independent FGD/LD temperatures, LD DFL-logit fail-fast checks, and CCLKD logit/token controls.

## Historical Variants

The older variants were mostly configuration changes rather than separate maintained source trees:

| Variant | How to reproduce from current code |
|---|---|
| cap2 initial mainline | `FREEZE_BN_STATS=0`, cap2 enabled, older A2/B optimizer settings in the corresponding run manifests/results. |
| A2 stability fix | A2 with `optimizer=MuSGD`, `lr0=0.001`, `warmup_epochs=0`. |
| B stability fix | B with `optimizer=MuSGD`, `lr0=0.001`, `warmup_epochs=0`. |
| B BN-freeze fix | Set `FREEZE_BN_STATS=1` or pass `--freeze-bn-stats`. |

The exact run outputs and `args.yaml` files are under `../results/90_formal_nomosaic_20260528/` and `../results/4090d_formal_nomosaic_20260528/`.
