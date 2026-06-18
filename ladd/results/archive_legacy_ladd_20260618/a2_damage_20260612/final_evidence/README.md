# LADD A2 Damage Evidence 20260612

This directory is a lightweight evidence snapshot for the A2/B damage diagnostics.

Included:

- `runs/*/results.csv`
- `runs/*/args.yaml`
- `runs/*/ladd_diagnostics.csv`
- `chain_logs/*/manifest.txt`
- `chain_logs/*/chain.log`
- compact `master_extract.log` files
- compact outer log extracts
- summary CSV files

Excluded:

- checkpoint weights
- `.pt` / `.pth`
- TensorBoard event files
- wandb directories
- complete run directories
- full large logs

Main analysis:

- `docs/experiments/LADD_A2_DAMAGE_ANALYSIS_20260612_CN.md`
- `docs/experiments/ladd_a2_damage_summary_20260612.csv`

Important caveat:

- `m_A2_lr3e4_retry2_A2_incomplete` stopped at A2 epoch 40 because saving `weights/last.pt` hit `Disk quota exceeded`. Its partial A2 curve is included and marked incomplete.
