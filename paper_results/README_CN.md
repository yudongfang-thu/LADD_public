# Paper Results

本目录只保存论文表格的 curated source rows，不保存 raw run directory。

规则：

1. 任何 raw `results.csv` 不能直接进入论文表格。
2. 必须先通过 `tools/paper_collect_results.py` 汇总成 canonical CSV。
3. 必须通过 `tools/paper_validate_main_table.py` 校验。
4. `claim_usable=yes` 不可手动随意填写，必须满足 hard checks。
5. smoke / partial / diagnostic / archive / old protocol 默认不可进入主表。

主表 schema 见 `paper_results/main_table_schema.csv`。
