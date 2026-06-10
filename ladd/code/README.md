# LADD HBB 代码快照

最后更新：2026-06-10

这里是给外部排查直接阅读的当前 HBB/LADD 代码快照。它已经与本 public 包中的
`../code_versions/current_hbb/` 同步；正式实验前仍需把本版本部署到私有工作区和
目标服务器。

关键点：

| 文件 | 作用 |
|---|---|
| `train_ladd_hbb.py` | LADD 与受控对比方法的统一入口 |
| `src/teacher_student_decomposition_kd_hbb/loss.py` | LADD loss、cap2 reach-rank、FGD/LD/CCLKD/HalluciDet-style profiles |
| `src/teacher_student_decomposition_kd_hbb/trainer.py` | A/B 阶段训练逻辑、BN-freeze |

已包含的稳定性相关开关：

```text
--freeze-bn-stats
--comparison-kd-profile {none,fgd,ld,cclkd,hallucidet_style}
--fgd-alpha 1.0 --fgd-beta 0.5 --fgd-gamma 1.0 --fgd-lambda 0.0
--fgd-temperature 0.5
--ld-temperature 10.0
--ld-use-vlr 1
--cclkd-logit-weight 1.0
--cclkd-max-tokens 512
```

此前此目录比 `code_versions/current_hbb/` 旧，会缺少 HalluciDet-style 和 BN-freeze。2026-06-04 已同步修正。
