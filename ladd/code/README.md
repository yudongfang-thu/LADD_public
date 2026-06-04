# LADD HBB 代码快照

最后更新：2026-06-04 08:55 CST

这里是给外部排查直接阅读的当前 HBB/LADD 代码快照。它已经与本 public 包中的 `../code_versions/current_hbb/` 以及私有工作区当前主线同步。

关键点：

| 文件 | 作用 |
|---|---|
| `train_ladd_hbb.py` | LADD 与受控对比方法的统一入口 |
| `src/teacher_student_decomposition_kd_hbb/loss.py` | LADD loss、cap2 reach-rank、FGD/LD/CCLKD/HalluciDet profiles 与历史审计 profiles |
| `src/teacher_student_decomposition_kd_hbb/trainer.py` | A/B 阶段训练逻辑、BN-freeze |

已包含的稳定性相关开关：

```text
--freeze-bn-stats
--comparison-kd-profile {none,fgd,mgd,ld,crosskd,cclkd,c2kd,mmanet,hallucidet}
```

此前此目录比 `code_versions/current_hbb/` 旧，会缺少 HalluciDet-style 和 BN-freeze。2026-06-04 已同步修正。
