# LADD 消融实验计划

最后更新：2026-06-02

状态：**尚未开始**。本文档为占位，等基线完成、LADD 主线和对比实验稳定后再启动。

## 计划消融项

| 消融 | 目的 | 预期 |
|---|---|---|
| no teacher decomposition | 验证 teacher 分解必要性 | 低于主线 |
| no reachability (reach_only_no_kd) | 验证 reach 独立贡献 | reach 有一定适应能力 |
| single_proj (SimKD-style) vs split | 验证 student branch 设计 | split 略优或持平 |
| no cap2 (original rank loss) | 验证 cap2 的反坍缩效果 | cap2 几何更干净，性能代价小 |
| lambda reach sweep | 验证 reach 权重敏感度 | 1.0 是最优点 |
| A2 detection scale sweep | 验证 A2 检测监督必要性 | 1.0 必须 |

## 消融协议

所有消融使用 YOLO11n seed0，遵循 formal no-mosaic baseline 规范。单 seed 消融，不做多 seed。
