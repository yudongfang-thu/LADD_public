# 90 Oldsplit HBB Adaptation

日期：2026-06-23

## 作用

记录从 90 服务器旧 Sixiang split iterative 方案迁移到 DroneVehicle HBB 小风洞的实验。它不是 OBB 原版逐字复刻，而是保留旧方案方法参数、使用 DroneVehicle CCLKD baseline 训练协议的 HBB adaptation。

## 当前 Run

```text
direction: IR teacher -> RGB student
student init: RGB baseline best
teacher: IR baseline best
phase chain: A1 50 epoch -> A2 100 epoch -> C 150 epoch
```

远端路径：

```text
runs_public/dronevehicle_method_search/sub2k_seed0_fullval/oldsplit_90_hbb/ir_to_rgb/oldsplit90_hbb_cclkdproto_ir2rgb_from_rgbbase_P1_20260623_2313_a1
logs/dronevehicle_method_search/sub2k_seed0_fullval/oldsplit_90_hbb/oldsplit90_hbb_cclkdproto_ir2rgb_from_rgbbase_P1_20260623_2313_gpu0
```

2026-06-23 23:16 CST 快照：A1 epoch 9/50，`mAP50-95=0.36089`，基本保持 RGB baseline best；`reach_match_loss=0.00059`。

2026-06-24 00:58 CST 快照：

```text
A2 completed: best AP50/AP50-95 0.56322 / 0.36326, latest 0.55846 / 0.36035, late5 0.36040
C epoch 133: best AP50/AP50-95 0.55536 / 0.35518, latest 0.54902 / 0.35164, late5 0.35155
```

解释：A2 有一个超过 RGB baseline AP50-95 `0.36087` 的短程正信号，但 C 阶段回落；当前不能把旧方案视作稳定主线候选。已准备 `oldsplit_a2only_controlled` 队列，后续用 low-LR/no-warmup 协议和同结构 A2 det-only split control 重新验证。
