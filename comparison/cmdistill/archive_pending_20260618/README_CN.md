# CMDistill Native Reproduction Pending Archive

归档日期：2026-06-18

`native_reproduction/` 是 CMDistill 在 VEDAI/其他数据集上的原生复现与探针实验工作区。当前这条线是否继续作为论文主线或附录尚未确定，因此先从 `comparison/cmdistill/` 当前入口移入本 pending archive。

保留内容包括：

- VEDAI/DroneVehicle 数据准备说明；
- YOLOv5 native CMDistill 训练脚本；
- 2026-06-18 alignment / sync-geo / component probe 图表与日志；
- 远程队列脚本和监控脚本。

注意：

1. 这里的脚本仍保留原运行路径文本，主要用于历史追溯；如果要恢复运行，应先移回或更新路径。
2. 本目录不作为 OGSOD mosaic100 paper main-table source。
3. 本地可能存在被 `.gitignore` 忽略的大型数据或中间产物，不应提交到 GitHub。
