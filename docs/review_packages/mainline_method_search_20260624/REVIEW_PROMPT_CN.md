# 给高级模型的审阅提示

请从 GitHub 仓库直接审阅：

- 仓库：`https://github.com/yudongfang-thu/LADD_public`
- 本审阅包路径：`docs/review_packages/mainline_method_search_20260624/`
- 审阅入口：`docs/review_packages/mainline_method_search_20260624/README_CN.md`
- 当前 LADD HBB 主线实现位置：`ladd/code/src/teacher_student_decomposition_kd_hbb/`
- 当前训练入口：`ladd/code/train_ladd_hbb.py`
- 审阅包内的源码快照：`docs/review_packages/mainline_method_search_20260624/code_refs/`

你正在审阅一个跨模态目标检测蒸馏项目 LADD 的主线方法搜索证据包。目标是帮助研究者找出一个稳定的 YOLO-init 主线方法，而不是包装已有结论。

请完成以下分析：

1. 判断目前所有 YOLO-init 候选中，哪些方向最可能发展成稳定主线，哪些应该停止。
2. 解释为什么当前 dynamic 系列多数只有小正增益，而没有 AutoDL 早期看到的 +1 到 +2 AP50-95 point 级别提升。
3. 基于 `code_refs/` 中的模型和 loss 设计，指出可能导致正增益被压低、训练不稳定或 teacher 信息未有效传递的机制问题。
4. 区分 dataset/protocol/source cache/implementation/optimization 五类原因，不要把 OGSOD 与 DroneVehicle 的现象混为一谈。
5. 给出下一批最小可解释改动，优先围绕 LADD/shared-private/reachability/dynamic/ProbeA，不要跳到无关 KD。
6. 明确每个建议应如何设置 same-pipeline det-only control，以及 100 epoch 早筛和 e800 最终判据。

请优先使用这些文件：

- `docs/review_packages/mainline_method_search_20260624/README_CN.md`
- `docs/review_packages/mainline_method_search_20260624/context/OGSOD_YOLOINIT_MAINLINE_SEARCH_20260624_CN.md`
- `docs/review_packages/mainline_method_search_20260624/tables/ogsod_3090_candidate_deltas.csv`
- `docs/review_packages/mainline_method_search_20260624/tables/all_results_summary.csv`
- `docs/review_packages/mainline_method_search_20260624/code_refs/model.py`
- `docs/review_packages/mainline_method_search_20260624/code_refs/loss.py`
- `docs/review_packages/mainline_method_search_20260624/code_refs/trainer.py`
- `docs/review_packages/mainline_method_search_20260624/evidence/remote_3090/tar_snapshot/`
- `docs/review_packages/mainline_method_search_20260624/evidence/local/ogsod_yoloinit_curves_20260624/`
- `docs/review_packages/mainline_method_search_20260624/evidence/local/dronevehicle_method_search_20260623/`

如果需要对照当前工作区中的真实实现，而不是审阅包快照，请再查看：

- `ladd/code/src/teacher_student_decomposition_kd_hbb/model.py`
- `ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py`
- `ladd/code/src/teacher_student_decomposition_kd_hbb/trainer.py`
- `ladd/code/src/teacher_student_decomposition_kd_hbb/base_hbb.py`
- `ladd/code/train_ladd_hbb.py`

注意事项：

- 主线证据只看 YOLO-init，不把 reload 作为正结果。
- 当前包中 4090 最新远端原始目录未能完全同步，需谨慎使用 4090 最新状态。
- 不要把 pre100 的 early signal 当作最终正结果。
- 只推理时使用 SAR 单模态，RGB 只允许作为训练期 teacher。
