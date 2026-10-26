# JointWithinCore 工程衔接交付

本目录封装已验证的 joint-within 估计器，并连接现有 EvidenceBankController / EvidenceLifecycleModel。此前实验及论文均未修改。本实现改变了原 A/W 广义特征空间估计器，不继承其 SOTA、长流或资源结论。

## 核心接口与边界

- `JointMomentCell` 保存 257 维 `[z, anchor/kappa]` 的 `(n,sum,cross)` 与样本身份；`JointWindow` 保存两个平衡类别的 cell。身份需采用调用方统一的来源命名空间。
- `add_history_window` / `remove_history_window` 管理历史窗口；`replace_current_window` 管理独立当前窗口。重复窗口及重复身份被拒绝。两类不平衡明确拒绝。
- 历史仅贡献各 window×class 的类内散度及自由度；当前提供均值、类别对比及当前散度。沿用固定 κ、λ、MB。
- `advance_window(new_current, archive_current=...)` 在副本上显式归档或丢弃旧 current，再替换 current；通过求解后才提交。
- `forget_core_window` 仅保证保留核心统计的删除重建等价。控制器既往 audit posterior/ledger 不随之删除，capsule 不变；不宣称完整控制器从未见样本式遗忘。
- 空 current 精确回到 anchor。固定 `R` 在所有 gamma 状态下排除 sensor。原标准化空间 adaptive 系数 `C.T@w` 与 sensor 正交；不是原文 raw-Pperp 后再 PCA 的逐值等价。
- `coordinate_view.py` 给出同一解的精确白化坐标 `G=R.T@(Wzz+lambda/n MB)@R=LL.T, U=R L^-T`。坐标随历史/当前统计与正则改变，不是仅历史决定的永久坐标。

`AtomicJointLifecycle.audit_and_update` 先用一个共同 pre-core 调原控制器，再在副本更新核心，全部成功才原子提交。此接口是单写者接口，不允许外部并发修改公开对象。`predict` 返回原 EvidenceDecision，包括 forensic 分数和保守下界。

## 实际传感器

复用 `joint_sensor.npz` 中的 v,m,beta；忽略旧 κ 和 initial_w/b。按根契约提升为 1102 维 capsule，以父 sensor/base/分配表绑定生成新 ID。48 个无标签配对原图和 48 个强度校准标签保持计数。单名 bank 为 `jpeg`，原控制器参数 `.25/.2/.8` 不变。

## 实际检查结果

- 曝光 world0/seed20260918/K4：H0 评分重放最大误差 `2.415845e-13`；H32 为 `8.526513e-14`。
- 整窗删除与重建、删尽历史与 H0、空 current 与 anchor 的评分误差均为 0。
- sensor 提升评分误差 `7.105427e-15`；adaptive 原空间 sensor 内积 `2.889399e-17`；固定 R 与 KKT 权重误差 `5.724587e-17`。
- 白化坐标与直接解的最大权重误差 `<9e-17`，白化矩阵误差 `<3e-15`。
- 字节序列化、完整 system snapshot、控制状态恢复通过；8 个旧 audit support 探针的各分数误差为 0。错误 base 和被更改 scaler 明确拒绝。共享 base 仅绑定、不复制。
- 审计后核心更新失败，以及非法审计输入，均保持核心/控制器状态不变。
- 真实 48 个旧审计支持的六阶段回放全部 `HOLD/PROVISIONAL`，没有启用/暂停/恢复证据。另列构造数据单元测试得到 `BIRTH→RETIRE→REACTIVATE`，不作性能证据。

`check_assembly.py` 补充坐标和完整恢复检查；新坐标帮助函数初次执行发现一个多余右括号，修正后通过，未影响任何既有结果或模型公式。

## 状态资源

实际 H32/K4 核心：无 sensor 序列化 8,490,503 字节；固定 sensor 约束序列化 8,487,771 字节。六阶段后核心 15,564,085 字节；bank 32,536 字节（已包含 capsule，不重复加 capsule）；controller 14,971 字节。完整带校验/base64 包装的 system snapshot 为 20,816,071 字节。完整 snapshot 是替代存储形式，不再叠加组件字节。

这些数字包括身份及矩统计序列化；不含共享 encoder/base、Python 对象开销及求解临时内存。详见 RESOURCE_COUNTS.json 与 ASSEMBLY_CHECKS.json，不沿用旧资源数字。

交付：joint_within_core.py、lifecycle_bridge.py、coordinate_view.py、system_snapshot.py；检查：ENGINEERING_CHECKS.json、EXPOSED_HEAD_REPLAY.csv、WINDOW_REBUILD_CHECKS.json、COORDINATE_EQUIVALENCE.json、ASSEMBLY_CHECKS.json、EXPOSED_AUDIT_LIFECYCLE.csv。没有计算新 AUROC，也未访问 reserve。
