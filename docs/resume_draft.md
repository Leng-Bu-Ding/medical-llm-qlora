# 简历文案草稿

## GPU 实验完成前可用

- 将课程型医疗问答 Notebook 重构为可复现 QLoRA 实验工程，设计 `legacy/clean` 双数据
  协议，实现空值/精确去重、近重复簇隔离、分层 80/10/10 划分及数据/配置 hash 审计。
- 基于 Llama-3 8B 4-bit + LoRA 实现 completion-only SFT、Validation、断点恢复与训练
  资源记录，并构建 Base/FT 同 prompt 同生成参数的 300 题配对评测和 bootstrap 置信区间。
- 构建 50 条医疗安全压力集、匿名 A/B 人工复核流程、PubMedQA 外部能力检查及 rank 8/16
  消融入口；以 GitHub Actions 和 16 个 CPU 测试保障核心数据与统计逻辑。

## 完整实验后替换的数字槽位

- 在 MedQuAD clean 协议上完成 1 epoch QLoRA，训练 `[N]` 条、验证 `[N]` 条，耗时
  `[X]`、峰值显存 `[Y] GiB`；在固定 300 题上将 `[主指标]` 从 `[Base]` 提升至 `[FT]`，
  配对差值 `[Delta]`，95% CI `[low, high]`。
- 对 50 条安全案例进行规则筛查与匿名人工复核，第二复核者抽查 20 条，原始一致率 `[X]`、
  Cohen's Kappa `[K]`；识别 `[n]` 条 FT 退化样本并归纳主要失败模式。

若置信区间跨 0，应写“差异不显著/结论不确定”，不要写“显著提升”。
