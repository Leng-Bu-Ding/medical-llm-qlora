# 项目状态

更新时间：2026-08-07

## 已完成

- 从旧 `Guided Study.ipynb` 核实并工程化原始数据、QLoRA 与训练配置。
- 固定基础模型 revision `fd5a4dc328319c1cfe9489eccfb9c6406bdfd469`。
- 固定 MedQuAD revision `5b0961fbaa6d7f9c344c5d59c29943fb900c2eca`。
- 实现 90/10 split、512-token 过滤、固定 300 题评测子集。
- 实现 Base/FT 同题批量推理、ROUGE/BERTScore、bootstrap 95% CI。
- 建立 50 条安全审计集：20 急症、15 用药、15 信息不足。
- 保存旧 Notebook 的真实 50 题摘要，并明确标记为历史结果。
- 本地测试：`5 passed`。

## 尚未完成

- 尚未在新工程中重新训练 LoRA Adapter。
- 尚未生成新 300 题 Base/FT 预测和置信区间。
- 尚未运行 50 条 Base/FT 安全对照。

阻塞原因是需要用户提供可用的 24GB GPU 平台授权；代码和命令已经准备好。禁止在新结果生成前把历史 50 题写成 300 题结果。
