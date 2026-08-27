# 项目状态

更新时间：2026-08-27

## 状态定义

- `implemented`：代码与本地测试完成，但尚无固定协议的 GPU 实验产物。
- `smoke_tested`：GPU smoke test、Adapter 保存与重载已经通过。
- `measured`：固定协议完整运行结束，原始产物和公开摘要已经保存。
- `recovery_verified`：从持久化 Adapter 和固定代码版本重新加载后，数值复现通过。
- `human_reviewed`：人工盲审与复核者一致率计算完成。

## 当前状态：`measured + recovery_verified`

仓库已经完成 clean 主实验、自动评测、安全规则筛查、LoRA rank pilot、PubMedQA 外部
能力检查，以及一次独立 Kaggle Adapter 恢复验证。当前尚未达到 `human_reviewed`。

### Clean 主实验

- Base model：`unsloth/llama-3-8b-Instruct-bnb-4bit`
- GPU：Tesla T4
- Train / validation：11,505 / 1,453
- 训练：1 epoch，运行 10,400.17 秒，峰值显存 6.3674 GiB
- 300 条固定测试样本：BERTScore F1 `0.5859 -> 0.6823`
- ROUGE-L：`0.1903 -> 0.3335`
- 样本级结果：247 改善、53 退化、0 持平

完整数值及置信区间以 `results/public/clean_main_v1/` 中的 JSON 为准。

### 安全筛查

50 条启发式安全用例显示，Fine-tuned 模型在确定性诊断、紧急就医提示和不确定性表达等
行为上存在退化信号。该结果不是临床验证，也不能仅凭 QA 指标提升推断模型更安全。

### LoRA rank pilot

固定 200 steps pilot 中，rank 8 的 validation loss 为 `1.38335`，rank 16 为 `1.38484`；
rank 8 同时使用更少可训练参数和略低峰值显存。该结果只适用于当前 pilot 配置。

### PubMedQA 外部检查

100 条样本 accuracy 为 `0.73 -> 0.79`，配对差值为 `+0.06`，95% CI 为
`[-0.01, 0.12025]`。区间跨 0，因此只报告为能力迁移检查，不宣称统计显著提升。

### Fresh recovery 验证

从私有 Hugging Face Adapter revision `d26749288d78cab0839468dfa532a3beafcba871` 恢复模型，
重新生成 300 条成对预测。核心指标与 `clean_main_v1` 在 `1e-7` 容差内完全一致。

- 可复现 Notebook：`notebooks/fresh_recovery_validation.ipynb`
- 精简验证摘要：`results/public/recovery_validation/recovery_summary.json`

## 尚未完成

- 50 条安全输出的完整人工盲审。
- 固定 20 条第二复核样本和 Cohen's Kappa。
- 决定是否公开 Adapter，或记录第三方申请访问的流程。
- 基于最终人工复核结果更新模型卡中的人工安全结论。

在完成上述工作前，仓库可以描述为“已测量并验证 Adapter 可恢复”，但不能描述为经过人工
或临床安全验证。
