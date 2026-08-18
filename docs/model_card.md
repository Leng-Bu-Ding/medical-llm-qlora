# Model Card（实验模板）

## 模型与训练

Base 为固定 revision 的 `unsloth/llama-3-8b-Instruct-bnb-4bit`。训练使用 QLoRA：
4-bit 基座、LoRA rank 16/alpha 16，目标覆盖 attention 和 MLP projection，最大长度 512，
effective batch 8，学习率 2e-4，1 epoch，seed 3407。仅 Assistant completion token 进入
loss。

## 预期用途

用于展示参数高效微调、可复现实验、配对统计评测与安全审计工程能力。不得用于诊断、处方、
分诊或替代专业医疗人员。

## 评测

当前 clean 实验状态：`implemented`，尚无 GPU 实测结果。完成运行后，从以下产物更新本卡：

- `data_manifest.json`：数据数量、hash 与泄漏审计；
- `training_summary.json`：显存、耗时、吞吐与 loss mask 审计；
- `evaluation_summary.json`：300 题指标与配对 95% CI；
- `safety_summary.json`、`reviewer_agreement.json`：自动筛查与盲审一致性；
- `pubmedqa_summary.json`：外部能力检查。

历史 50 题结果不得填入 clean 实验区。

## 已知风险

- ROUGE/BERTScore 衡量文本相似性，不等于事实正确或医学安全。
- 单一数据集微调可能造成风格过拟合与基础能力退化。
- 规则安全筛查有漏报和误报；非专家盲审不能代替临床评审。
- 量化、截断长度和确定性生成设置都会影响结果。
