# Model Card

## 模型与训练

Base model 为固定 revision 的 `unsloth/llama-3-8b-Instruct-bnb-4bit`。训练采用 QLoRA：
4-bit 基座、LoRA rank 16 / alpha 16、最大长度 512、effective batch 8、学习率 2e-4、
1 epoch、seed 3407。仅 Assistant completion token 进入 loss。

当前实验状态为 `measured + recovery_verified`，尚未达到 `human_reviewed`。

## 预期用途

用于展示参数高效微调、可复现实验、成对统计评测和安全审计工程能力。不得用于诊断、
处方、分诊或替代专业医疗人员。

## Clean 主实验

训练在单张 Tesla T4 上完成：11,505 条训练样本、1,453 条验证样本、1 epoch；运行
10,400.17 秒，峰值显存 6.3674 GiB。

| 指标 | Base | Fine-tuned | 配对差值 |
|---|---:|---:|---:|
| BERTScore F1 | 0.5859 | 0.6823 | +0.0965 |
| ROUGE-1 | 0.3206 | 0.4287 | +0.1080 |
| ROUGE-2 | 0.0967 | 0.2486 | +0.1519 |
| ROUGE-L | 0.1903 | 0.3335 | +0.1432 |

固定 300 条测试样本中，247 条改善、53 条退化。BERTScore F1 和 ROUGE-L 的配对
bootstrap 95% CI 均不跨 0。完整精度和评测配置见
`results/public/clean_main_v1/evaluation_summary.json`。

## 外部检查与消融

- PubMedQA 100 条 accuracy：`0.73 -> 0.79`，配对差值 95% CI 跨 0，不能宣称显著提升。
- Rank pilot：rank 8 与 rank 16 的 validation loss 接近；rank 8 参数量和峰值显存更低。

## 恢复验证

从固定 Hugging Face Adapter revision 重新加载模型后，fresh recovery 的 300 条成对评测
与原始 `clean_main_v1` 在 `1e-7` 容差内一致。验证 Notebook 和摘要分别位于：

- `notebooks/fresh_recovery_validation.ipynb`
- `results/public/recovery_validation/recovery_summary.json`

Adapter 当前为私有仓库，因此该验证证明所有者侧的持久化和恢复能力，不等同于匿名第三方
可以直接下载复现。

## 安全结果

自动规则筛查发现 Fine-tuned 模型存在安全退化信号：更容易给出确定性诊断，紧急情境下
及时建议就医的比例降低，对信息不足问题表达不确定性的比例也降低。这是启发式审计，尚未
完成人工盲审，不能解释为临床安全评估。

## 已知限制

- ROUGE/BERTScore 衡量参考答案相似性，不等于事实正确或医学安全。
- 单一数据集微调可能造成风格过拟合、重复生成和基础能力退化。
- 规则安全筛查可能漏报或误报；非专家盲审也不能代替临床评审。
- PubMedQA 样本量有限，置信区间跨 0。
- Adapter 为私有资产，限制了无授权第三方的端到端复现。
- 量化、截断长度、依赖版本和确定性生成设置都会影响结果。
