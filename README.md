# Medical LLM QLoRA：可复现的医学问答微调与安全审计

这是一个面向算法工程岗位作品集的独立项目：使用固定 revision 的 Llama-3 8B 4-bit 模型，在 MedQuAD 上进行 QLoRA 监督微调，并用同一批未参与训练的问题公平比较 Base 与 Fine-tuned 模型。

本项目是研究与工程评测，不是医疗器械，也不提供诊断或用药建议。

## 当前事实状态

- 旧课程实验已真实完成一次 T4 训练。原 Notebook 的实际输出为过滤后 Train 12,996、Test 1,437，1 epoch 共 1,625 steps，约 156.43 分钟。
- 旧 50 题结果保存在 `results/public/historical_50_summary.json`，并明确标记为历史结果。
- 新工程已固定数据/模型 revision、90/10 split、seed 3407、训练参数和 300 题评测协议。
- 新 300 题 Base/FT、bootstrap 95% CI 与 50 条安全案例仍需在 24GB GPU 上重新运行；运行前不得把历史 50 题写成新结果。

## 实验设计

数据严格复现旧实验顺序：

1. 下载 `keivalya/MedQuad-MedicalQnADataset` 的固定 revision。
2. 先用 seed 3407 做 90/10 train/test split。
3. 使用 Llama-3 chat template 格式化问答。
4. 再过滤格式化后超过 512 tokens 的样本。
5. 从固定 Test 中确定性选择 300 题；Base 与 FT 使用完全相同的问题和生成参数。

核心训练配置：

| 项目 | 值 |
|---|---:|
| Base model | `unsloth/llama-3-8b-Instruct-bnb-4bit` |
| Quantization | 4-bit |
| LoRA | r=16, alpha=16 |
| Target modules | q/k/v/o + gate/up/down projections |
| Max sequence length | 512 |
| Batch / accumulation | 2 / 4（effective 8） |
| Learning rate | 2e-4 |
| Epochs | 1 |
| Seed | 3407 |

## 目录

```text
configs/                    固定训练与评测配置
data/safety_cases.jsonl     20 急症 + 15 用药 + 15 信息不足案例
src/medical_llm/            数据、训练、推理、评测与安全审计模块
scripts/                    GPU 可执行入口
tests/                      不依赖 GPU 的确定性测试
results/public/             可公开的小型历史结果与最终摘要
```

## GPU 运行

建议 T4、A10 或其他至少 24GB 显存的 Linux/Colab 环境。不要在当前 Windows CPU 环境尝试 8B QLoRA 训练。

```bash
python -m pip install -r requirements-train.txt
python scripts/prepare_data.py
python scripts/train_qlora.py
python scripts/run_inference.py --adapter outputs/llama3_medquad/adapter
python scripts/evaluate_predictions.py --predictions outputs/predictions_300.jsonl
python scripts/run_safety_inference.py --adapter outputs/llama3_medquad/adapter
python scripts/evaluate_safety.py --predictions outputs/safety_predictions.jsonl
```

训练会保存 LoRA Adapter、Trainer log 和训练摘要，但这些大文件默认不提交 Git。公开仓库只保存代码、配置、50 条安全输入和小型结果摘要。

## 指标与边界

300 题报告 ROUGE-1/2/L、BERTScore F1 和 bootstrap 95% CI，并按逐题 ROUGE-L 差异统计变好、变差和持平。安全集只做作品集层面的规则审计：急症是否建议及时就医、是否给出确定诊断、是否出现具体剂量、是否表达不确定性。它不能被描述为临床安全评测。
