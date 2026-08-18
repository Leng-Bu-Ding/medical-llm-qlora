# Medical LLM QLoRA：从课程 Notebook 到可审计实验

这是一个面向 LLM 应用算法岗位的独立项目：使用固定 revision 的 Llama-3 8B
4-bit 模型，在 MedQuAD 上进行 QLoRA 监督微调，并在同一批未参与训练的问题上成对比较
Base 与 Fine-tuned 模型。

本项目用于研究与工程评估，不是医疗器械，不提供诊断或用药建议，也没有经过临床验证。

## 当前结论

- 旧 `Guided Study.ipynb` 完成过一次真实 T4 训练：过滤后 Train 12,996、Test
  1,437，1 epoch 共 1,625 steps，约 156.43 分钟。
- 旧 Notebook 的 50 题历史结果保存在
  `results/public/historical_50_summary.json`。它只证明旧实验跑过，不能替代新协议结果。
- 当前仓库已实现 clean 数据协议、completion-only loss、Validation、断点恢复、300 题配对
  评测、bootstrap 95% CI、盲化安全复核、外部迁移检查、消融入口、CI 和云端 runner。
- 新 Adapter、300 题指标与安全输出尚需在 GPU 上实跑。在实跑前，任何简历文案都不得把历史
  50 题结果表述为当前工程结果。

## 为什么不是只会跑 Notebook

`legacy` 协议保留旧实验顺序，用于复现历史。`clean` 协议用于正式结论：

1. 清理空值并按规范化问答精确去重。
2. 用字符 TF-IDF 发现近重复问题，将同一重复簇放在同一个 split。
3. 按 qtype 和重复簇进行 80/10/10 Train/Validation/Test 划分，seed 固定为 3407。
4. 从 Test 确定性选取 300 题；Base/FT 共用 prompt hash 与 generation hash。
5. 训练采用 prompt-completion 数据，并在真实训练 batch 上审计：prompt/padding token 被
   `-100` mask，回答 token 才进入 loss。
6. 同时报告 ROUGE、BERTScore、长度/重复度、FT−Base 配对 bootstrap 95% CI、改善/退化
   数量与典型错误样本。
7. 对 50 条安全案例做自动筛查和匿名 A/B 人工复核；它是安全审计，不是临床验证。

每次运行都会记录数据和配置 hash、依赖版本、Git commit、运行时间、吞吐、峰值显存与
checkpoint 信息。私有大文件进入 `outputs/<run_id>/`，可公开的小摘要进入
`results/public/<run_id>/`。

## 一键 GPU 流程

推荐在 Kaggle/Colab GPU 或 AutoDL RTX 3090 24GB Linux 环境运行。Windows CPU 只执行
测试和静态检查，不适合 8B QLoRA。

```bash
python -m pip install -r requirements-train.txt
python scripts/run_pipeline.py \
  --run-id clean_main_v1 \
  --protocol clean
```

流水线依次执行：

```text
prepare → train → paired inference → evaluation → safety inference → public summary
```

先运行 8 样本、10 step 的强制 smoke：

```bash
python scripts/run_pipeline.py \
  --run-id smoke_clean \
  --protocol clean \
  --smoke
```

中断后恢复：

```bash
python scripts/run_pipeline.py \
  --run-id clean_main_v1 \
  --protocol clean \
  --resume-from-checkpoint outputs/clean_main_v1/training/checkpoints/checkpoint-800
```

薄 Notebook 入口为 `notebooks/cloud_runner.ipynb`。它只负责克隆仓库、安装依赖、检查
GPU、调用脚本和下载产物，不复制训练或评测逻辑。

## 增强实验

rank 8/16 固定 pilot 消融：

```bash
python scripts/run_ablation.py \
  --data-dir outputs/clean_main_v1/data \
  --output-dir outputs/ablation_rank
```

使用主实验 Adapter 运行 PubMedQA 100 题能力迁移检查：

```bash
python scripts/run_external_eval.py \
  --adapter outputs/clean_main_v1/training/adapter \
  --output-dir outputs/clean_main_v1
```

PubMedQA 检查的是 yes/no/maybe 迁移与能力保持，不是临床效果。

## 人工安全复核

完成安全推理后会生成：

- `review_primary.csv`：50 条匿名 A/B，全量由项目作者复核；
- `review_secondary.csv`：固定抽取 20 条，由第二位普通复核者独立复核；
- `review_key.json`：A/B 到 Base/FT 的盲化映射，复核结束前不要打开。

填写规则见 `docs/human_review_rubric.md`。完成后运行：

```bash
python scripts/summarize_human_review.py \
  --primary outputs/clean_main_v1/review_primary.csv \
  --secondary outputs/clean_main_v1/review_secondary.csv \
  --output outputs/clean_main_v1/reviewer_agreement.json
```

## 本地质量门禁

```bash
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
python -m pytest -q
python -m ruff check .
```

GitHub Actions 对每次 push/PR 执行同样的 CPU 测试和静态检查。GPU smoke 与完整实验需在
云端单独执行。

## 目录

```text
configs/                     固定的数据、训练、生成与评测配置
data/safety_cases.jsonl      20 急症 + 15 用药 + 15 信息不足案例
docs/                        复核规范、数据/模型卡和面试讲解
notebooks/cloud_runner.ipynb 薄云端入口
scripts/                     prepare/train/inference/eval/pipeline 入口
src/medical_llm/             可测试的核心实现
tests/                       不依赖 GPU 的确定性测试
results/public/              历史结果和未来可公开小型摘要
```

## 核心训练配置

| 项目 | 值 |
|---|---:|
| Base model | `unsloth/llama-3-8b-Instruct-bnb-4bit` |
| Quantization | 4-bit |
| LoRA | rank 16, alpha 16 |
| Target modules | q/k/v/o + gate/up/down projections |
| Max sequence length | 512 |
| Batch / accumulation | 2 / 4（effective batch 8） |
| Learning rate | 2e-4 |
| Epochs | 1 |
| Seed | 3407 |

项目状态、已验证事实和剩余 GPU 工作见 `PROJECT_STATUS.md`。
