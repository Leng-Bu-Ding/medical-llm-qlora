# Medical LLM QLoRA

An engineering-oriented and reproducible LLM fine-tuning project based on **Llama-3 8B + QLoRA**, focusing on medical question answering, controlled evaluation, safety auditing, ablation study, external transfer evaluation, and experiment artifact management.

本项目从一个课程级 Notebook 实验扩展为完整的工程化训练与评测 Pipeline，覆盖：

- MedQuAD 数据清洗与去重
- Train / Validation / Test 隔离
- Llama-3 8B 4-bit QLoRA SFT
- Completion-only Loss
- Base / Fine-tuned 成对推理
- ROUGE / BERTScore 评测
- Paired Bootstrap 95% Confidence Interval
- Error Case Analysis
- Medical Safety Audit
- LoRA Rank Ablation
- PubMedQA External Evaluation
- Checkpoint Resume
- Model Artifact 持久化
- Full Reproduction / Fast Recovery

> 本项目仅用于研究、工程实验与能力评估，不是医疗器械，不提供医疗诊断或用药建议，也没有经过临床验证。

---

# 1. Project Overview

核心目标不是简单地“把 Llama-3 在医疗数据上微调一次”，而是回答以下几个问题：

1. **领域 SFT 是否真的提升医疗问答能力？**
2. **提升是否在固定测试集上稳定存在？**
3. **领域能力提升是否伴随 Safety Regression？**
4. **LoRA rank 增大是否真的带来收益？**
5. **MedQuAD 上获得的能力能否迁移到 PubMedQA？**
6. **模型训练完成后，是否能够脱离原始 GPU Session 快速恢复？**
7. **整个实验是否能够被重新运行、验证和审计？**

因此项目采用如下实验结构：

```text
Raw Dataset
    ↓
Cleaning / Deduplication
    ↓
Leakage-aware Train / Validation / Test Split
    ↓
QLoRA Supervised Fine-tuning
    ↓
Base vs Fine-tuned Paired Inference
    ↓
Automatic Evaluation
    ↓
Bootstrap Confidence Interval
    ↓
Error Analysis
    ↓
Safety Audit
    ↓
Rank Ablation
    ↓
External Transfer Evaluation
```

---

# 2. Main Experiment

主实验：

```text
run_id = clean_main_v1
protocol = clean
```

Base Model：

```text
unsloth/llama-3-8b-Instruct-bnb-4bit
```

固定 model revision：

```text
fd5a4dc328319c1cfe9489eccfb9c6406bdfd469
```

训练数据：

```text
MedQuAD
```

固定 dataset revision：

```text
5b0961fbaa6d7f9c344c5d59c29943fb900c2eca
```

随机种子：

```text
3407
```

---

# 3. Data Pipeline

正式实验使用 `clean` protocol，而不是直接随机划分原始问答数据。

数据处理流程：

```text
MedQuAD
    ↓
Remove invalid / empty records
    ↓
Exact deduplication
    ↓
Near-duplicate detection
    ↓
Duplicate-group-aware split
    ↓
Train / Validation / Test
    ↓
Fixed 300-sample evaluation subset
```

## Exact Deduplication

基于规范化后的：

```text
question + answer
```

执行精确去重。

## Near-Duplicate Detection

使用：

```text
character TF-IDF
n-gram = 3–5
nearest-neighbor search
threshold = 0.90
```

识别语义和文本形式高度相似的问题。

相同 near-duplicate group 不允许同时出现在 Train 和 Test 中，从而降低：

```text
train-test leakage
```

风险。

## Data Split

最终主实验：

| Split | Samples |
|---|---:|
| Train | 11,505 |
| Validation | 1,453 |
| Test | 独立保留 |
| Fixed evaluation subset | 300 |

300 条正式评测样本从 Test 中确定性抽取。

---

# 4. QLoRA Training

核心训练配置：

| Item | Value |
|---|---:|
| Base model | Llama-3 8B Instruct |
| Quantization | 4-bit |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| Max sequence length | 512 |
| Per-device batch size | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Learning rate | 2e-4 |
| Epoch | 1 |
| Optimizer | AdamW 8-bit |
| Scheduler | Linear |
| Seed | 3407 |

LoRA target modules：

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

---

# 5. Completion-only Loss

训练采用 instruction / response 形式。

Loss 只作用于 assistant answer tokens：

```text
User Prompt Tokens      → ignored
Padding Tokens          → ignored
Assistant Answer Tokens → supervised
```

真实训练 batch 上进行了 loss mask audit：

```text
masked prompt / padding tokens = 48
supervised answer tokens       = 443
```

保证训练目标不是简单地学习复现 prompt，而是真正优化回答部分。

---

# 6. Training Efficiency

正式 Main Run：

| Metric | Result |
|---|---:|
| Train samples | 11,505 |
| Validation samples | 1,453 |
| Tokens seen | 2.41M |
| Total parameters | 4.58B |
| Trainable parameters | 41.94M |
| Trainable ratio | ~0.92% |
| Epoch | 1 |
| Train loss | 0.9871 |
| GPU | Tesla T4 |
| Peak VRAM | 6.37 GiB |
| Runtime | 10,400 s |
| Runtime | ~2.89 h |

仅训练约：

```text
0.92%
```

的模型参数，即可完成 8B 模型的领域适配。

---

# 7. Paired Base vs Fine-tuned Evaluation

正式测试使用固定的：

```text
300 samples
```

每一个问题都同时生成：

```text
Base Prediction
Fine-tuned Prediction
```

两者使用完全相同的：

```text
question
prompt template
generation config
model revision
```

为了确保成对实验的一致性，每条样本记录：

```text
sample_id
prompt_hash
generation_hash
model_revision
```

生成采用：

```text
do_sample = false
```

从而降低随机 sampling 带来的评测噪声。

---

# 8. Main Evaluation Results

## 300-sample Base vs Fine-tuned

| Metric | Base | Fine-tuned | Delta |
|---|---:|---:|---:|
| BERTScore F1 | 0.5859 | **0.6823** | **+0.0965** |
| ROUGE-1 | 0.3206 | **0.4287** | **+0.1080** |
| ROUGE-2 | 0.0967 | **0.2486** | **+0.1519** |
| ROUGE-L | 0.1903 | **0.3335** | **+0.1432** |
| Output tokens | 92.72 | **70.71** | -22.01 |
| Repeated 4-gram rate | 0.0157 | 0.0303 | +0.0146 |

样本级比较：

```text
Improved : 247
Regressed: 53
Tied     : 0
```

即：

```text
247 / 300 = 82.3%
```

的测试样本在 Fine-tuned 模型上取得改善。

---

# 9. Statistical Significance

为了避免只比较单个均值，本项目对：

```text
FT − Base
```

进行 paired bootstrap。

配置：

```text
bootstrap samples = 2000
confidence level  = 95%
```

关键结果：

### BERTScore F1

```text
Mean Delta = +0.0965

95% CI:
[0.0838, 0.1095]
```

### ROUGE-L

```text
Mean Delta = +0.1432

95% CI:
[0.1199, 0.1671]
```

两个核心指标的置信区间均不跨 0。

因此主实验中的提升不仅体现在平均值上，也具有较稳定的统计支持。

---

# 10. Error Analysis

Pipeline 自动保存：

```text
error_cases.jsonl
```

并分别提取：

```text
largest improvements
largest regressions
```

用于分析：

- Fine-tuning 在什么问题类型上有效
- 哪些回答发生退化
- 回答长度变化
- 重复生成问题
- Domain SFT 带来的行为变化

这使项目从：

```text
"指标提高了"
```

进一步扩展到：

```text
"为什么提高 / 为什么退化"
```

---

# 11. Safety Audit

仅提高医疗 QA 指标并不能说明模型更安全。

因此另外设计了：

```text
50 medical safety cases
```

覆盖：

```text
20 urgent-care cases
15 medication-safety cases
15 insufficient-information cases
```

评测指标包括：

```text
Certain Diagnosis Rate
Timely Care Rate
Dosage Rate When Prohibited
Uncertainty Rate When Expected
```

结果：

| Metric | Base | Fine-tuned |
|---|---:|---:|
| Certain diagnosis rate | **0.08** | 0.52 |
| Timely-care rate on urgent cases | **0.50** | 0.35 |
| Uncertainty rate when expected | **0.733** | 0.433 |
| Dosage rate when prohibited | 0.00 | 0.029 |

结果表明：

> Domain SFT 明显提升了医疗 QA 能力，但同时削弱了部分安全行为。

例如 Fine-tuned 模型：

```text
更倾向直接给出确定诊断
更少表达不确定性
更少提醒紧急就医
```

因此：

```text
Domain Knowledge Improvement
≠
Safety Alignment Improvement
```

这是本项目的重要实验发现之一。

未来可进一步研究：

```text
Safety-aware SFT
Refusal Data
Uncertainty-aware Training
Preference Optimization
DPO
```

> Warning: 本 Safety Audit 是 heuristic portfolio evaluation，不是 clinical validation。

---

# 12. LoRA Rank Ablation

为了回答：

```text
为什么 Main Run 使用 rank = 16？
更大的 rank 是否真的更好？
```

额外进行了 controlled pilot ablation。

固定：

```text
Train samples      = 2000
Validation samples = 300
Training steps     = 200
```

只改变：

```text
LoRA Rank
```

比较：

```text
rank = 8
rank = 16
```

结果：

| Metric | Rank 8 | Rank 16 |
|---|---:|---:|
| Trainable parameters | **20.97M** | 41.94M |
| Validation loss | **1.38335** | 1.38484 |
| Peak VRAM | **6.15 GiB** | 6.35 GiB |
| Runtime | **1364 s** | 1400 s |
| Steps / second | **0.147** | 0.143 |

结论：

```text
Rank 16
→ trainable parameters ×2
→ slightly higher memory
→ slightly longer runtime
→ no validation-loss improvement
```

而 Rank 8：

```text
参数减少 50%
Validation Loss 基本不变
```

因此在该 pilot 条件下：

> **Rank 8 提供了更优的 parameter-efficiency trade-off。**

Main Run 的 Rank 16 仍然保留为正式 baseline，而后续训练可优先尝试 Rank 8。

---

# 13. PubMedQA External Evaluation

为了避免只在 MedQuAD 域内测试，使用：

```text
PubMedQA
```

进行了外部迁移检查。

测试样本：

```text
100
```

结果：

| Metric | Result |
|---|---:|
| Base accuracy | 0.73 |
| Fine-tuned accuracy | **0.79** |
| Accuracy delta | **+0.06** |
| Improved | 9 |
| Regressed | 3 |
| Tied | 88 |

Paired accuracy delta：

```text
+0.06
```

95% CI：

```text
[-0.01, 0.12025]
```

因此可以观察到：

```text
positive transfer signal
```

但由于置信区间跨 0：

> 不能宣称该提升具有统计显著性。

更准确的结论是：

> MedQuAD SFT 后的模型在 PubMedQA 上没有发生明显能力崩塌，并出现一定正向迁移趋势，但现有 100 样本不足以确认稳定显著增益。

> Warning: 这是 capability-transfer check，不是 clinical validation。

---

# 14. Experiment Artifact Management

一个完整的 ML 项目不等于一个 GitHub Repository。

本项目将实验资产拆分为：

```text
Code
Data
Base Model
Fine-tuned Weights
Experiment Results
Environment
```

不同资产采用不同方式持久化。

## Asset Architecture

```text
                         Medical LLM Project
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
           GitHub            Hugging Face          Compute
              │                   │                   │
         Source Code         QLoRA Adapter      Kaggle / Colab
         Configs             Model Artifact     GPU Server
         Tests
         README
         Public Results
```

具体划分：

| Asset | Storage |
|---|---|
| Source Code | GitHub |
| Config | GitHub |
| Environment Definition | GitHub |
| Public Metrics | GitHub |
| Base Model | Hugging Face |
| QLoRA Adapter | Hugging Face |
| Runtime Checkpoints | Compute / Artifact Storage |
| Large Predictions | Compute / Artifact Storage |
| Processed Dataset | Compute / Artifact Storage |

核心原则：

> **Compute Platform 不应该是唯一的 Permanent Storage。**

Kaggle / Colab / GPU Server 主要负责：

```text
Compute
```

而不是承担所有长期资产保存。

---

# 15. Model Artifact

正式 Main Run 得到的 QLoRA Adapter 已上传到 Hugging Face：

```text
Lengbuding/llama3-medquad-qlora
```

Model Hub：

https://huggingface.co/Lengbuding/llama3-medquad-qlora

主要包含：

```text
adapter_model.safetensors
adapter_config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
chat_template.jinja
```

Base Model 本身不重复保存。

最终 Fine-tuned Model 可以理解为：

```text
Llama-3 Base Model
        +
QLoRA Adapter
        =
Medical Fine-tuned Model
```

---

# 16. Artifact Portability Verification

为了验证上传到 Hugging Face 的 Adapter 没有损坏，进行了两层验证。

## Level 1: Binary-level Verification

分别计算：

```text
Original Adapter
Restored Adapter
```

核心文件 SHA256。

验证结果：

```text
adapter_model.safetensors: True
adapter_config.json: True
tokenizer.json: True
tokenizer_config.json: True
special_tokens_map.json: True
chat_template.jinja: True
```

即核心文件：

> **逐字节完全一致。**

## Level 2: Inference-level Verification

使用：

```text
相同 Base Model
相同 revision
相同 prompt
相同 tokenizer
相同 generation config
do_sample = false
```

分别加载：

```text
Original Adapter
Restored Adapter
```

进行相同样本推理。

验证：

```text
sample_id       identical
prompt_hash     identical
generation_hash identical
base_prediction identical
ft_prediction   identical
```

因此 Hugging Face 上的 Adapter 已验证可以：

```text
Download
→ Load
→ Inference
```

并复现原始模型行为。

---

# 17. Full Reproduction

**Full Reproduction** 指：

> 从源码和原始数据开始，重新走完整实验流程。

适用于：

```text
验证整个实验是否真的可复现
重新训练新的模型版本
修改数据处理策略
修改训练超参数
```

流程：

```text
GitHub Repository
        ↓
Install Environment
        ↓
Download Raw Dataset
        ↓
Data Cleaning
        ↓
Deduplication
        ↓
Train / Validation / Test Split
        ↓
QLoRA Training
        ↓
Base / FT Inference
        ↓
Evaluation
        ↓
Safety Audit
```

运行：

```bash
python -m pip install -r requirements-train.txt

python scripts/run_pipeline.py \
  --run-id clean_main_v1 \
  --protocol clean
```

Main Pipeline：

```text
prepare
→ train
→ inference
→ evaluation
→ safety
→ pipeline_manifest.json
```

当：

```text
outputs/clean_main_v1/pipeline_manifest.json
```

生成时，表示完整 Main Run 已成功结束。

---

# 18. Fast Recovery

**Fast Recovery** 与 Full Reproduction 不同。

它的目标不是重新训练，而是：

> 在新的 Kaggle / Colab / GPU Server 上快速恢复已经训练好的模型。

流程：

```text
git clone
        ↓
install dependencies
        ↓
download Base Model
        ↓
download QLoRA Adapter
        ↓
load model
        ↓
inference / evaluation
```

因此无需重新执行：

```text
3-hour QLoRA Training
```

核心思想：

```text
可复现
≠
每次都从头训练
```

正式项目应该同时支持：

```text
Full Reproduction
+
Fast Recovery
```

前者用于验证实验。

后者用于日常开发和迁移。

---

# 19. Checkpoint Recovery

训练过程中每：

```text
200 steps
```

保存 checkpoint。

例如：

```text
checkpoint-200
checkpoint-400
checkpoint-600
...
```

Checkpoint 的主要作用不是最终部署，而是：

```text
Training Recovery
```

例如：

```text
Training
→ Step 1200
→ Session disconnected
→ Resume from checkpoint-1200
```

命令：

```bash
python scripts/run_pipeline.py \
  --run-id clean_main_v1 \
  --protocol clean \
  --resume-from-checkpoint \
  outputs/clean_main_v1/training/checkpoints/checkpoint-1200
```

因此：

```text
Checkpoint = Recovery Artifact
Adapter    = Final Model Artifact
```

二者用途不同。

---

# 20. Public Experiment Results

为了避免把大型模型和 checkpoint 放进 GitHub，只把小型实验结果公开保存。

Main Run：

```text
results/public/clean_main_v1/
```

包含：

```text
data_manifest.json
training_summary.json
evaluation_summary.json
error_cases.jsonl
safety_summary.json
pipeline_manifest.json
```

Rank Ablation：

```text
results/public/ablation_rank/
└── ablation_summary.json
```

PubMedQA：

```text
results/public/pubmedqa_external/
└── pubmedqa_summary.json
```

因此 GitHub 保存的是：

```text
Code
+
Config
+
Experiment Evidence
```

而不是几十 GB 的 Runtime Artifacts。

---

# 21. Reproducibility Metadata

主实验自动记录：

```text
Git commit
Config SHA256
Dataset revision
Base model revision
Python version
PyTorch version
Transformers version
TRL version
PEFT version
Unsloth version
GPU
Peak VRAM
Training runtime
Throughput
```

本次正式环境：

```text
Python       3.12.13
PyTorch      2.10.0+cu128
Transformers 4.57.6
TRL          0.24.0
PEFT         0.19.1
Unsloth      2026.8.18
GPU          Tesla T4
```

从而避免：

```text
"不知道当时到底用了什么环境"
```

的问题。

---

# 22. Repository Structure

```text
medical-llm-qlora/
│
├── configs/
│   └── qlora_llama3_8b.yaml
│
├── data/
│   └── safety_cases.jsonl
│
├── docs/
│
├── notebooks/
│   └── cloud_runner.ipynb
│
├── scripts/
│   ├── prepare_data.py
│   ├── train_qlora.py
│   ├── run_inference.py
│   ├── evaluate_predictions.py
│   ├── run_safety_inference.py
│   ├── evaluate_safety.py
│   ├── run_pipeline.py
│   ├── run_ablation.py
│   └── run_external_eval.py
│
├── src/
│   └── medical_llm/
│
├── tests/
│
├── results/
│   └── public/
│       ├── clean_main_v1/
│       ├── ablation_rank/
│       └── pubmedqa_external/
│
├── requirements-train.txt
├── requirements-dev.txt
└── README.md
```

---

# 23. Running the Experiments

## Smoke Test

```bash
python scripts/run_pipeline.py \
  --run-id smoke_clean \
  --protocol clean \
  --smoke
```

用于：

```text
验证环境
验证数据流程
验证模型加载
验证训练
验证 inference
验证 evaluation
```

而不会先消耗数小时运行完整实验。

---

## Main Run

```bash
python scripts/run_pipeline.py \
  --run-id clean_main_v1 \
  --protocol clean
```

---

## Rank Ablation

```bash
python scripts/run_ablation.py \
  --data-dir outputs/clean_main_v1/data \
  --output-dir outputs/ablation_rank
```

---

## PubMedQA External Evaluation

```bash
python scripts/run_external_eval.py \
  --adapter outputs/clean_main_v1/training/adapter \
  --output-dir outputs/pubmedqa_external
```

---

# 24. Engineering Takeaways

这个项目最终得到的不只是一个 Fine-tuned Model。

主要工程结论包括：

### 1. QLoRA 可以低成本完成 8B 模型领域适配

```text
Trainable Parameters ≈ 0.92%
Peak VRAM ≈ 6.37 GiB
```

单张 Tesla T4 即可完成训练。

### 2. Domain SFT 显著提升 QA 指标

300 条固定测试集：

```text
BERTScore F1
0.5859 → 0.6823

ROUGE-L
0.1903 → 0.3335
```

### 3. 领域能力提升不等于 Safety 提升

Fine-tuned 模型表现出：

```text
higher certain-diagnosis tendency
lower uncertainty expression
lower urgent-care reminder rate
```

### 4. 更大的 LoRA Rank 不一定值得

Pilot Ablation：

```text
Rank 8
≈ Rank 16 validation loss

但参数量减少 50%
```

### 5. 外部迁移需要统计谨慎

PubMedQA：

```text
0.73 → 0.79
```

但：

```text
95% CI crosses zero
```

因此只能称：

```text
positive transfer signal
```

而不是 statistically significant improvement。

### 6. ML 项目不应该只保存代码

完整项目资产包括：

```text
Code
Data
Environment
Model Weights
Experiment Artifacts
Metrics
```

训练完成后应当能够：

```text
Train Once
→ Persist Artifacts
→ Reuse Repeatedly
```

而不是每次更换 GPU Session 都重新训练。

---

# 25. Disclaimer

This repository is an engineering and research portfolio project.

It is **not**:

- a medical device
- a clinical decision system
- a diagnostic system
- a medication recommendation system

The reported safety evaluation is a heuristic engineering audit and must not be interpreted as clinical validation.
