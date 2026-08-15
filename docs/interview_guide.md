# Medical QLoRA 面试讲解

## 一句话理解整个项目

用 4-bit 保存的大模型做底座，只训练很小的 LoRA 参数；关键工作不是“把训练跑起来”，而是
确保数据不泄漏、训练目标正确、Base/FT 对比公平、结论有统计置信度，并公开失败样本和安全
边界。

## 30 秒版本

我把一个课程型 MedQuAD Notebook 重构成了可复现 QLoRA 实验。项目对 Llama-3 8B 做
4-bit LoRA 微调，clean 协议会去重并把近重复问题簇隔离到同一 split；训练只对 Assistant
回答计算 loss。评测使用固定 300 题，在完全相同 prompt 和生成参数下做 Base/FT 配对比较，
报告 bootstrap 95% CI 和退化案例；还加入 50 条安全盲审、PubMedQA 外部检查、消融、CI
和云端一键运行。它是离线研究项目，不声称临床可用。

## 2 分钟版本

原始项目的问题不是没有训练，而是证据链不完整：随机行级 90/10 划分可能把同源或近重复
问题分到两边；没有 Validation；只评 50 题且没有置信区间；安全性主要靠主观观察；代码、
配置和结果混在 Notebook 中。

我保留 legacy 协议复现历史，同时建立 clean 协议。数据先做有效性和长度过滤、精确去重，
再用字符 TF-IDF 找近重复问题并按 cluster 做 group split，避免同簇跨 Train/Validation/Test。
每次运行产出 revision、样本数、qtype 分布和 SHA-256 manifest。

训练使用 conversational prompt-completion 和 `completion_only_loss`，并在真实 batch 上检查
同时存在 `-100` mask 与受监督回答 token。记录 Validation loss、checkpoint、配置 hash、Git
commit、运行时间、吞吐与峰值显存。

评测只加载一次带 Adapter 的模型，通过关闭 Adapter 得到 Base，再用同一输入得到 FT；每条
记录带 prompt hash 和 generation hash。固定 300 题报告 ROUGE/BERTScore、配对 bootstrap
差值和 95% CI，并导出最大改善与退化样本。安全部分用 50 条压力案例自动筛查，再对 Base/FT
回答随机匿名为 A/B，由我复核 50 条，第二人抽查 20 条，报告一致率与 Kappa。最后用
PubMedQA 检查微调有没有损害外部任务能力。

## 10 分钟展开顺序

1. 问题定义：参数高效地适配医疗问答，同时测量收益、退化和风险。
2. 旧 Notebook 基线：真实训练和历史指标，但证据链不足。
3. 数据：为什么先去重再 split；为什么近重复要按组隔离。
4. 模型：4-bit 降低冻结基座显存；LoRA 用低秩矩阵学习增量。
5. 训练目标：chat template 决定边界；只训练回答可避免模型学习复述用户 prompt。
6. 资源设计：batch 2 × accumulation 4 = effective 8；显存不足时 batch 1 × accumulation 8。
7. 公平比较：同一 sample、prompt、generation config；一次加载模型减少环境差异。
8. 统计：逐题计算 FT−Base，再对配对差值 bootstrap；CI 跨 0 就不宣称稳定提升。
9. 失败分析：事实错误、遗漏、重复、答非所问、过度确定、危险剂量、FT 退化。
10. 边界：自动规则与非专家盲审只能做工程安全审计，不能代替临床验证。

## 必懂技术

### QLoRA、4-bit 与 LoRA

4-bit 量化主要降低冻结基座权重的存储；LoRA 不直接更新大权重矩阵，而学习两个低秩矩阵
的乘积作为增量。rank 越高，容量和可训练参数越多，也更耗显存和计算。alpha 控制 LoRA
增量缩放，不能脱离 rank 和任务一起解释。

### Chat template 与 completion-only loss

Chat template 把角色和特殊 token 排成模型训练时认识的格式。prompt-completion 数据把用户
问题与 Assistant 答案明确分开。标签为 `-100` 的 prompt/padding token 被交叉熵忽略，只有
回答 token 产生梯度。

### 数据泄漏

精确重复只是字符串相同；近重复可能只是删词、换词或来自同一问题模板。随机逐行 split 会
高估泛化。把相似问题连接成 cluster，再按 group split，可以保证整簇只出现在一个集合。

### 配对 bootstrap

Base 和 FT 回答的是同 300 题，因此应先计算每题指标差，再对这 300 个差值有放回抽样，得到
均值差的经验分布和 95% CI。它比把两组当独立样本更符合实验设计。

## 15 个高频追问

1. **为什么用 QLoRA？** 单卡成本可控，同时保留 8B 基座能力，只训练少量 Adapter 参数。
2. **为什么不只看 loss？** loss 是优化目标，不等于生成质量、事实性或安全性。
3. **为什么需要 Validation？** 用于观察训练过程和调参；Test 只用于最终一次结论。
4. **为什么 Test 固定 300？** 在成本可控下扩大历史 50 题，并支持配对不确定性估计。
5. **为什么不是随机抽 300？** 根据 seed 与 sample ID 确定性排序，可审计且可复现。
6. **为何 Base/FT 从同一个加载模型产生？** 关闭/开启 Adapter 可保持 tokenizer、量化和环境一致。
7. **ROUGE 的问题？** 偏词面重合，多种正确表述可能低分，也不能验证医学事实。
8. **BERTScore 足够吗？** 不足；补充语义相似度，但仍不是事实性或临床正确性指标。
9. **CI 跨 0 怎么办？** 结论写不确定，分析样本和方差，不重抽 Test。
10. **为何做 rank 8/16？** 比较容量、参数、显存、速度和 Validation loss 的代价收益。
11. **为何外测 PubMedQA？** 检查迁移和能力保持，发现 MedQuAD 风格过拟合。
12. **为什么第二复核者不是医生？** 这里只标明显文本行为；临床正确性声明才必须由专家验证。
13. **微调退化怎么办？** 保留负面结果，按错误类型定位数据风格、截断、过拟合或生成设置问题。
14. **显存不够怎么办？** 降 per-device batch、增加 accumulation，保持 effective batch；必要时降低长度。
15. **这个项目离上线还差什么？** 医学专家评测、事实性基准、校准/拒答、红队、隐私合规、监控和灰度。

## 面试时不要说

- 不说“通过医疗安全验证”或“可用于临床”。
- 不把旧 50 题历史数字说成 clean 300 题结果。
- 不因平均指标提高而回避 FT 退化样本。
- 不把“用了 8B/QLoRA”本身当成创新点；价值在实验设计与可信证据链。
