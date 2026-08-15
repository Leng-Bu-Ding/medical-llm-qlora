# Dataset Card

## MedQuAD 主任务

- 来源：`keivalya/MedQuad-MedicalQnADataset`
- 固定 revision：`5b0961fbaa6d7f9c344c5d59c29943fb900c2eca`
- 字段：`Question`、`Answer`、`qtype`
- 用途：英文医疗问答监督微调与离线参考答案相似度评测
- 限制：参考答案并不等同于临床金标准；答案风格、来源分布与真实用户问题存在偏差。

`legacy` 先 90/10 split 再按 chat-template token 长度过滤，只用于追溯旧 Notebook。
`clean` 先清理和去重，再通过近重复簇隔离进行 80/10/10 划分，是正式实验协议。
最终数量与 SHA-256 只从运行生成的 `data_manifest.json` 读取，不在运行前硬编码。

## 安全集

`data/safety_cases.jsonl` 共 50 条人工编写英文情境：20 条急症、15 条不应给具体剂量的
用药问题、15 条信息不足问题。它用于发现明显风险信号，不覆盖真实临床分布。

## PubMedQA 外部检查

- 来源：`qiaojin/PubMedQA`，`pqa_labeled`
- 固定 revision：`323752ad243ce751ba75c9222d34a3130d080030`
- 固定选择 100 条完整 prompt 不超过 512 tokens 的专家标注样本
- 指标：yes/no/maybe accuracy 与 Base/FT 配对差值

该检查只观察任务迁移和能力保持，不能解释为临床泛化。
