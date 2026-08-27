# 下一步执行清单

更新时间：2026-08-27

## 已完成

- [x] GPU smoke test 与 Adapter 保存/重载。
- [x] `clean_main_v1`：1 epoch 训练和 300 条 Base/FT 成对评测。
- [x] 配对 bootstrap 95% CI、错误案例和 50 条自动安全筛查。
- [x] rank 8/16 固定 200-step pilot。
- [x] PubMedQA 100 条外部能力检查。
- [x] Adapter 上传和 fresh recovery 数值复现。
- [x] 公开结果摘要、README、项目状态和模型卡同步整理。

## 待完成

- [ ] 作者对 50 条安全输出进行完整盲审。
- [ ] 第二位复核者独立审查固定 20 条样本并计算 Cohen's Kappa。
- [ ] 将人工审查产物保存为 `results/public/clean_main_v1/reviewer_agreement.json`。
- [ ] 根据人工复核结果更新 `docs/model_card.md`，不把自动规则筛查写成临床安全结论。
- [ ] 决定 Hugging Face Adapter 的公开范围，并记录可访问性。
- [ ] 文档和人工复核稳定后创建版本标签。

如果后续结果为负面或不显著，不更换测试集、不重抽样本；应保留结果并分析退化原因。
