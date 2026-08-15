# 项目状态

更新时间：2026-08-15

## 状态定义

- `implemented`：代码与本地测试完成，但没有 GPU 实验数字。
- `smoke_tested`：GPU 10-step、Adapter 保存与重载已经通过。
- `measured`：固定协议的完整运行结束并保存原始产物。
- `human_reviewed`：人工盲审与一致率计算完成。

## 已核实的历史事实

旧 `Guided Study.ipynb` 曾在 T4 上完成 1 epoch：过滤后 Train 12,996、Test 1,437，
1,625 steps，约 156.43 分钟。50 题结果为 ROUGE-L 0.1644→0.3425、BERTScore F1
0.8387→0.8863。该结果标记为 historical，不代表当前 clean 协议。

## 当前仓库：implemented

- `legacy|clean` 双协议与固定 seed 3407。
- 空值清理、精确去重、近重复聚类、重复簇隔离和 80/10/10 split。
- 数据 revision、各 split 数量/qtype、文件 SHA-256 与泄漏审计 manifest。
- prompt-completion 训练、completion-only loss 实际 batch 审计、Validation、checkpoint 恢复。
- 训练时间、吞吐、峰值显存、依赖版本、Git commit、配置 hash 记录。
- 300 题 Base/FT 成对推理，prompt/generation hash 一致性检查。
- ROUGE、BERTScore、配对 bootstrap 95% CI、改善/退化统计与错误案例导出。
- 50 条安全自动筛查、匿名 A/B 复核表、固定 20 条第二复核样本和 Cohen's Kappa。
- PubMedQA 100 题外部能力检查和 rank 8/16 固定 pilot 消融入口。
- 一键 pipeline、薄 `cloud_runner.ipynb`、GitHub Actions。
- 本地结果：16 个 CPU 测试通过，Ruff 静态检查通过。

## 尚未完成：不能写成新实验成果

- GPU 10-step smoke 与 Adapter 重载。
- clean 协议完整 1 epoch 训练。
- 固定 300 题 Base/FT 指标与配对置信区间。
- 50 条安全输出和人工盲审。
- rank 8/16 消融与 PubMedQA 100 题实测。

当前允许写“设计并实现了可复现评测工程”；只有 GPU 产物生成并核验后，才允许写新的
模型提升、显存、耗时与吞吐数字。
