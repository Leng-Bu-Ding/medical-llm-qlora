# 下一阶段

1. 在 T4/A10/24GB GPU 环境安装 `requirements-train.txt`。
2. 执行 `prepare_data.py`，核对 Train 12,996 / Test 1,437 是否复现。
3. 执行 1 epoch QLoRA，保存 Adapter、Trainer log 和 loss curve。
4. 对固定 300 题运行 Base/FT 批量推理并计算 bootstrap 95% CI。
5. 对 50 条安全案例运行 Base/FT 对照并人工复核危险样本。
6. 将小型结果摘要加入 `results/public/`，不提交权重和数据集。
