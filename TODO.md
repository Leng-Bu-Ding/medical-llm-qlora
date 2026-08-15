# 下一步执行清单

1. 在 Kaggle/Colab 免费 GPU 上打开 `notebooks/cloud_runner.ipynb`。
2. 30 分钟内未获得稳定 GPU 或 smoke 连续失败两次，切换 AutoDL RTX 3090 24GB；总预算
   不超过 30 元。
3. 完成 `smoke_clean`：8 条样本、10 steps、Adapter 保存/重载和 8 条成对推理。
4. 完成 `clean_main_v1`：1 epoch、300 条成对预测、95% CI 和 50 条安全输出。
5. 作者盲审全部 50 条；第二位普通复核者独立盲审固定 20 条。
6. 运行 rank 8/16 pilot 与 PubMedQA 100 题能力迁移检查。
7. 只依据核验后的 `results/public/clean_main_v1/` 更新简历与 Notion。

若 FT 没有提升，不更换 Test、不重抽样本；如实发布负面结果并分析退化原因。
