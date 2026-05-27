# NEO Hazard Risk — 專案 Wiki（中文）

本 Wiki 收錄 NEO Hazard Risk 分類專案的完整技術文件。
請使用下方導覽表前往所需主題。

## 文件導覽

| 文件 | 說明 |
|---|---|
| [資料集與 EDA](Data_ZH.md) | 資料來源、欄位說明、品質檢查、EDA 結果 |
| [流程架構](Pipeline_ZH.md) | 端對端流程設計、模組職責、資料流 |
| [特徵工程](Features_ZH.md) | 原始特徵、工程特徵、轉換策略 |
| [模型與實驗](Models_ZH.md) | 所有模型、超參數調整、比較 |
| [評估與結果](Evaluation_ZH.md) | 指標、閾值調整、校準、最終測試結果 |
| [可解釋性](Explainability_ZH.md) | Permutation importance、SHAP 全域與局部分析 |
| [消融研究](Ablation_ZH.md) | 5 種 variation 比較，驗證每個設計決策的效果 |
| [程式碼參考](Code_ZH.md) | 逐模組原始碼說明 |

## English Version

See [Home\_EN.md](../EN/Home_EN.md) for the English navigation index.

## 一段話摘要

本專案建立一個具可解釋性的二元分類系統，用於預測 NASA 近地天體（NEO）是否被標記為 hazardous。
流程涵蓋資料載入與驗證、EDA、特徵工程、模型訓練與超參數調整、閾值選擇、機率校準，
以及基於 SHAP 的可解釋性分析。
最終模型為經校準的 Balanced Random Forest，在 test set 上以 threshold = 0.19
達到 F1 = 0.5332、Recall = 0.7474。
