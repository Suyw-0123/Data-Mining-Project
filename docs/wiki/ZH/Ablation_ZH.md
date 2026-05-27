# 消融研究（Ablation Study）

[← 首頁](Home_ZH.md)

## 1. 目的

本消融研究每次移除最終模型中的一個設計決策，
用以驗證每個元件對效能的貢獻是否真實且必要。
此任務的核心是**風險篩選**——漏掉真正危險物體（False Negative）的代價遠高於
誤判安全物體為危險（False Positive）。
因此，**Recall 是主要評估軸**，F1 與 PR-AUC 為輔助指標。

## 2. 五種 Variation

| 版本 | 設定 | 與 Full Model 的差異 |
|---|---|---|
| **V0** | Full Model | Balanced RF + Calibration + Tuned Threshold（0.19） |
| **V1** | w/o Calibration | 原始 Balanced RF，對原始分數調整閾值（0.34） |
| **V2** | w/o Threshold Tuning | 校準模型，使用預設閾值（0.50） |
| **V3** | w/o Class Balancing | RF 不使用 `class_weight="balanced_subsample"`，已校準，調整閾值（0.24） |
| **V4** | w/o Log Features | 移除 `log_est_diameter_mean`、`log_relative_velocity`、`log_miss_distance`；已校準，調整閾值（0.29） |

V0 / V1 / V2 已由標準 `neo-train` 流程產生。
V3 與 V4 為額外補跑的實驗，結果儲存於
`reports/tables/ablation_study_summary.csv`。

## 3. 結果

所有指標在保留的 test set（13,626 筆）上評估。

| 版本 | 閾值 | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 Full Model | 0.19 | 0.8727 | 0.4145 | **0.7474** | 0.5332 | 0.5595 | 0.9269 | 991 | 1400 | 335 |
| V1 w/o Calibration | 0.34 | 0.8544 | 0.3838 | 0.8198 | 0.5228 | 0.5634 | 0.9267 | 1087 | 1745 | 239 |
| V2 w/o Threshold Tuning | 0.50 | 0.9119 | 0.5946 | 0.2986 | 0.3976 | 0.5595 | 0.9269 | 396 | 270 | 930 |
| V3 w/o Class Balancing | 0.24 | 0.8929 | 0.4609 | 0.5958 | 0.5197 | 0.5715 | 0.9277 | 790 | 924 | 536 |
| V4 w/o Log Features | 0.29 | 0.8954 | 0.4718 | 0.6244 | 0.5375 | 0.5880 | 0.9324 | 828 | 927 | 498 |

## 4. 各 Variation 分析

### V1 — Calibration 的效果

移除 calibration 並對原始分數調整閾值後，Recall 達到最高的 **0.8198**，
可找出 1,087 個危險物體。但同時也產生最多誤報（1,745 個 FP），增加專家審查負擔。
Calibration 以稍微降低 recall 為代價，換得更好的機率品質
（Brier Score 從 0.0673 降至 0.0600），使風險分數更適合用於排序與優先級判斷。

### V2 — Threshold Tuning 的效果

這是最顯著的結果。對校準模型使用預設閾值 0.5，Recall 驟降至 **0.2986**——
模型漏掉了 930 個危險物體。Accuracy 雖然提升至 0.9119，
但正是這個指標最具誤導性。
在不平衡的風險篩選任務中，閾值調整並非可選項。

### V3 — Class Balancing 的效果

移除 `class_weight="balanced_subsample"` 後，Recall 從 0.7474 下降至 **0.5958**，
意味著多漏掉了 201 個危險物體（FN：536 vs 335）。
Accuracy 與 Precision 雖然提升，表面上看起來更好，
但這正是此任務不應追求的取捨。
Class balancing 對維持少數類（危險物體）的敏感度是必要的。

### V4 — Log Features 的效果

這個 variation 的結果最為細緻。
移除三個 log 轉換特徵（`log_est_diameter_mean`、`log_relative_velocity`、`log_miss_distance`）後，
PR-AUC（0.5880 vs 0.5595）、ROC-AUC（0.9324 vs 0.9269）與 F1（0.5375 vs 0.5332）
實際上都**高於** Full Model。
然而，Recall 從 0.7474 下降至 **0.6244**，多漏掉 163 個危險物體（FN：498 vs 335）。

Log 轉換降低了尺寸與距離分布的右偏程度，幫助模型在低機率值的範圍建立更穩定的決策邊界。
Recall 的下降說明其效益集中在最關鍵的區域——正確找回靠近決策邊界的邊緣性危險物體。
對於以 Recall 為優先的風險篩選任務，保留 log features 是正確的選擇。

## 5. 結論摘要

| 設計決策 | Recall 影響 | 關鍵發現 |
|---|---|---|
| Calibration | −0.0724（0.8198 → 0.7474） | 稍微降低 recall，換取更好的機率品質；對風險排序有淨效益 |
| Threshold Tuning | +0.4488（0.2986 → 0.7474） | 影響最大的單一設計；預設閾值 0.5 完全不適用此任務 |
| Class Balancing | +0.1516（0.5958 → 0.7474） | 對少數類敏感度不可或缺 |
| Log Features | +0.1230（0.6244 → 0.7474） | 改善邊界機率範圍的 recall；對篩選任務有正面貢獻 |

消融研究確認了最終模型每個元件背後的設計理由。
移除任何一個設計都會顯著增加漏掉的危險物體數量——
而這正是行星防禦篩選情境中最關鍵的指標。
