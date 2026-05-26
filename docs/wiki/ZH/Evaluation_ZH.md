# 評估與結果

[← 首頁](Home_ZH.md)

## 1. 為什麼 Accuracy 具有誤導性

資料集中 90.27% 為非危險物體。永遠預測 `False` 的模型可達到 90.27% accuracy，
但完全沒找到任何危險物體（recall = 0）。如果以 accuracy 作為主要指標，
最差的模型反而看起來很好。

本專案以下列指標作為主要評估依據：

| 指標 | 衡量內容 |
|---|---|
| **Recall** | 真正危險物體中被模型正確標記的比例 |
| **Precision** | 模型標記為危險的物體中真正危險的比例 |
| **F1** | Recall 與 Precision 的調和平均 |
| **PR-AUC** | Precision-Recall 曲線下面積；與閾值無關；適合不平衡資料 |
| **ROC-AUC** | 排序品質；對類別不平衡的敏感度低於 PR-AUC |
| **Brier Score** | 預測機率的均方誤差；越低代表校準越好 |
| **混淆矩陣** | 給定閾值下的 TP/FP/FN/TN 絕對計數 |

Accuracy 仍會回報，但不用於模型選擇或比較。

## 2. 資料切分

| 集合 | 筆數 | 用途 |
|---|---:|---|
| Train | 63,585 | 模型訓練 |
| Validation | 13,625 | 模型選擇、閾值調整、校準擬合 |
| Test | 13,626 | 最終一次性評估 |

切分依 `hazardous` 分層，確保每個子集保持 9.73% 的正類比例。
測試集保留至最終評估，不參與任何模型選擇決策。

## 3. 閾值調整

### 為何預設閾值 0.5 在此不適用

在閾值 = 0.5 時，Balanced Random Forest 的 recall = 0.6154。
這意味著 38.5% 的危險物體被遺漏。對行星防禦篩選任務而言，這個遺漏率太高。

### 閾值調整方法

在以 validation PR-AUC 選出最佳模型後，進行細粒度閾值掃描：

```python
thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)  # 91 個候選值
```

對每個閾值，`metric_row()` 在 validation set 上計算 F1、recall、precision、
PR-AUC、ROC-AUC、Brier score 與混淆矩陣。

### 選擇標準

```python
candidates.sort_values(
    ["f1", "recall", "precision", "threshold"],
    ascending=[False, False, False, True],
)
```

主要標準：F1。第一順序排名：recall。第二順序排名：precision。第三順序排名：較低閾值。
此設計偏向保守——F1 相同時，傾向更高 recall 的閾值。

### 結果

校準後的 Balanced Random Forest 所選閾值為 **0.19**。

## 4. 機率校準

### 目的

未校準的模型可能輸出過於自信或過於保守的機率。
校準後的機率更適合作為專家分類的風險分數。

### 方法

```python
CalibratedClassifierCV(best_model, method="sigmoid", cv=3)
```

Sigmoid（Platt）縮放使用訓練資料的 3-fold CV，
在原始模型分數之上擬合一個 logistic 轉換。

### 對指標的影響

| 設定 | Brier Score | PR-AUC |
|---|---:|---:|
| 原始 Random Forest | 0.0673 | 0.5634 |
| 校準後 Random Forest | 0.0600 | 0.5595 |

校準降低了 Brier score（機率品質更好），代價是 PR-AUC 略微下降。
這個取捨是可接受的，因為主要目標是風險排序品質，而非單純最大化 AUC。

## 5. 最終測試結果

在測試集上評估獲勝模型的三種設定：

| 設定 | 閾值 | Accuracy | Precision | Recall | F1 | PR-AUC | Brier | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原始 Random Forest | 0.34 | 0.8544 | 0.3838 | 0.8198 | 0.5228 | 0.5634 | 0.0673 | 0.9267 |
| **校準後 Random Forest** | **0.19** | **0.8727** | **0.4145** | **0.7474** | **0.5332** | **0.5595** | **0.0600** | **0.9269** |
| 校準後 RF，預設閾值 | 0.50 | 0.9119 | 0.5946 | 0.2986 | 0.3976 | 0.5595 | 0.0600 | 0.9269 |

建議使用閾值 = 0.19 的校準模型作為部署配置。

## 6. 混淆矩陣（校準後 RF，閾值 = 0.19）

|  | 預測 False | 預測 True |
|---|---:|---:|
| **真實 False** | 10,900 | 1,400 |
| **真實 True** | 335 | 991 |

- **True Positives（991）：** 危險物體被正確標記以供審查。
- **False Negatives（335）：** 被遺漏的危險物體——此領域最嚴重的錯誤類型。
- **False Positives（1,400）：** 非危險物體被誤判為危險——增加審查工作量。
- **True Negatives（10,900）：** 正確被排除的非危險物體。

與預設閾值 0.5 相比，此設定多找出 991 − 396 = **595 個危險物體**，
代價是多出 1,400 − 270 = **1,130 個誤報**。

## 7. 閾值對 Precision-Recall 權衡的影響

下表顯示閾值選擇如何影響篩選行為：

| 閾值 | Recall | Precision | F1 | 專家審查工作量 |
|---:|---:|---:|---:|---|
| 0.10 | ~0.89 | ~0.28 | ~0.43 | 非常高——大多數物體被標記 |
| 0.19 | 0.7474 | 0.4145 | 0.5332 | 中等——選定的設定 |
| 0.34 | 0.8198 | 0.3838 | 0.5228 | 中等（原始模型） |
| 0.50 | 0.2986 | 0.5946 | 0.3976 | 低——許多危險物體被遺漏 |

## 8. 輸出檔案

| 檔案 | 說明 |
|---|---|
| `model_metrics_validation.csv` | 所有模型在 validation set 上閾值 = 0.5 的指標 |
| `hyperparameter_tuning_results.csv` | 調整模型的 CV PR-AUC 與最佳參數 |
| `threshold_tuning_validation.csv` | 未校準模型的閾值掃描結果 |
| `threshold_tuning_validation_calibrated.csv` | 校準模型的閾值掃描結果 |
| `final_test_metrics.csv` | 三種設定在測試集上的指標 |
| `final_test_predictions.csv` | 每筆資料的機率、閾值、預測標籤 |
| `training_summary.json` | 關鍵決策與輸出路徑摘要 |
| `final_precision_recall_curve.png` | 測試集 PR 曲線 |
| `final_roc_curve.png` | 測試集 ROC 曲線 |
| `final_calibration_curve.png` | 可靠度圖（預測機率 vs 實際機率） |

[← 首頁](Home_ZH.md)