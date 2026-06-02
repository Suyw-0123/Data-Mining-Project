# 模型與實驗

[← 首頁](Home_ZH.md)

## 1. 設計理念

模型比較遵循三個原則：

1. **納入 dummy baseline** 以揭示類別不平衡對 accuracy 的影響。
2. **比較線性與非線性模型** 以確認決策邊界是否為線性。
3. **輕量化調整** 讓每個樹模型都有公平機會，但不造成過高計算成本。

模型選擇以 **validation PR-AUC** 為主要標準，F1 與 ROC-AUC 為次要標準。
測試集僅在最終報告時使用一次。

## 2. 類別不平衡策略

正類（`hazardous = True`）只佔資料的 9.73%。
本專案採用三種策略：

| 策略 | 應用位置 |
|---|---|
| `class_weight="balanced"` | Logistic Regression（balanced 版本） |
| `class_weight="balanced_subsample"` | Random Forest |
| `scale_pos_weight=neg/pos` | XGBoost、LightGBM |
| 在 validation set 上進行閾值調整 | 所有模型 |

欠採樣與 SMOTE 已被考慮但不採用：以 weight-based 方法即可有效處理此不平衡比例，
且合成採樣可能產生物理上不合理的 NEO 樣本。

## 3. 模型目錄

### 3.1 Majority Baseline

```python
DummyClassifier(strategy="most_frequent")
```

永遠預測多數類（`False`）。用於展示 90.27% accuracy 在此任務中毫無意義。

### 3.2 Logistic Regression（預設）

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=3000, random_state=42)),
])
```

線性決策邊界。提供校準機率的基準。
`max_iter=3000` 確保在此資料集大小下收斂。

### 3.3 Balanced Logistic Regression

與上述相同，但加上 `class_weight="balanced"`。
根據類別頻率的倒數按比例增加少數類的權重。

### 3.4 Balanced Random Forest

```python
RandomForestClassifier(
    n_estimators=120,
    min_samples_leaf=3,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42,
)
```

`balanced_subsample` 對每個 bootstrap 樣本獨立重新加權，
比全域 `balanced` 在集成方法中更穩定。

### 3.5 HistGradientBoostingClassifier

```python
HistGradientBoostingClassifier(
    learning_rate=0.08,
    max_iter=250,
    l2_regularization=0.01,
    random_state=42,
)
```

scikit-learn 的直方圖梯度提升。不原生支援 `scale_pos_weight`，
主要透過閾值調整處理類別不平衡。

### 3.6 XGBoost

```python
XGBClassifier(
    n_estimators=180, max_depth=4, learning_rate=0.06,
    subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
    reg_lambda=1.0, objective="binary:logistic",
    eval_metric="logloss", tree_method="hist",
    scale_pos_weight=neg/pos,
    random_state=42, n_jobs=-1,
)
```

`scale_pos_weight` 設定為從訓練集計算出的負類/正類比例。

### 3.7 LightGBM

```python
LGBMClassifier(
    n_estimators=180, max_depth=-1, num_leaves=31,
    learning_rate=0.06, subsample=0.9, colsample_bytree=0.9,
    min_child_samples=30, reg_lambda=1.0,
    scale_pos_weight=neg/pos,
    random_state=42, n_jobs=-1, verbose=-1,
)
```

葉子優先生長策略。`verbose=-1` 抑制訓練日誌輸出。

## 4. 超參數調整

在預設模型訓練完畢後進行調整。
使用 `RandomizedSearchCV`，配合分層 2-fold CV，以 `average_precision`（PR-AUC）為評分標準。

**設定：** 每個模型 `n_iter=4`（輕量、可重現、計算成本低）。

調整四個模型：

### Random Forest（調整版）

```python
param_distributions = {
    "n_estimators": randint(80, 220),
    "max_depth": [None, 6, 10, 14],
    "min_samples_leaf": randint(1, 8),
    "max_features": ["sqrt", "log2", None],
}
```

### HistGradientBoosting（調整版）

```python
param_distributions = {
    "learning_rate": uniform(0.03, 0.12),
    "max_iter": randint(120, 360),
    "max_leaf_nodes": randint(15, 64),
    "l2_regularization": uniform(0.0, 0.15),
    "min_samples_leaf": randint(10, 60),
}
```

### XGBoost（調整版）

```python
param_distributions = {
    "n_estimators": randint(120, 320),
    "max_depth": randint(2, 7),
    "learning_rate": uniform(0.03, 0.12),
    "subsample": uniform(0.75, 0.25),
    "colsample_bytree": uniform(0.75, 0.25),
    "min_child_weight": randint(1, 8),
    "reg_lambda": uniform(0.5, 2.0),
}
```

### LightGBM（調整版）

```python
param_distributions = {
    "n_estimators": randint(120, 320),
    "num_leaves": randint(15, 80),
    "learning_rate": uniform(0.03, 0.12),
    "subsample": uniform(0.75, 0.25),
    "colsample_bytree": uniform(0.75, 0.25),
    "min_child_samples": randint(10, 80),
    "reg_lambda": uniform(0.5, 2.0),
}
```

**調整 CV PR-AUC 結果：**

| 模型 | 最佳 CV PR-AUC |
|---|---:|
| LightGBM Tuned | 0.5284 |
| XGBoost Tuned | 0.5276 |
| Random Forest Tuned | 0.5233 |
| HistGradientBoosting Tuned | 0.5208 |

## 5. Validation 比較（閾值 = 0.5）

| 模型 | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Majority Baseline | 0.9027 | 0.0000 | 0.0000 | 0.0000 | 0.0973 | 0.5000 |
| Logistic Regression | 0.9078 | 0.6606 | 0.1086 | 0.1865 | 0.4170 | 0.8883 |
| Balanced Logistic Regression | 0.7869 | 0.3073 | 0.9487 | 0.4642 | 0.3951 | 0.8908 |
| HistGradientBoosting | 0.9163 | 0.7218 | 0.2270 | 0.3454 | 0.5591 | 0.9255 |
| HistGradientBoosting Tuned | 0.9155 | 0.7384 | 0.2044 | 0.3201 | 0.5591 | 0.9253 |
| XGBoost | 0.7918 | 0.3169 | 0.9864 | 0.4797 | 0.5411 | 0.9212 |
| XGBoost Tuned | 0.8080 | 0.3312 | 0.9540 | 0.4916 | 0.5536 | 0.9248 |
| LightGBM | 0.8037 | 0.3288 | 0.9766 | 0.4920 | 0.5647 | 0.9281 |
| LightGBM Tuned | 0.8027 | 0.3275 | 0.9751 | 0.4903 | 0.5687 | 0.9292 |
| Random Forest Tuned | 0.8094 | 0.3351 | 0.9744 | 0.4987 | 0.5548 | 0.9259 |
| **Balanced Random Forest** | **0.8979** | **0.4808** | **0.6154** | **0.5399** | **0.5880** | **0.9347** |

**勝出：`random_forest_balanced`** — PR-AUC（0.5880）、F1（0.5399）、ROC-AUC（0.9347）均最高。

XGBoost 與 LightGBM 在預設閾值下有非常高的 recall（> 0.97），
但代價是非常低的 precision（~ 0.33），產生大量誤報。
Balanced Random Forest 在所有指標間取得更好的平衡。

## 6. 模型選擇決策

`random_forest_balanced` 被選為校準與最終測試的基礎模型。
決策依據為 validation PR-AUC，並由 F1 與 ROC-AUC 排名確認。

LightGBM 與 XGBoost 的調整版在 tuning 交叉驗證中有更高的 PR-AUC，
但在實際 validation set 上並未超越未調整的 Balanced Random Forest。
這說明此資料集上的調整搜尋範圍不夠廣泛，不足以彌補差距。

## 7. 訓練與預測效率

由於所有模型共用同一套特徵與 train/test 切分，效率比較具公平性。
以下由 `neo-benchmark`（純 CPU，支援者使用 `n_jobs=-1`）量測：
訓練集 63,585 筆的中位數 fit 時間、完整 13,626 筆 test set 的批次評分時間，以及單筆評分延遲。

| 模型 | 訓練時間（秒） | 批次預測 13,626 筆（ms） | 吞吐量（筆/秒） | 單筆延遲（ms） |
|---|---:|---:|---:|---:|
| Majority Baseline | 0.001 | 0.01 | — | 0.007 |
| LightGBM | 0.39 | 5.7 | 2,400,000 | 0.77 |
| HistGradientBoosting | 0.39 | 11.9 | 1,140,000 | 1.96 |
| XGBoost | 0.57 | 3.5 | 3,850,000 | 1.18 |
| Balanced Logistic Regression | 0.72 | 0.85 | 16,100,000 | 0.44 |
| Logistic Regression | 0.77 | 0.80 | 17,100,000 | 0.46 |
| **Balanced Random Forest** | **1.29** | **59.3** | **230,000** | **38.4** |

最終選用的 Balanced Random Forest 是所有候選中成本最高者，訓練最久、評分最慢
（單筆 38 ms 主要來自 120 棵樹對單列輸入的執行緒派發開銷）。
但 NEO 危險篩選屬離線批次任務，59 ms 掃完整個 test set（約每秒 23 萬筆）
遠超專家審查所需吞吐量，因此預測延遲並非限制因素——
這使得模型選擇能由篩選品質（recall、PR-AUC、機率品質）而非速度主導。

> 數據可由 `uv run neo-benchmark` 重現，結果寫入
> `reports/tables/efficiency_benchmark.csv`（執行環境記錄於同目錄的 `_environment.json`）。
> 絕對數值會隨硬體浮動，但模型間的相對排序穩定。

[← 首頁](Home_ZH.md)