# 可解釋性

[← 首頁](Home_ZH.md)

## 1. 為何可解釋性在此重要

`hazardous` 標籤是安全相關的分類任務。接收模型預測的專家不只需要知道「它危險嗎？」，
還需要知道「模型為什麼這麼認為？」，否則他們無法審查、信任或糾正模型的判斷。

本專案使用兩種互補的解釋方法：

| 方法 | 範圍 | 回答的問題 |
|---|---|---|
| Permutation Importance | 全域 | 哪些特徵對所有預測最重要？ |
| SHAP | 全域 + 局部 | 每個特徵如何讓每筆預測的機率上升或下降？ |

## 2. Permutation Importance

### 運作方式

逐一隨機打亂每個特徵（破壞其與目標的相關性），
並在 5 次隨機重複中測量 PR-AUC 的下降程度。下降越多代表特徵越重要。

```python
permutation_importance(
    model, X_test, y_test,
    scoring="average_precision",
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)
```

### 結果（前 5 名）

| 排名 | 特徵 | Importance Mean |
|---:|---|---:|
| 1 | `miss_distance` | 0.0915 |
| 2 | `log_miss_distance` | 0.0893 |
| 3 | `absolute_magnitude` | 0.0451 |
| 4 | `log_est_diameter_mean` | 0.0443 |
| 5 | `est_diameter_min` | 0.0424 |

### 關鍵洞察

`miss_distance` 與 `log_miss_distance` 排名最高，儘管 `miss_distance` 與目標的線性相關只有
r = 0.042。這證實了 Random Forest 在利用距離的非線性交互作用——接近地球但同時大且亮的物體
更常被標記為危險，模型已學習到這個聯合條件。

### 輸出檔案

- `reports/tables/permutation_importance.csv` — 特徵 × importance_mean × importance_std
- `reports/figures/permutation_importance.png` — 前 12 個特徵的水平長條圖

## 3. SHAP

### 設定方式

使用校準模型的預測函式建立 `shap.Explainer`。
因為模型是校準包裝器（非原生樹模型），使用 80 筆訓練資料的背景樣本作為參考分布：

```python
background = X_train.sample(80, random_state=42)
explainer = shap.Explainer(predict_positive, background)
shap_values = explainer(explain_sample)  # 120 筆測試資料
```

`predict_positive` 回傳 `model.predict_proba(X)[:, 1]`，即正類機率。

### 全域 SHAP 圖表

**Bar plot：** 每個特徵的 SHAP 絕對值平均，顯示整體全域重要性。
輸出：`reports/figures/shap_global_bar.png`

**Beeswarm plot：** 每個點代表一筆樣本；水平位置表示 SHAP 值；
顏色表示特徵值（紅色 = 高，藍色 = 低）。此圖揭示每個特徵的影響方向。
輸出：`reports/figures/shap_summary_beeswarm.png`

### 全域 SHAP 解讀

從 beeswarm 圖可觀察到：

- **`miss_distance` / `log_miss_distance`：** 低值（靠近地球）會使危險機率**上升**；高值使其下降。
- **`absolute_magnitude`：** 低值（更亮/更大的物體）會使機率**上升**。
- **`log_est_diameter_mean` / `est_diameter_min`：** 高值（更大的物體）會使機率**上升**。
- **`relative_velocity`：** 中等正向影響。

這些模式符合物理直覺：體積較大、較亮且接近地球的物體更容易被標記為潛在危險。

## 4. 局部案例研究

`selected_cases()` 從測試集中選取三個案例：

| 案例類型 | 選擇規則 |
|---|---|
| True Positive | 正確預測為危險的物體中，危險機率最高者 |
| False Negative | 被遺漏的危險物體中，危險機率最高者 |
| False Positive | 誤判為危險的非危險物體中，危險機率最高者 |

若無 False Positive，改用危險機率最高的 True Negative。

### 案例 1：True Positive（正確命中）

| 屬性 | 值 |
|---|---|
| `id` | 3774091 |
| `name` | (2017 HP3) |
| 真實標籤 | hazardous |
| 預測機率 | 0.8957 |
| 閾值 | 0.19 |
| 預測標籤 | hazardous ✓ |

SHAP 顯示 `log_miss_distance`、`miss_distance`、`est_diameter_min`、
`absolute_magnitude` 與 `log_est_diameter_mean` 都將機率往**上**推。
五個特徵方向一致：該物體接近地球、miss distance 相對小，且體積/亮度較高。

### 案例 2：False Negative（漏報）

| 屬性 | 值 |
|---|---|
| `id` | 3713941 |
| `name` | (2015 EO61) |
| 真實標籤 | hazardous |
| 預測機率 | 0.1897 |
| 閾值 | 0.19 |
| 預測標籤 | non-hazardous ✗ |

預測機率（0.1897）略低於閾值（0.19）。這是邊界案例。
SHAP 顯示尺寸相關特徵往上推，但 `log_miss_distance` 往下拉——
模型認為此物體的距離相對較遠。閾值的微小改變即可翻轉預測。

### 案例 3：False Positive（誤報）

| 屬性 | 值 |
|---|---|
| `id` | 3566975 |
| `name` | (2011 KO17) |
| 真實標籤 | non-hazardous |
| 預測機率 | 0.8954 |
| 閾值 | 0.19 |
| 預測標籤 | hazardous ✗ |

儘管真實標籤為非危險，模型仍賦予高風險分數。
SHAP 顯示距離、星等與尺寸特徵都強力往上推。
從篩選角度看，此誤報增加了審查工作量，但比漏掉真正危險物體代價小。

### 局部 SHAP 輸出欄位（`shap_local_case_contributions.csv`）

| 欄位 | 說明 |
|---|---|
| `source_index` | 在測試集 DataFrame 中的列索引 |
| `rank` | 以 SHAP 絕對值排名（1 = 最有影響力） |
| `feature` | 特徵名稱 |
| `feature_value` | 此物體的實際特徵值 |
| `shap_value` | SHAP 貢獻（正值 = 推向危險方向） |
| `direction` | `pushes_probability_up` 或 `pushes_probability_down` |

## 5. 局限性

1. SHAP 解釋模型行為，而非物理因果關係。若模型學習到資料集偏差，SHAP 也會反映這些偏差。
2. 背景樣本大小（80 筆）是解釋品質與計算時間的取捨。更大的背景樣本會給出更穩定的 SHAP 值。
3. 危險機率是對資料集標籤的估計，並非真實撞擊地球的物理機率。
   不應將 SHAP 貢獻解讀為使此物體物理上危險的因素。

[← 首頁](Home_ZH.md)