# 特徵工程

[← 首頁](Home_ZH.md)

## 1. 捨棄的欄位

以下欄位在建立特徵之前即從模型輸入中排除：

| 欄位 | 原因 |
|---|---|
| `id` | 識別碼，無預測信號 |
| `name` | 識別碼，無預測信號 |
| `orbiting_body` | 此資料集快照中恆為 `Earth`，無變異量 |
| `sentry_object` | 此資料集快照中恆為 `False`，無變異量 |

`id` 與 `name` 會保留在獨立的 `metadata` frame 中，
供可解釋性階段進行個別案例說明使用。

## 2. 基礎數值特徵

以下五個欄位在型別強制轉換後直接作為模型輸入：

| 特徵 | 單位 | 說明 |
|---|---|---|
| `est_diameter_min` | km | 估計最小直徑 |
| `est_diameter_max` | km | 估計最大直徑 |
| `relative_velocity` | km/h | 最接近時相對地球的速度 |
| `miss_distance` | km | 最接近時的距離 |
| `absolute_magnitude` | — | 絕對星等 H；值越低代表越亮，通常越大 |

## 3. 工程化特徵

`features.py` 中的 `build_feature_frame()` 額外計算五個特徵：

| 特徵 | 定義 | 目的 |
|---|---|---|
| `est_diameter_mean` | `(est_diameter_min + est_diameter_max) / 2` | 降低共線性，提供單一代表性尺寸估計 |
| `est_diameter_range` | `est_diameter_max − est_diameter_min` | 捕捉直徑估計的不確定性範圍 |
| `log_est_diameter_mean` | `log1p(est_diameter_mean)` | 降低尺寸值的右偏分布 |
| `log_relative_velocity` | `log1p(relative_velocity)` | 降低速度值的右偏分布 |
| `log_miss_distance` | `log1p(miss_distance)` | 降低距離值的右偏分布 |

使用 `log1p` 而非 `log`，以安全處理接近零的值。
`data.py` 中的 `safe_log1p()` 輔助函式在套用轉換前會先驗證沒有負值；
若有負值會拋出 `ValueError`。

## 4. 最終特徵集

模型輸入矩陣共有 **10 個特徵**：

```
est_diameter_min
est_diameter_max
relative_velocity
miss_distance
absolute_magnitude
est_diameter_mean       ← 工程化
est_diameter_range      ← 工程化
log_est_diameter_mean   ← 工程化
log_relative_velocity   ← 工程化
log_miss_distance       ← 工程化
```

`est_diameter_min` 和 `est_diameter_max` 與 `est_diameter_mean`、`est_diameter_range`
並存於特徵集中。樹模型可以處理它們之間的共線性而不受影響，
且保留原始欄位可讓 SHAP 分別對直徑上下限給予歸因。

## 5. 特徵縮放

樹模型（Random Forest、HistGradientBoosting、XGBoost、LightGBM）不需要特徵縮放。

Logistic Regression 模型使用包在 `sklearn.pipeline.Pipeline` 內的 `StandardScaler`，
縮放器只在訓練資料上 fit，正確地套用於驗證集與測試集。

## 6. 特徵重要度預覽

訓練完成後，以 permutation importance（PR-AUC drop）排名的前五特徵為：

| 排名 | 特徵 | Importance Mean |
|---:|---|---:|
| 1 | `miss_distance` | 0.0915 |
| 2 | `log_miss_distance` | 0.0893 |
| 3 | `absolute_magnitude` | 0.0451 |
| 4 | `log_est_diameter_mean` | 0.0443 |
| 5 | `est_diameter_min` | 0.0424 |

儘管 `miss_distance` 與目標的線性相關極弱（r = 0.042），
在非線性的 Random Forest 中卻是最重要的特徵。
這也證實了 `log1p` 轉換版本提供了互補的信號，兩者都值得保留。

[← 首頁](Home_ZH.md)