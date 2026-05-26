# 資料集與 EDA

[← 首頁](Home_ZH.md)

## 1. 資料來源

| 項目 | 內容 |
|---|---|
| 名稱 | NASA Nearest Earth Objects |
| 提供者 | Kaggle（資料集作者：Sameep Vani） |
| 上游來源 | NASA Open API + JPL CNEOS close-approach data |
| 授權 | CC0 Public Domain |
| 本地路徑 | `data/neo.csv` |

每一筆資料代表一次 NEO 的接近地球事件（close-approach record）。

## 2. 欄位說明

| 欄位 | 型別 | 用途 | 備註 |
|---|---|---|---|
| `id` | integer | 識別碼 | 不作為模型特徵；保留供案例追蹤 |
| `name` | string | 識別碼 | 不作為模型特徵；保留供案例說明 |
| `est_diameter_min` | float | 基礎特徵 | 估計最小直徑（km） |
| `est_diameter_max` | float | 基礎特徵 | 估計最大直徑（km） |
| `relative_velocity` | float | 基礎特徵 | 接近時相對速度（km/h） |
| `miss_distance` | float | 基礎特徵 | 最近接近距離（km） |
| `orbiting_body` | string | 捨棄 | 此快照中恆為 `Earth`，無變異量 |
| `sentry_object` | boolean | 捨棄 | 此快照中恆為 `False`，無變異量 |
| `absolute_magnitude` | float | 基礎特徵 | 絕對星等（H），值越小代表越亮/越大 |
| `hazardous` | boolean | 目標欄位 | `True` = 潛在危險小行星（PHA） |

**資料集規模：** 90,836 筆 × 10 欄。本地快照無缺值。

## 3. 目標欄位分布

| 類別 | 筆數 | 比例 |
|---|---:|---:|
| `hazardous = False` | 81,996 | 90.27% |
| `hazardous = True` | 8,840 | 9.73% |

不平衡比例約 1 : 9.3（正類 : 負類）。

此不平衡對建模有兩個重要影響：

1. Accuracy 是誤導性指標。永遠預測 `False` 的模型可達到 90.27% accuracy，但 recall 為 0。
2. 模型選擇與閾值調整必須以 Recall、F1、PR-AUC 為主要指標。

## 4. 數值特徵摘要

| 特徵 | Min | Q1 | Median | Q3 | Max | Mean |
|---|---:|---:|---:|---:|---:|---:|
| `est_diameter_min` | 0.000609 | 0.019 | 0.048 | 0.143 | 37.89 | 0.127 |
| `est_diameter_max` | 0.001362 | 0.043 | 0.108 | 0.321 | 84.73 | 0.285 |
| `relative_velocity` | 203 | 28,619 | 44,190 | 62,924 | 236,990 | 48,067 |
| `miss_distance` | 6,746 | 17,210,820 | 37,846,579 | 56,548,996 | 74,798,651 | 37,066,546 |
| `absolute_magnitude` | 9.23 | 21.34 | 23.70 | 25.70 | 33.20 | 23.53 |

所有數值特徵均呈右偏分布；`relative_velocity` 與 `miss_distance` 的數值跨越數個數量級，
這是進行 `log1p` 轉換的主要動機。

## 5. 與目標欄位的 Pearson 相關

| 特徵對 | Pearson r | 解讀 |
|---|---:|---|
| `hazardous` vs `absolute_magnitude` | −0.365 | 中度負相關；星等越低（越亮/越大）→ 被標記為危險的機率越高 |
| `hazardous` vs `relative_velocity` | +0.191 | 弱到中度正相關；速度越快 → 被標記為危險的機率略高 |
| `hazardous` vs `miss_distance` | +0.042 | 非常弱；接近距離單獨的線性預測能力有限 |
| `est_diameter_min` vs `est_diameter_max` | ≈ 1.000 | 近乎完美共線性；兩欄高度冗餘 |
| `absolute_magnitude` vs `est_diameter_min` | −0.560 | 符合星等與尺寸的已知物理關係 |

**重要觀察：** `miss_distance` 雖與目標的線性相關很弱，
但在非線性的 Random Forest 模型中卻是最重要的特徵。
線性相關係數低估了它的預測價值。

## 6. 資料品質驗證

`data.py` 中的 `load_neo_data()` 在讀取時執行以下驗證：

- `EXPECTED_COLUMNS` 定義的所有欄位必須存在，缺少欄位會拋出 `ValueError`。
- 所有基礎數值特徵以 `pd.to_numeric(errors="raise")` 強制轉型，遇到無法轉換的值會報錯。
- `hazardous` 欄位只能包含 `True` / `False`；任何意外值會拋出 `ValueError`。

## 7. EDA 輸出檔案

執行 `uv run neo-eda` 後產生以下檔案：

| 檔案 | 說明 |
|---|---|
| `reports/tables/dataset_summary.json` | 資料列數、欄數、類別計數、缺值數 |
| `reports/tables/numeric_summary.csv` | 各基礎數值特徵的描述統計 |
| `reports/tables/class_distribution.csv` | 類別計數與比例 |
| `reports/tables/correlation_matrix.csv` | Pearson 相關矩陣（含編碼後的目標欄） |
| `reports/tables/missing_values.csv` | 各欄的缺值數與比例 |
| `reports/tables/constant_values.csv` | 唯一值 ≤ 3 的欄位 |
| `reports/figures/target_distribution.png` | 類別不平衡長條圖 |
| `reports/figures/numeric_distributions.png` | 五個基礎數值特徵的直方圖 |
| `reports/figures/correlation_heatmap.png` | 標注數值的 Pearson 相關熱圖 |

[← 首頁](Home_ZH.md)