# 資料探勘期末專題 Proposal（中文）


## 選用資料集
- Kaggle： https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects
- 資料來源（作者註記）：NASA Open API 與 JPL CNEOS close-approach data
- 授權：CC0（Public Domain）

---

## 1) Scenario（10%）

每天都有近地天體（NEO）接近地球，絕大多數不具威脅，但少數在尺寸、速度與軌道條件下可能具有風險。  
我們將情境設定為「行星防禦資料分析小組」：希望建立一個資料驅動的前置篩選模型，協助專家優先檢查較可能危險的天體，降低人工負擔並提升早期預警效率。  
此外，系統必須提供**可理解的解釋**（為何模型判定某天體為危險／不危險），讓領域專家可檢核與信任模型決策。

---

## 2) Problem Definition（15%）

### 預測任務
根據 NEO 的觀測特徵，**預測其是否危險（`hazardous` = True/False）**。

### 解釋任務（Explainable AI）
針對每次預測，同時提供：
- **全域解釋**：整體資料中各特徵的重要性
- **局部解釋**：單一樣本被判定為危險／不危險的主要原因

### 問題型態
- 監督式學習（Supervised Learning）
- 二元分類（Binary Classification）

### 主要輸入特徵（規劃）
- `est_diameter_min`
- `est_diameter_max`
- `relative_velocity`
- `miss_distance`
- `absolute_magnitude`
- （以及後續特徵工程）

### 目標欄位
- `hazardous`

### 為何符合資料探勘題目
- 任務明確且可預測
- 可量化評估（precision/recall/F1/ROC-AUC）
- 具實務價值（天體風險分流與優先級排序）
- 不只輸出分類結果，也提供 Explainable AI 的可解釋決策依據

---

## 3) Dataset Observation（40%）

### 3.1 資料集規格檢查
- 本資料集共有 **90,836 筆、10 欄** 

欄位如下：
`id`, `name`, `est_diameter_min`, `est_diameter_max`, `relative_velocity`, `miss_distance`, `orbiting_body`, `sentry_object`, `absolute_magnitude`, `hazardous`

### 3.2 基本統計

#### 資料品質
- 10 個欄位缺值皆為 **0**
- `orbiting_body`：全部為 `Earth`（單一值欄位）
- `sentry_object`：全部為 `False`（單一值欄位）

#### 目標欄位分布
- `hazardous = True`：**8,840**（9.73%）
- `hazardous = False`：**81,996**（90.27%）
- 類別比例約 **1 : 9.27**（明顯不平衡）

#### 數值欄位摘要（本機統計）

方法說明：以下統計皆來自同一份 `neo.csv` 快照。文中相關係數採 Pearson correlation；與 `hazardous` 的相關係數計算時，標籤編碼為 True=1、False=0。

| 欄位 | Min | Q1 | Median | Q3 | Max | Mean |
|---|---:|---:|---:|---:|---:|---:|
| est_diameter_min | 0.000609 | 0.019256 | 0.048368 | 0.143402 | 37.892650 | 0.127432 |
| est_diameter_max | 0.001362 | 0.043057 | 0.108153 | 0.320656 | 84.730541 | 0.284947 |
| relative_velocity | 203.35 | 28617.55 | 44190.11 | 62923.54 | 236990.13 | 48066.92 |
| miss_distance | 6745.53 | 17210647.11 | 37845843.44 | 56548383.98 | 74798651.45 | 37066546.03 |
| absolute_magnitude | 9.23 | 21.34 | 23.70 | 25.70 | 33.20 | 23.53 |

#### Pearson 相關係數摘要（支援 3.3 觀察結論）

| 變數組合 | Pearson r | 解讀 |
|---|---:|---|
| hazardous vs absolute_magnitude | -0.365 | 中度負相關；絕對星等越低（通常越亮/尺寸傾向越大），危險機率越高 |
| hazardous vs relative_velocity | 0.191 | 弱到中度正相關；相對速度越高，越可能被判為危險 |
| hazardous vs miss_distance | 0.042 | 極弱正相關；僅靠 miss_distance 的判別力有限 |
| est_diameter_min vs est_diameter_max | 1.000 | 幾乎完全共線（高度冗餘） |
| absolute_magnitude vs est_diameter_min | -0.560 | 中度負相關；符合星等與尺寸的物理關係 |
| absolute_magnitude vs est_diameter_max | -0.560 | 中度負相關；符合星等與尺寸的物理關係 |

### 3.3 性質、規律與有趣觀察

1. **高度冗餘**：`est_diameter_min` 與 `est_diameter_max` 幾乎完全共線（相關係數約 1.0）。
2. **物理一致性**：`absolute_magnitude` 與直徑呈中度負相關（約 -0.56）。
3. **類別不平衡明顯**：危險樣本僅約 9.7%，單看 accuracy 容易誤判模型好壞。
4. **低資訊欄位存在**：`orbiting_body`、`sentry_object` 在本資料快照中無變異。
5. **危險性訊號非單一因素**：與 `hazardous` 的關聯度以 `absolute_magnitude`（約 -0.365）與 `relative_velocity`（約 0.191）較高，`miss_distance`（約 0.042）較弱，表示距離本身不足以判定風險。
6. **可解釋性需求明確**：此任務具安全性背景，預測結果需附上特徵層級解釋，才能支持實務判讀與審查。

---

## 4) Challenges（25%）

### 挑戰一：類別不平衡
**困難點：** 模型容易偏向預測「不危險」，造成危險樣本召回率不足。

### 挑戰二：特徵冗餘與低資訊欄位
**困難點：** 共線與常數欄位可能降低模型穩定性與可解釋性，增加過擬合風險。

### 挑戰三：數值範圍差異大且分布偏態
**困難點：** 速度與距離跨越範圍大，對部分模型與決策閾值設定不友善。

### 挑戰四：預測可解釋性與可信度
**困難點：** 在安全相關情境中，只有高準確率不足；必須提供穩定且可理解的理由，並確認解釋與領域直覺一致（例如尺寸、速度對風險的影響）。

---

## 5) To-dos（10%）

### 對應挑戰一（不平衡）的待辦
1. 使用 **stratified split + stratified k-fold**。
2. 比較 **class weight** 與 **重抽樣方法**（如 RandomUnderSampler / SMOTE）。
3. 以 **F1、PR-AUC、Recall 導向**指標調整決策閾值。

### 對應挑戰二（特徵問題）的待辦
1. 移除或轉換低資訊欄位（`orbiting_body`、`sentry_object`）。
2. 進行相關性篩選與精簡特徵組實驗。
3. 導入可解釋性分析（feature importance / SHAP）。

### 對應挑戰三（尺度與分布）的待辦
1. 對偏態連續變數測試 `log1p` 轉換。
2. 比較 StandardScaler 與 RobustScaler（視模型需求）。
3. 同時建立線性與樹模型基線（如 Logistic Regression、Random Forest、XGBoost/LightGBM）。

### 對應挑戰四（Explainable AI）的待辦
1. 建立 **SHAP** 解釋流程（全域特徵重要度 + 單筆樣本局部解釋）。
2. 加入可解釋基線（例如 Logistic Regression 係數）並與樹模型 SHAP 歸因比較。
3. 於簡報中設計可視化輸出：整體前幾名關鍵因子 + 2–3 個個案解釋（為何判定危險／不危險）。


---

## 參考資料

- Kaggle 資料集頁面： https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects
- NASA Open API： https://api.nasa.gov/
- JPL CNEOS Close Approaches： https://cneos.jpl.nasa.gov/ca/
