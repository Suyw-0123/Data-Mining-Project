# 資料探勘期末專題實作計畫書

## 1. 專題名稱

**近地天體危險性預測與可解釋風險分數分析**

本專題使用 NASA Near-Earth Objects（NEO）資料集，建立一個能預測近地天體是否被標記為 `hazardous` 的資料探勘流程。任務本質為**監督式二元分類**，但模型輸出不只包含最終類別，也會提供 `hazardous` 的模型估計機率，作為行星防禦情境下的風險排序分數。

---

## 2. 專題背景與情境

近地天體會定期接近地球，多數不具立即威脅，但仍有部分物體因尺寸、速度、軌道接近條件等因素，被標記為潛在危險。若專家需逐筆審查所有觀測物件，人工成本會很高。因此，本專題將情境設定為「行星防禦資料分析小組」，目標是設計一個資料驅動的前置篩選模型，協助專家快速排序需要優先檢查的 NEO。

此系統不應只輸出「危險」或「不危險」，而應提供模型判斷的理由。例如：哪些特徵使某個物件的危險機率提高，哪些特徵降低其風險。這能讓模型結果更容易被檢查、解釋與信任。

---

## 3. 問題定義

### 3.1 任務型態

- 學習類型：監督式學習
- 任務類型：二元分類
- 目標欄位：`hazardous`
- 正類別：`hazardous = True`
- 負類別：`hazardous = False`

### 3.2 核心預測目標

根據 NEO 的觀測特徵，預測該物體是否屬於危險天體：

```text
hazardous ∈ {True, False}
```

### 3.3 機率輸出定位

本專題不將任務改成回歸問題，而是採用**機率式二元分類**：

```text
P(hazardous = True | observed features)
```

此機率代表「在此資料集標籤定義與模型條件下，該物件被判為 hazardous 的模型估計機率」，可作為風險分數與優先級排序依據。它不能被直接解讀為真實撞擊地球的物理機率。

### 3.4 決策閾值

模型會先輸出危險機率，再透過決策閾值轉換為最終分類：

```text
if P(hazardous=True) >= threshold:
    predict hazardous
else:
    predict non-hazardous
```

由於此資料集存在明顯類別不平衡，且漏判危險樣本的成本較高，實作階段不會只依賴預設的 `0.5` 閾值，而會根據 validation set 的 F1、Recall、Precision、PR-AUC 與實務情境進行 threshold tuning。

---

## 4. 資料集概述

### 4.1 資料來源

- 資料集：NASA Nearest Earth Objects
- Kaggle：<https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects>
- 原始來源註記：NASA Open API 與 JPL CNEOS close-approach data
- 授權：CC0 Public Domain

### 4.2 本地資料快照

目前本地資料位於：

```text
data/neo.csv
```

資料規格：

- 筆數：90,836
- 欄位數：10
- 目標欄位：`hazardous`
- 缺值：目前快照中未觀察到缺值

### 4.3 欄位列表

| 欄位 | 說明 | 預計用途 |
|---|---|---|
| `id` | 物件識別碼 | 不作為模型特徵，保留於案例追蹤 |
| `name` | 物件名稱 | 不作為模型特徵，保留於案例展示 |
| `est_diameter_min` | 估計最小直徑 | 候選數值特徵 |
| `est_diameter_max` | 估計最大直徑 | 候選數值特徵 |
| `relative_velocity` | 相對速度 | 候選數值特徵 |
| `miss_distance` | 最近接近距離 | 候選數值特徵 |
| `orbiting_body` | 繞行天體 | 目前為常數欄位，預計移除 |
| `sentry_object` | 是否為 sentry object | 目前為常數欄位，預計移除 |
| `absolute_magnitude` | 絕對星等 | 候選數值特徵 |
| `hazardous` | 是否危險 | 目標欄位 |

### 4.4 目標分布

| 類別 | 筆數 | 比例 |
|---|---:|---:|
| `hazardous = False` | 81,996 | 90.27% |
| `hazardous = True` | 8,840 | 9.73% |

此分布代表資料具有明顯不平衡。模型評估不可只使用 accuracy，否則可能高估總是預測非危險的模型。

---

## 5. 研究目標

本專題的目標分為四個層次：

1. 建立可靠的 hazardous 二元分類模型。
2. 輸出 calibrated 或至少經檢查的 hazardous probability，作為風險分數。
3. 在不平衡資料下選擇合適的評估指標與決策閾值。
4. 建立可解釋流程，提供全域與局部解釋，支援專家審查。

最終成果應能回答：

- 哪些特徵最影響 hazardous 判斷？
- 哪種模型在 recall、precision、F1、PR-AUC 之間取得較好平衡？
- 如果以風險分數排序，哪些 NEO 最值得優先檢查？
- 對單一 NEO，模型為何給出高或低的危險機率？

---

## 6. 方法設計

### 6.1 整體流程

```text
資料讀取
→ 資料品質檢查
→ 探索式資料分析
→ 特徵處理與特徵工程
→ train / validation / test stratified split
→ 建立 baseline models
→ 類別不平衡處理
→ 模型比較與 threshold tuning
→ 機率校準檢查
→ 最終模型選擇
→ SHAP / feature importance 解釋
→ 圖表、表格與案例輸出
```

### 6.2 資料切分策略

為避免資料不平衡造成訓練與測試分布偏移，實作時會採用 stratified split。

建議切分：

- Train：70%
- Validation：15%
- Test：15%

用途：

- Train：模型訓練
- Validation：模型選擇、threshold tuning、calibration 選擇
- Test：最終一次性評估，不參與調參

若時間允許，可在 train set 上加入 stratified k-fold cross-validation，用於比較模型穩定性。

### 6.3 特徵處理

預計處理方式：

1. 移除識別欄位：
   - `id`
   - `name`

2. 移除低資訊常數欄位：
   - `orbiting_body`
   - `sentry_object`

3. 處理高度共線特徵：
   - `est_diameter_min` 與 `est_diameter_max` 幾乎完全共線。
   - 實作時會比較兩種設計：
     - 保留兩者，交給樹模型處理。
     - 建立 `est_diameter_mean` 或 `est_diameter_range`，降低冗餘。

4. 針對偏態數值欄位進行轉換實驗：
   - `est_diameter_min`
   - `est_diameter_max`
   - `relative_velocity`
   - `miss_distance`
   - 可測試 `log1p` 版本。

5. 針對線性模型進行尺度標準化：
   - Logistic Regression 需要搭配 StandardScaler 或 RobustScaler。
   - Random Forest、Gradient Boosting 類模型通常不需要 scaler。

### 6.4 候選特徵工程

預計實驗的衍生特徵：

| 衍生特徵 | 定義 | 動機 |
|---|---|---|
| `est_diameter_mean` | `(est_diameter_min + est_diameter_max) / 2` | 以單一尺寸估計降低共線性 |
| `est_diameter_range` | `est_diameter_max - est_diameter_min` | 表示尺寸估計不確定範圍 |
| `log_relative_velocity` | `log1p(relative_velocity)` | 降低速度偏態 |
| `log_miss_distance` | `log1p(miss_distance)` | 降低距離偏態 |
| `log_est_diameter_mean` | `log1p(est_diameter_mean)` | 降低尺寸偏態 |

特徵工程不應過度擴張。每個新特徵都需要透過 validation 表現與解釋合理性判斷是否保留。

---

## 7. 模型設計

### 7.1 Baseline 模型

Baseline 用於確認任務難度與後續模型提升是否有意義。

1. Majority Class Baseline
   - 永遠預測 `hazardous = False`
   - 用於提醒 accuracy 的局限

2. Logistic Regression
   - 可解釋性高
   - 可提供係數方向
   - 適合作為線性基準

3. Logistic Regression with Class Weight
   - 使用 `class_weight="balanced"`
   - 作為處理不平衡的簡單方法

### 7.2 主要候選模型

1. Random Forest
   - 能處理非線性關係
   - 對特徵尺度不敏感
   - 可提供 feature importance

2. Gradient Boosting / HistGradientBoosting
   - 通常在表格資料上有穩定表現
   - 可和 SHAP 解釋搭配

3. XGBoost 或 LightGBM
   - 已納入主要模型比較
   - 使用 `scale_pos_weight` 對應類別不平衡
   - 適合作為表格資料上的強基線，並與 Random Forest、HistGradientBoosting 比較

### 7.3 Hyperparameter Tuning

實作階段會對主要樹模型進行輕量化 `RandomizedSearchCV` 調參，避免完整 grid search 的計算成本過高。

調參設定：

- Cross-validation：stratified 2-fold
- Search method：RandomizedSearchCV
- Scoring：`average_precision`（PR-AUC）
- 每個模型候選組數：4
- 調參模型：
  - Random Forest
  - HistGradientBoosting
  - XGBoost
  - LightGBM

調參結果會輸出至：

```text
reports/tables/hyperparameter_tuning_results.csv
```

### 7.4 不平衡處理方法

預計比較：

| 方法 | 優點 | 風險 |
|---|---|---|
| `class_weight="balanced"` | 簡單、穩定、不改變資料分布 | 對所有模型不一定同樣有效 |
| Random Undersampling | 降低多數類壓制 | 可能丟失重要非危險樣本 |
| SMOTE | 增加少數類樣本 | 可能產生不符合物理直覺的合成資料 |
| Threshold tuning | 不改變訓練資料，直接調整決策標準 | 需要謹慎使用 validation set，避免 test leakage |

本專題的優先順序為：

1. 先建立 class weight 與 threshold tuning。
2. 再視時間比較 undersampling。
3. SMOTE 只作為延伸實驗，並需在報告中說明其對天文資料的潛在限制。

---

## 8. 評估設計

### 8.1 主要評估指標

由於 hazardous 為少數類，主要指標應聚焦正類別表現。

| 指標 | 使用目的 |
|---|---|
| Recall | 衡量危險樣本被找回的比例 |
| Precision | 衡量預測為危險的樣本中有多少是真的危險 |
| F1-score | Precision 與 Recall 的折衷 |
| PR-AUC | 適合不平衡二元分類 |
| ROC-AUC | 補充整體排序能力 |
| Confusion Matrix | 顯示 FP、FN 的實際數量 |
| Brier Score | 評估機率輸出的校準品質 |

Accuracy 會列出，但不作為主要模型選擇依據。

### 8.2 Threshold Tuning

模型輸出機率後，會在 validation set 上比較不同 threshold：

- `0.1`
- `0.2`
- `0.3`
- `0.4`
- `0.5`
- 或更細的 grid search

選擇 threshold 時會考慮：

- 若重視找出更多危險物件：偏向高 recall。
- 若重視減少專家審查負擔：提高 precision。
- 若需要平衡：使用 F1 或 F-beta score。

建議主要報告：

```text
Best threshold by validation F1
Recall-oriented threshold
Default 0.5 threshold
```

如此可以清楚展示決策閾值如何影響實務結果。

### 8.3 機率校準

若模型輸出要被解讀為風險分數，需檢查 calibration。

預計方法：

- Calibration curve
- Brier score
- CalibratedClassifierCV
  - sigmoid / Platt scaling
  - isotonic regression

若時間有限，至少應輸出 calibration curve 與 Brier score，並在報告中說明模型機率是否適合被當作可信風險分數。

---

## 9. 可解釋性設計

### 9.1 全域解釋

全域解釋回答：「整體而言，哪些特徵最影響模型判斷？」

預計輸出：

- Logistic Regression coefficient plot
- Tree-based feature importance
- SHAP summary plot
- SHAP mean absolute value ranking

重點觀察：

- `absolute_magnitude` 是否為高影響特徵。
- `relative_velocity` 是否提高 hazardous probability。
- `miss_distance` 的影響是否較弱。
- 尺寸相關特徵與星等特徵是否呈現合理關係。

### 9.2 局部解釋

局部解釋回答：「為什麼這一筆 NEO 被模型判成高風險或低風險？」

預計選擇 2 至 3 個案例：

1. True Positive：模型正確判定危險。
2. False Negative：模型漏判危險，用於分析風險。
3. False Positive 或高風險非危險樣本：用於分析模型為何誤報。

每個案例應包含：

- `id` / `name`
- 真實標籤
- 預測機率
- 使用 threshold
- 預測類別
- 前 3 至 5 個主要推高或拉低機率的特徵
- 人類可讀的解釋文字

### 9.3 Threshold 解釋

除了解釋模型機率，也要說明「為何在某 threshold 下被歸為 hazardous」。

範例格式：

```text
此物件的 hazardous probability 為 0.42。
在 threshold = 0.30 的 recall-oriented 設定下，模型會將其列為需優先檢查。
若使用 threshold = 0.50，則不會被列為 hazardous。
```

這能讓報告清楚呈現模型分數與最終決策之間的差異。

---

## 10. 實驗規劃

### 10.1 實驗 A：Baseline 與資料不平衡影響

目的：

- 證明 accuracy 不足以評估本任務。
- 建立 Logistic Regression 與 Majority Baseline。

輸出：

- Accuracy、Precision、Recall、F1、PR-AUC
- Confusion matrix
- 預設 threshold 的表現

### 10.2 實驗 B：特徵處理比較

目的：

- 比較原始特徵、log transform、尺寸衍生特徵對模型表現與解釋的影響。

比較組：

1. 原始數值特徵。
2. 移除常數欄位與 ID/name 後的精簡特徵。
3. 加入 log transform。
4. 加入 `est_diameter_mean` 等尺寸特徵。

### 10.3 實驗 C：模型比較

目的：

- 比較線性模型與樹模型在不平衡資料上的效果。

候選模型：

- Logistic Regression
- Logistic Regression with class weight
- Random Forest
- Random Forest with class weight
- HistGradientBoosting / Gradient Boosting
- XGBoost
- LightGBM
- Tuned Random Forest / HistGradientBoosting / XGBoost / LightGBM

### 10.3.1 實驗 C-2：Hyperparameter Tuning

目的：

- 比較預設參數與快速調參後的模型表現。
- 使用 PR-AUC 作為調參目標，使搜尋更符合不平衡資料情境。

輸出：

- `hyperparameter_tuning_results.csv`
- tuned model validation 指標
- 最終模型是否由 tuned model 取代的判斷依據

### 10.4 實驗 D：Threshold Tuning

目的：

- 找到比 `0.5` 更適合本情境的決策閾值。

輸出：

- threshold vs precision / recall / F1 表格
- precision-recall curve
- 最終選定 threshold 的理由

### 10.5 實驗 E：機率校準

目的：

- 檢查模型機率是否可作為風險分數。

輸出：

- calibration curve
- Brier score
- 校準前後比較

### 10.6 實驗 F：可解釋性分析

目的：

- 提供全域與局部解釋，支援報告展示。

輸出：

- feature importance
- SHAP summary plot
- SHAP bar plot
- 2 至 3 個個案解釋

---

## 11. 預期成果

### 11.1 技術成果

- 完整 EDA 結果。
- 至少一個 baseline 模型。
- 至少兩種主要模型比較。
- 不平衡處理與 threshold tuning 結果。
- hazardous probability 風險分數。
- calibration 檢查結果。
- 全域與局部解釋圖表。

### 11.2 報告成果

最終報告應包含：

- 問題背景與資料集介紹。
- 資料品質與目標分布分析。
- 方法流程圖。
- 模型比較表。
- threshold tuning 分析。
- calibration 分析。
- SHAP / feature importance 解釋。
- 代表性案例分析。
- 限制與未來改進。

### 11.3 簡報成果

簡報建議聚焦：

1. 任務為二元分類，但輸出風險機率。
2. 資料不平衡使 accuracy 不可靠。
3. 調整 threshold 後，模型如何更符合行星防禦篩選情境。
4. SHAP 如何解釋高風險或低風險判斷。
5. 模型限制：資料標籤不等於真實撞擊機率。

---

## 12. 專案檔案規劃

建議後續實作時採用以下結構：

```text
Data-Mining-Projects/
├── data/
│   └── neo.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_explainability.ipynb
├── reports/
│   ├── figures/
│   └── tables/
├── src/
│   ├── data_preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
├── project-proposal/
├── plan/
│   ├── plan_ZH.md
│   └── plan_EN.md
├── README.md
├── pyproject.toml
└── uv.lock
```

目前仍處於計畫階段，以上結構只作為實作開始後的建議，不在本階段強制建立。

### 12.1 Python 環境管理

本專題後續實作將使用 `uv` 管理 Python 環境與套件依賴，避免混用系統 Python、手動建立的 virtualenv 與 `pip` 安裝狀態。

預計原則：

- 使用 `pyproject.toml` 宣告專案依賴。
- 使用 `uv.lock` 固定可重現的套件版本。
- 使用 `uv add <package>` 新增依賴。
- 使用 `uv run <command>` 執行 Python 腳本、notebook kernel 相關命令與測試命令。
- 若執行環境限制預設 cache 寫入位置，可暫時指定 `UV_CACHE_DIR` 到可寫目錄。

預期核心依賴包含：

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `shap`
- `jupyter`
- `imbalanced-learn`（若進行 undersampling 或 SMOTE 實驗）
- `xgboost`
- `lightgbm`

---

## 13. 時程規劃

| 階段 | 工作內容 | 預期產出 |
|---|---|---|
| 第 1 階段 | 資料確認、EDA、欄位檢查 | EDA 圖表、資料品質摘要 |
| 第 2 階段 | Baseline model、資料切分、初步評估 | baseline 指標表 |
| 第 3 階段 | 特徵工程與模型比較 | 模型比較表、最佳候選模型 |
| 第 4 階段 | 不平衡處理與 threshold tuning | threshold 分析、PR curve |
| 第 5 階段 | calibration 與風險分數檢查 | calibration curve、Brier score |
| 第 6 階段 | SHAP 與個案解釋 | 全域與局部解釋圖 |
| 第 7 階段 | 報告與簡報整理 | final report、slides、圖表 |

---

## 14. 風險與限制

### 14.1 資料標籤限制

`hazardous` 是資料集中的標籤，不等於實際撞擊機率。模型學到的是資料集中 hazardous 標籤與觀測特徵之間的關係。

### 14.2 特徵不足

資料集中欄位有限，缺少完整軌道參數。若要建立更接近真實天文風險的模型，可能需要加入半長軸、離心率、軌道傾角、MOID 等更完整資訊。

### 14.3 類別不平衡

危險樣本只佔約 9.73%，模型可能偏向預測非危險。需特別關注 recall、PR-AUC 與 false negative。

### 14.4 機率解讀風險

未經校準的模型機率可能過度自信或低估風險。因此報告中必須區分：

- classification probability
- calibrated probability
- real-world physical hazard probability

本專題主要處理前兩者，不宣稱預測真實撞擊機率。

### 14.5 可解釋性限制

SHAP 能解釋模型如何使用特徵，但不等於因果解釋。若模型學到資料偏差，SHAP 也會反映該偏差。

---

## 15. 成功標準

本專題若達成以下條件，即可視為實作成功：

1. 完成可重現的資料處理與模型評估流程。
2. 至少比較 baseline、線性模型與樹模型。
3. 使用適合不平衡資料的指標，而非只看 accuracy。
4. 完成 threshold tuning，並清楚說明 chosen threshold 的理由。
5. 對最終模型輸出 hazardous probability 並檢查 calibration。
6. 提供全域特徵重要度與 2 至 3 個局部案例解釋。
7. 使用 `uv` 管理 Python 環境與依賴，保留 `pyproject.toml` 與 `uv.lock` 以支援重現。
8. 在報告中清楚說明模型能力與限制，尤其是 hazardous probability 的正確解讀。

---

## 16. 最終定位

本專題最合理的定位是：

> 一個具可解釋性的機率式二元分類系統，用於根據 NEO 觀測特徵估計其被標記為 hazardous 的機率，並依風險分數協助專家進行優先級排序。

此定位同時保留二元分類任務的嚴謹性，也讓後續的 threshold tuning、probability calibration 與 SHAP 解釋流程更符合行星防禦篩選情境。
