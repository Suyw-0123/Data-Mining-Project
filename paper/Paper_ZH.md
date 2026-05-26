# 近地天體危險性預測與可解釋風險分數分析

## 摘要

近地天體（Near-Earth Objects, NEOs）會週期性接近地球，其中少數物體可能因尺寸、速度與接近條件而被標記為潛在危險。面對大量觀測資料，若完全依賴人工逐筆檢查，將造成專家審查負擔。因此，本研究以 NASA Nearest Earth Objects 資料集為基礎，建立一個具可解釋性的二元分類流程，用於預測 NEO 是否被標記為 `hazardous`。本研究保留二元分類任務設定，但進一步輸出 `hazardous=True` 的模型估計機率，將其作為風險分數以支援優先級排序。實驗比較 Majority Baseline、Logistic Regression、class-weighted Logistic Regression、Random Forest、HistGradientBoosting、XGBoost 與 LightGBM，並對主要樹模型進行輕量化 hyperparameter tuning。結果顯示，`random_forest_balanced` 在 validation set 上具有最佳 PR-AUC；經 sigmoid calibration 後，在 test set 上以 threshold = 0.19 達到 F1 = 0.5332、Recall = 0.7474、Precision = 0.4145、PR-AUC = 0.5595、ROC-AUC = 0.9269。可解釋性分析顯示，`miss_distance`、`absolute_magnitude` 與尺寸相關特徵是主要影響因子。本研究最後透過 permutation importance 與 SHAP 提供全域與局部解釋，說明模型如何將單一 NEO 判定為高風險或低風險。

**關鍵字：** 近地天體、二元分類、類別不平衡、風險分數、機率校準、SHAP、可解釋人工智慧

## 1. 導論

近地天體是行星防禦與天文觀測中的重要研究對象。雖然多數 NEO 不會對地球造成威脅，但少數物體若具有較大的尺寸、較高的速度或較接近地球的軌道條件，仍需要被優先關注。實務上，專家通常需要先篩選大量候選物件，再針對高風險樣本進行更深入的軌道分析與人工審查。因此，一個能夠協助排序與解釋的資料探勘模型具有實務價值。

本研究的核心問題是：根據 NEO 的觀測特徵，預測該物件是否被標記為 `hazardous`。此任務本質上是監督式二元分類。然而，若模型只輸出最終類別，將難以支援風險排序與審查資源分配。因此，本研究將模型輸出設計為兩層：第一層為 `hazardous=True` 的估計機率，第二層再透過決策閾值轉換為最終分類。此設計使模型不只回答「是否危險」，也能回答「多值得優先檢查」。

需要強調的是，本文中的 hazardous probability 並不代表真實撞擊地球的物理機率，而是模型在此資料集標籤定義下，估計某筆資料被標記為 `hazardous` 的機率。因此，本文同時納入 probability calibration 與可解釋性分析，以避免對模型分數做過度解讀。

本文貢獻如下：

1. 建立一個可重現的 NEO hazardous 二元分類流程。
2. 針對不平衡資料比較多種模型與 class-weight 策略。
3. 使用 threshold tuning 將模型輸出轉換為更符合風險篩選情境的決策規則。
4. 使用 calibration curve、Brier score、permutation importance 與 SHAP 分析模型可信度與解釋性。
5. 提供全域特徵重要度與三個局部案例解釋，支援後續報告與簡報展示。

## 2. 資料集與問題定義

### 2.1 資料來源

本研究使用 Kaggle 上的 NASA Nearest Earth Objects 資料集。該資料集作者標示其來源為 NASA Open API 與 JPL CNEOS close-approach data，授權為 CC0 Public Domain。本地實驗使用的資料檔為 `data/neo.csv`。

資料集包含 90,836 筆資料與 10 個欄位：

`id`, `name`, `est_diameter_min`, `est_diameter_max`, `relative_velocity`, `miss_distance`, `orbiting_body`, `sentry_object`, `absolute_magnitude`, `hazardous`

其中 `hazardous` 為目標欄位，其餘欄位則作為候選特徵或輔助識別資訊。

### 2.2 資料品質與目標分布

本地資料快照中未觀察到缺值。`orbiting_body` 全部為 `Earth`，`sentry_object` 全部為 `False`，兩者在此資料集中沒有變異，因此在建模階段不作為有效特徵。`id` 與 `name` 屬於識別欄位，也不作為模型輸入，但會保留於案例分析。

目標欄位分布如下：

| 類別 | 筆數 | 比例 |
|---|---:|---:|
| `hazardous = False` | 81,996 | 90.27% |
| `hazardous = True` | 8,840 | 9.73% |

此分布顯示資料具有明顯類別不平衡。若模型永遠預測 `False`，仍可得到約 90.27% accuracy，但 recall 為 0，無法找出真正危險樣本。因此，本研究不以 accuracy 作為主要模型選擇指標，而以 Recall、Precision、F1、PR-AUC 與 ROC-AUC 進行綜合評估。

![Target distribution](../reports/figures/target_distribution.png)

### 2.3 數值特徵觀察

主要數值欄位摘要如下：

| 欄位 | Min | Q1 | Median | Q3 | Max | Mean |
|---|---:|---:|---:|---:|---:|---:|
| `est_diameter_min` | 0.000609 | 0.019256 | 0.048368 | 0.143402 | 37.892650 | 0.127432 |
| `est_diameter_max` | 0.001362 | 0.043057 | 0.108153 | 0.320656 | 84.730541 | 0.284947 |
| `relative_velocity` | 203.35 | 28619.02 | 44190.12 | 62923.60 | 236990.13 | 48066.92 |
| `miss_distance` | 6745.53 | 17210820.24 | 37846579.26 | 56548996.45 | 74798651.45 | 37066546.03 |
| `absolute_magnitude` | 9.23 | 21.34 | 23.70 | 25.70 | 33.20 | 23.53 |

Pearson correlation 顯示，`absolute_magnitude` 與 `hazardous` 的相關係數為 -0.3653，代表絕對星等較低的物體較可能被標記為危險；`relative_velocity` 與 `hazardous` 的相關係數為 0.1912，表示速度也具有一定風險訊號；`miss_distance` 與目標的線性相關較弱，為 0.0423。然而後續模型重要度顯示，`miss_distance` 仍可能透過非線性或 threshold-like 關係影響模型判斷。

![Correlation heatmap](../reports/figures/correlation_heatmap.png)

### 2.4 任務定義

本研究任務為二元分類：

```text
hazardous ∈ {True, False}
```

模型輸出為：

```text
P(hazardous = True | observed features)
```

接著透過決策閾值轉換為最終分類：

```text
if P(hazardous=True) >= threshold:
    predict hazardous
else:
    predict non-hazardous
```

此設計可支援兩種用途：一是建立分類模型，二是使用 hazardous probability 作為風險排序分數。

## 3. 方法

### 3.1 資料切分

資料以 stratified split 切分為 train、validation 與 test 三組，比例約為 70%、15%、15%。使用 stratified split 的原因是正類別僅佔 9.73%，若隨機切分未保留類別比例，可能造成 validation 或 test 評估不穩定。

實際切分後：

| Split | 筆數 |
|---|---:|
| Train | 63,585 |
| Validation | 13,625 |
| Test | 13,626 |

Validation set 用於模型選擇與 threshold tuning；test set 僅用於最終評估，以降低資料洩漏風險。

### 3.2 特徵處理與特徵工程

建模時移除 `id`、`name`、`orbiting_body` 與 `sentry_object`。前兩者為識別欄位，後兩者在本資料快照中為常數欄位。基礎數值特徵包含：

- `est_diameter_min`
- `est_diameter_max`
- `relative_velocity`
- `miss_distance`
- `absolute_magnitude`

此外建立以下衍生特徵：

| 特徵 | 定義 | 目的 |
|---|---|---|
| `est_diameter_mean` | `(est_diameter_min + est_diameter_max) / 2` | 以單一尺寸估計降低冗餘 |
| `est_diameter_range` | `est_diameter_max - est_diameter_min` | 表示尺寸估計範圍 |
| `log_est_diameter_mean` | `log1p(est_diameter_mean)` | 降低尺寸偏態 |
| `log_relative_velocity` | `log1p(relative_velocity)` | 降低速度偏態 |
| `log_miss_distance` | `log1p(miss_distance)` | 降低距離偏態 |

最終輸入模型的特徵數為 10。

### 3.3 模型

本研究比較以下模型：

1. **Majority Baseline**：永遠預測多數類別 `hazardous=False`。
2. **Logistic Regression**：線性模型，搭配 StandardScaler。
3. **Balanced Logistic Regression**：使用 `class_weight="balanced"`。
4. **Balanced Random Forest**：使用 `class_weight="balanced_subsample"`。
5. **HistGradientBoostingClassifier**：以梯度提升模型捕捉非線性關係。
6. **XGBoost**：使用 histogram tree method 與 `scale_pos_weight` 處理不平衡。
7. **LightGBM**：使用 leaf-wise boosting 與 `scale_pos_weight` 作為高效率表格模型。

此外，本研究對 Random Forest、HistGradientBoosting、XGBoost 與 LightGBM 進行快速 hyperparameter tuning。調參使用 `RandomizedSearchCV`、stratified 2-fold cross-validation、每個模型 4 組候選參數，並以 `average_precision`（PR-AUC）作為 scoring。此設計能納入調參比較，同時避免完整 grid search 的計算成本過高。

### 3.4 Threshold Tuning 與 Probability Calibration

模型先輸出 `hazardous=True` 的機率，再於 validation set 上搜尋 threshold。搜尋範圍為 0.05 至 0.95，間距為 0.01，並以 F1 為主要選擇標準，同時參考 recall 與 precision。此設定比預設 threshold = 0.5 更適合不平衡資料與風險篩選情境。

此外，本研究使用 sigmoid calibration 對最佳模型進行機率校準，並以 calibration curve 與 Brier score 檢查機率品質。Calibration 的目的不是提高所有分類指標，而是讓機率輸出更適合作為風險分數解讀。

![Calibration curve](../reports/figures/final_calibration_curve.png)

### 3.5 可解釋性方法

本研究使用兩種方法解釋模型：

1. **Permutation Importance**：以 PR-AUC 作為 scoring，測量單一特徵被打亂後模型表現下降程度。
2. **SHAP**：提供全域特徵貢獻與局部案例解釋，說明各特徵如何推高或拉低 hazardous probability。

Permutation importance 偏向全域模型行為，SHAP 則能針對單一樣本提供局部歸因。

## 4. 實驗結果

### 4.1 Validation 模型比較

Validation set 在 threshold = 0.5 下的主要結果如下：

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
| Balanced Random Forest | 0.8979 | 0.4808 | 0.6154 | 0.5399 | 0.5880 | 0.9347 |

結果顯示，Majority Baseline 雖然 accuracy 達到 0.9027，但 recall 與 F1 皆為 0，證明 accuracy 不適合作為本任務主要指標。XGBoost 與 LightGBM 類模型能取得很高 recall，但 precision 偏低，代表會產生較多誤報；調參後 LightGBM 的 PR-AUC 提升至 0.5687，但仍低於 Balanced Random Forest 的 0.5880。Balanced Random Forest 在 F1、PR-AUC 與 ROC-AUC 上表現較佳，因此被選為後續 calibration 與 test 評估的基礎模型。

### 4.1.1 Hyperparameter Tuning 結果

快速調參的 cross-validation PR-AUC 結果如下：

| Tuned Model | Best CV PR-AUC |
|---|---:|
| LightGBM Tuned | 0.5284 |
| XGBoost Tuned | 0.5276 |
| Random Forest Tuned | 0.5233 |
| HistGradientBoosting Tuned | 0.5208 |

此結果表示，XGBoost 與 LightGBM 在調參搜尋中具競爭力，但其 validation set 表現仍未超過未調參的 balanced Random Forest。考量最終模型選擇以 validation PR-AUC、F1 與 ROC-AUC 綜合排序，本研究維持 `random_forest_balanced` 作為最終模型。

### 4.2 Test Set 最終結果

最終 test set 比較三種設定：

| 設定 | Threshold | Accuracy | Precision | Recall | F1 | PR-AUC | Brier | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw Random Forest | 0.34 | 0.8544 | 0.3838 | 0.8198 | 0.5228 | 0.5634 | 0.0673 | 0.9267 |
| Calibrated Random Forest | 0.19 | 0.8727 | 0.4145 | 0.7474 | 0.5332 | 0.5595 | 0.0600 | 0.9269 |
| Calibrated Random Forest, default threshold | 0.50 | 0.9119 | 0.5946 | 0.2986 | 0.3976 | 0.5595 | 0.0600 | 0.9269 |

校準後模型在 threshold = 0.19 下取得最佳 F1 = 0.5332，recall = 0.7474。若使用預設 threshold = 0.5，accuracy 最高，但 recall 只有 0.2986，表示大量 hazardous 樣本未被找出。這支持本研究的核心設計：在不平衡且具風險篩選意義的任務中，threshold tuning 是必要步驟。

![Precision-recall curve](../reports/figures/final_precision_recall_curve.png)

![ROC curve](../reports/figures/final_roc_curve.png)

### 4.3 混淆矩陣解讀

Calibrated Random Forest 在 threshold = 0.19 下的 test confusion matrix 為：

|  | Predicted False | Predicted True |
|---|---:|---:|
| True False | 10,900 | 1,400 |
| True True | 335 | 991 |

此設定找回 991 個 hazardous 樣本，漏判 335 個 hazardous 樣本。相較於 default threshold = 0.5，其 true positive 從 396 提升到 991，但 false positive 也從 270 增加到 1,400。這反映了 planetary defense screening 情境中的典型取捨：若目標是降低漏判，必須接受較多需要人工複查的候選樣本。

## 5. 可解釋性分析

### 5.1 全域特徵重要度

Permutation importance 顯示前五個重要特徵為：

| 排名 | 特徵 | Importance Mean |
|---:|---|---:|
| 1 | `miss_distance` | 0.0915 |
| 2 | `log_miss_distance` | 0.0893 |
| 3 | `absolute_magnitude` | 0.0451 |
| 4 | `log_est_diameter_mean` | 0.0443 |
| 5 | `est_diameter_min` | 0.0424 |

此結果顯示，模型除了使用星等與尺寸訊號，也強烈依賴接近距離相關特徵。雖然 `miss_distance` 與 `hazardous` 的 Pearson correlation 僅為 0.0423，但在非線性模型中仍可能扮演重要角色。這說明線性相關係數不足以完整描述模型可利用的風險訊號。

![Permutation importance](../reports/figures/permutation_importance.png)

SHAP 全域圖也提供類似觀察：距離、星等與尺寸相關特徵是模型判斷 hazardous probability 的主要來源。

![SHAP global bar](../reports/figures/shap_global_bar.png)

![SHAP summary beeswarm](../reports/figures/shap_summary_beeswarm.png)

### 5.2 局部案例分析

本研究選擇三個 test set 案例：true positive、false negative 與 false positive。

#### 案例一：True Positive

- `id`: 3774091
- `name`: (2017 HP3)
- 真實標籤：hazardous
- 預測機率：0.8957
- threshold：0.19
- 預測結果：hazardous

SHAP 顯示，`log_miss_distance`、`miss_distance`、`est_diameter_min`、`absolute_magnitude` 與 `log_est_diameter_mean` 均將機率往上推。此案例代表模型能根據距離、尺寸與星等訊號正確識別高風險物件。

#### 案例二：False Negative

- `id`: 3713941
- `name`: (2015 EO61)
- 真實標籤：hazardous
- 預測機率：0.1897
- threshold：0.19
- 預測結果：non-hazardous

此案例的預測機率僅略低於 threshold，因此屬於邊界樣本。SHAP 顯示尺寸相關特徵將風險往上推，但 `log_miss_distance` 將機率往下拉。這種案例在實務上值得額外檢查，因為小幅調整 threshold 即可能改變分類結果。

#### 案例三：False Positive

- `id`: 3566975
- `name`: (2011 KO17)
- 真實標籤：non-hazardous
- 預測機率：0.8954
- threshold：0.19
- 預測結果：hazardous

此案例雖然真實標籤為 non-hazardous，但模型給出高風險分數。SHAP 顯示接近距離、星等與尺寸相關特徵都強烈推高機率。這表示模型可能將某些與 hazardous 樣本相似的觀測條件視為高風險。從篩選系統角度看，false positive 會增加人工審查負擔，但比 false negative 更容易被專家後續排除。

## 6. 討論

### 6.1 Threshold 的影響

本研究結果清楚顯示，threshold 決策會顯著改變模型行為。Default threshold = 0.5 雖然提高 precision 與 accuracy，但 recall 明顯下降。對於 planetary defense screening，漏判 hazardous 樣本的成本通常高於誤報，因此較低 threshold 是合理選擇。然而，threshold 不應任意降低，否則 false positive 會過多，導致人工審查負擔過重。本研究選擇 validation F1 最佳的 threshold = 0.19，作為 precision 與 recall 的折衷。

### 6.2 機率分數的正確解讀

Calibrated Random Forest 的 Brier score 為 0.0600，略優於未校準模型的 0.0673，表示 calibration 有助於改善機率品質。然而，此機率仍只代表模型對資料集標籤的估計，不代表真實撞擊機率。若要建立天文物理意義更強的 hazard model，仍需要更完整的軌道參數與領域模型。

### 6.3 特徵解釋與領域直覺

`absolute_magnitude` 與尺寸相關特徵的重要性符合直覺，因為星等與物體大小具有關聯，較大或較亮的物體較可能被標記為潛在危險。`relative_velocity` 在 correlation 中與目標呈正相關，但在 permutation importance 中低於距離與尺寸特徵，表示模型更依賴其他訊號。`miss_distance` 的重要度較高，說明其可能以非線性方式影響預測。

## 7. 限制

本研究仍有以下限制：

1. `hazardous` 是資料集標籤，不等於實際撞擊機率。
2. 資料集欄位有限，缺少完整軌道參數，例如 MOID、軌道傾角、半長軸與離心率。
3. XGBoost、LightGBM 與 hyperparameter tuning 已納入，但本研究採用快速搜尋設定；若要追求最佳效能，仍可擴大搜尋空間與交叉驗證折數。
4. SHAP 解釋反映模型行為，不代表因果關係。
5. Threshold 選擇依據 validation F1，若實務成本函數不同，最佳 threshold 也可能不同。

## 8. 結論

本研究建立了一個可重現、可解釋的 NEO hazardous 二元分類流程。結果顯示，在高度不平衡資料下，accuracy 不能作為主要評估指標；相較之下，PR-AUC、F1、recall 與 confusion matrix 更能反映模型是否適合風險篩選情境。Balanced Random Forest 在 validation set 上表現最佳，經 sigmoid calibration 與 threshold tuning 後，在 test set 上達到 F1 = 0.5332、recall = 0.7474、precision = 0.4145。可解釋性分析顯示，接近距離、星等與尺寸相關特徵是模型主要依據。

整體而言，本研究最合適的定位是：一個具可解釋性的機率式二元分類系統，用於估計 NEO 被標記為 hazardous 的機率，並以風險分數協助專家進行優先級排序。未來若能加入更完整的軌道資料與明確的審查成本函數，模型將更接近實務行星防禦決策需求。

## 參考資料

1. Sameep Vani. NASA Nearest Earth Objects. Kaggle Dataset. <https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects>
2. NASA Open APIs. <https://api.nasa.gov/>
3. NASA Jet Propulsion Laboratory, Center for Near Earth Object Studies. Close-Approach Data. <https://cneos.jpl.nasa.gov/ca/>
4. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems.
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
