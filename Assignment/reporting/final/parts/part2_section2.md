<div style="page-break-after: always;"></div>

# 2. INTEGRATED PERFORMANCE DISCUSSION

**Contributors:** Muhammad Usama Fazal (TP086008), Imran Shahadat Noble (TP087895), Md Sohel Rana (TP086217)

## 2.1 Experimental Setup

### 2.1.1 Dataset Description

The NSL-KDD dataset used in this study consists of network connection records, each described by 41 features and labelled with specific attack types grouped into five categories.

**Table 3: Dataset Composition**

| Dataset | Records | Benign | DoS | Probe | R2L | U2R |
|---------|---------|--------|-----|-------|-----|-----|
| Training Set | 63,280 | 33,672 (53.2%) | 23,066 (36.5%) | 5,911 (9.3%) | 575 (0.9%) | 56 (0.09%) |
| Test Set | 22,544 | 9,711 (43.1%) | 7,458 (33.1%) | 2,421 (10.7%) | 2,754 (12.2%) | 200 (0.9%) |

**Key Challenge:** The significant distribution shift between training and test sets for R2L class (0.9% to 12.2%) poses a particular challenge for model generalisation.

### 2.1.2 Data Preprocessing Pipeline

A consistent preprocessing pipeline was applied across all classifiers to ensure fair comparison:

**Step 1: Categorical Encoding**
One-hot encoding was applied to three categorical features:
- **protocol_type:** tcp, udp, icmp (3 categories)
- **service:** http, ftp, smtp, etc. (70 categories)
- **flag:** SF, S0, REJ, etc. (11 categories)

This expanded the feature space from 41 to 122 features.

**Step 2: Feature Scaling**
MinMax normalisation (0-1 range) was applied to all numeric features. This ensures equal contribution to distance-based calculations (critical for KNN) and improves convergence for gradient-based methods.

**Step 3: Train-Test Split**
The pre-defined NSL-KDD train and test splits were used to maintain consistency with benchmark studies and ensure reproducibility.

### 2.1.3 Computational Environment

All experiments were conducted using Python 3.x with the scikit-learn machine learning library (Pedregosa et al., 2011). The consistent software environment ensures reproducibility and fair comparison across all classifiers.

## 2.2 Performance Metrics Selection

For multi-class network intrusion detection systems, the choice of evaluation metrics is crucial. The following metrics were selected based on their relevance to cybersecurity applications:

**Table 4: Selected Performance Metrics and Their Significance**

| Metric | Description | Significance for Multi-class IDS |
|--------|-------------|----------------------------------|
| **Accuracy** | Overall correctness | Basic measure, but misleading for imbalanced data |
| **F1-Score (Weighted)** | Weighted average of per-class F1 | Accounts for class imbalance by weighting |
| **F1-Score (Macro)** | Unweighted average of per-class F1 | Equal importance to all classes regardless of size |
| **MCC** | Matthews Correlation Coefficient | Most informative single metric for imbalanced multi-class problems |
| **MCC per Class** | Class-specific correlation | Identifies which attack types are well/poorly detected |

**Why MCC is the Primary Metric:**
Matthews Correlation Coefficient is selected as the primary ranking metric because:
1. It is more informative for imbalanced datasets (Chicco & Jurman, 2020)
2. It considers all four confusion matrix categories (TP, TN, FP, FN)
3. It provides a balanced measure even when class sizes are very different
4. A value of +1 indicates perfect prediction, 0 is random, -1 is total disagreement

## 2.3 Comparative Analysis of Optimised Models

### 2.3.1 Overall Performance Comparison

**Table 5: Optimised Model Performance Metrics**

| Classifier | Author | Accuracy | F1 (Weighted) | F1 (Macro) | MCC | Train Time |
|------------|--------|----------|---------------|------------|-----|------------|
| LDA (Linear) | Muhammad Usama Fazal (TP086008) | 0.775 | 0.763 | 0.671 | 0.671 | 0.34s |
| Random Forest (Ensemble) | Imran Shahadat Noble (TP087895) | 0.868 | 0.840 | 0.778 | 0.810 | 3.03s |
| **KNN (Non-Linear)** | **Md Sohel Rana (TP086217)** | **0.875** | **0.865** | **0.770** | **0.816** | 15.23s |

The results reveal that **KNN (Md Sohel Rana - TP086217) achieved the highest overall MCC (0.816)**, indicating the best correlation between predictions and actual labels among all tested classifiers.

![Figure 2: Model Performance Comparison](../../figures/fig1_model_comparison.png)

*Figure 2: Bar chart comparing accuracy, F1-weighted, F1-macro, and MCC scores across all three optimised classifiers. KNN achieves the highest overall performance.*

![Figure 3: Radar Chart Comparison](../../figures/fig2_radar_comparison.png)

*Figure 3: Radar chart showing multi-dimensional performance comparison across all metrics. The larger area indicates better overall performance.*

### 2.3.2 MCC Per Attack Class

**Table 6: MCC Performance by Attack Category**

| Attack Class | LDA (Muhammad Usama Fazal) | Random Forest (Imran Shahadat Noble) | KNN (Md Sohel Rana) | Best Performer |
|--------------|-----|---------------|-----|----------------|
| **Benign** | 0.673 | 0.757 | **0.786** | KNN (Md Sohel Rana) |
| **DoS** | 0.786 | **0.984** | 0.946 | Random Forest (Imran Shahadat Noble) |
| **Probe** | 0.575 | **0.911** | 0.850 | Random Forest (Imran Shahadat Noble) |
| **R2L** | 0.513 | 0.336 | **0.567** | KNN (Md Sohel Rana) |
| **U2R** | 0.579 | **0.847** | 0.572 | Random Forest (Imran Shahadat Noble) |

**Key Observations:**
1. Random Forest (Imran Shahadat Noble - TP087895) excels at detecting DoS (0.984), Probe (0.911), and U2R (0.847) attacks
2. KNN (Md Sohel Rana - TP086217) performs best on Benign traffic (0.786) and R2L attacks (0.567)
3. LDA (Muhammad Usama Fazal - TP086008) struggles across all attack types, with particularly poor performance on Probe (0.575)
4. R2L detection remains challenging for all classifiers due to severe class imbalance

![Figure 4: MCC Per Class Comparison](../../figures/fig4_mcc_per_class.png)

*Figure 4: Grouped bar chart showing MCC performance for each attack class across all three classifiers. Random Forest dominates in DoS, Probe, and U2R detection, while KNN leads in Benign and R2L.*

![Figure 5: Performance Heatmap](../../figures/fig3_performance_heatmap.png)

*Figure 5: Heatmap visualization of classifier performance across different metrics. Darker colors indicate higher performance values.*

### 2.3.3 Cross-Validation Results

5-fold stratified cross-validation was performed to assess model stability:

**Table 7: Cross-Validation Results (F1-Weighted)**

| Classifier | Author | CV Mean | CV Std | 95% CI |
|------------|--------|---------|--------|--------|
| LDA | Muhammad Usama Fazal (TP086008) | 0.931 | 0.003 | 0.931 +/- 0.005 |
| Random Forest | Imran Shahadat Noble (TP087895) | **0.996** | **0.000** | 0.996 +/- 0.001 |
| KNN | Md Sohel Rana (TP086217) | 0.992 | 0.001 | 0.992 +/- 0.001 |

Random Forest (Imran Shahadat Noble) demonstrates the highest cross-validation performance with the lowest variance, indicating excellent training set fitting. However, test set performance shows KNN (Md Sohel Rana) generalises better to unseen data.

![Figure 6: Cross-Validation Box Plot](../../figures/cross_validation_boxplot.png)

*Figure 6: Box plot showing the distribution of cross-validation scores for each classifier. Random Forest shows the tightest distribution with highest median score.*

### 2.3.4 Feature Selection Impact

**Table 8: Feature Reduction Summary**

| Classifier | Author | Original Features | Selected Features | Reduction | Selection Method |
|------------|--------|-------------------|-------------------|-----------|------------------|
| LDA | Muhammad Usama Fazal | 122 | 30 | 75.4% | Correlation-based (threshold > 0.1) |
| Random Forest | Imran Shahadat Noble | 122 | 38 | 68.9% | Importance-based (95% cumulative) |
| KNN | Md Sohel Rana | 122 | 30 | 75.4% | Correlation-based (threshold > 0.1) |

Feature selection improved or maintained performance while reducing computational requirements for all classifiers.

![Figure 7: Feature Reduction Impact](../../figures/fig7_feature_reduction.png)

*Figure 7: Comparison of original vs selected features for each classifier, showing the effectiveness of feature selection strategies.*

### 2.3.5 Baseline vs Optimised Comparison

![Figure 8: Baseline vs Optimised MCC](../../figures/baseline_vs_optimised_mcc.png)

*Figure 8: Comparison of MCC scores before and after optimisation for each classifier. KNN shows the most significant improvement.*

### 2.3.6 Confusion Matrix Analysis

![Figure 9: Confusion Matrices for All Models](../../figures/confusion_matrices_all_models.png)

*Figure 9: Side-by-side confusion matrices for all three optimised classifiers, showing the distribution of predictions across all five classes.*

### 2.3.7 Training Time Comparison

![Figure 10: Training Time Comparison](../../figures/fig5_training_time.png)

*Figure 10: Bar chart comparing training times across classifiers. LDA is the fastest, while KNN requires the most time due to its instance-based nature.*

## 2.4 Key Findings and Insights

### 2.4.1 Key Insights

1. **Best Overall Performer:** KNN (Md Sohel Rana - TP086217) with optimised parameters achieved the highest MCC (0.816), demonstrating that a well-tuned instance-based learner can outperform more complex ensemble methods for this dataset.

2. **Linear Methods Limitation:** LDA (Muhammad Usama Fazal - TP086008) achieved an MCC of only 0.671, significantly lower than non-linear methods. This highlights the limitation of linear decision boundaries for capturing complex multi-class attack patterns.

3. **Ensemble Methods Stability:** Random Forest (Imran Shahadat Noble - TP087895) showed consistent, high performance with minimal change after optimisation (-0.4% MCC), suggesting its default parameters are already well-suited for intrusion detection tasks.

4. **Class-Specific Performance Varies:** No single classifier dominates across all attack types:
   - Random Forest (Imran Shahadat Noble) excels at DoS, Probe, and U2R
   - KNN (Md Sohel Rana) excels at Benign and R2L
   - An ensemble of classifiers might further improve overall performance

5. **Distance Metric Importance:** KNN's improvement was largely attributed to using Manhattan distance (p=1) instead of Euclidean distance, which is more robust in high-dimensional spaces.

### 2.4.2 Ranking of Classifiers

**Table 9: Final Classifier Ranking (Multi-criteria)**

| Rank | Classifier | Author | Category | MCC | Accuracy | Key Strength |
|------|------------|--------|----------|-----|----------|--------------|
| **1st** | KNN | **Md Sohel Rana (TP086217)** | Non-Linear | 0.816 | 87.5% | Best overall balance, highest accuracy |
| **2nd** | Random Forest | **Imran Shahadat Noble (TP087895)** | Ensemble | 0.810 | 86.8% | Fastest training, best DoS/Probe detection |
| **3rd** | LDA | **Muhammad Usama Fazal (TP086008)** | Linear | 0.671 | 77.5% | Fast prediction, interpretable |

![Figure 11: Final Ranking Podium](../../figures/fig9_ranking_podium.png)

*Figure 11: Visual representation of the final classifier ranking based on overall MCC performance.*

## 2.5 Recommendations for Deployment

### For Maximum Detection Accuracy:
**Recommended Classifier:** K-Nearest Neighbors (Md Sohel Rana - TP086217)
- Configuration: k=3, distance weighting, Manhattan distance (p=1)
- Feature selection: correlation-based (threshold > 0.1), 30 features
- Expected Performance: MCC 0.816, Accuracy 87.5%
- Trade-off: Higher computational cost at prediction time

### For Real-Time Detection Requirements:
**Recommended Classifier:** Random Forest (Imran Shahadat Noble - TP087895)
- Configuration: 100 trees, max_depth=None, class_weight='balanced'
- Feature selection: importance-based, 38 features
- Expected Performance: MCC 0.810, Accuracy 86.8%
- Trade-off: Slightly lower R2L detection

### For Balanced Performance and Robustness:
**Recommended Classifier:** Random Forest (Imran Shahadat Noble - TP087895)
- Default or slightly tuned parameters work well
- Provides feature importance for security analyst interpretation
- Most consistent cross-validation performance

---
