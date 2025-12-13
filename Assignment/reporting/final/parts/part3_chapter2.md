
<div style="page-break-after: always;"></div>

# 2. INTEGRATED PERFORMANCE DISCUSSION

**Contributors:** Muhammad Usama Fazal (TP086008), Imran Shahadat Noble (TP087895), Md Sohel Rana (TP086217)

## 2.1 Experimental Setup

### 2.1.1 Dataset Description

The NSL-KDD dataset consists of network connection records, each described by 41 features and labelled with specific attack types grouped into five categories.

**Table 3: Dataset Composition**

| Dataset | Records | Benign | DoS | Probe | R2L | U2R |
|---------|---------|--------|-----|-------|-----|-----|
| Training | 63,280 | 33,672 (53.2%) | 23,066 (36.5%) | 5,911 (9.3%) | 575 (0.9%) | 56 (0.09%) |
| Test | 22,544 | 9,711 (43.1%) | 7,458 (33.1%) | 2,421 (10.7%) | 2,754 (12.2%) | 200 (0.9%) |

Table 3 details the dataset composition showing significant distribution shift between training and test sets. The R2L class increases from 0.9% in training to 12.2% in testing, creating a challenging generalisation scenario. This shift tests each classifier's ability to detect attack patterns from limited training examples, particularly relevant for real-world deployment where novel attack variants may emerge.

### 2.1.2 Data Preprocessing

A consistent preprocessing pipeline was applied across all classifiers:
- **Categorical Encoding:** One-hot encoding expanded features from 41 to 122
- **Feature Scaling:** MinMax normalisation (0-1 range) applied to all numeric features
- **Train-Test Split:** Pre-defined NSL-KDD splits maintained for reproducibility

### 2.1.3 Performance Metrics

Matthews Correlation Coefficient (MCC) was selected as the primary metric because it is more informative for imbalanced datasets (Chicco & Jurman, 2020). MCC considers all four confusion matrix categories and provides a balanced measure even when class sizes differ significantly.

## 2.2 Comparative Analysis of Optimised Models

### 2.2.1 Overall Performance Comparison

**Table 4: Optimised Model Performance Metrics**

| Classifier | Author | Accuracy | F1 (Weighted) | F1 (Macro) | MCC |
|------------|--------|----------|---------------|------------|-----|
| LDA | Muhammad Usama Fazal (TP086008) | 0.775 | 0.763 | 0.671 | 0.671 |
| Random Forest | Imran Shahadat Noble (TP087895) | 0.868 | 0.840 | 0.778 | 0.810 |
| **KNN** | **Md Sohel Rana (TP086217)** | **0.875** | **0.865** | **0.770** | **0.816** |

Table 4 presents the overall performance metrics for all three optimised classifiers. KNN achieves the highest MCC (0.816) and accuracy (87.5%), demonstrating that a well-tuned instance-based learner can outperform ensemble methods for this dataset. The performance gap between linear (LDA: 0.671) and non-linear methods (KNN: 0.816, RF: 0.810) highlights the importance of capturing non-linear patterns in network intrusion detection. This 14.5% MCC difference represents a substantial improvement in detection capability.

**Figure 2: Optimised Models MCC Comparison**

```
MCC Score Comparison (Higher is Better)
═══════════════════════════════════════════════════════════════════

KNN (Sohel Rana)      ████████████████████████████████████████  0.816
Random Forest (Imran) ███████████████████████████████████████   0.810
LDA (Usama Fazal)     ██████████████████████████████            0.671

                      0.0   0.2   0.4   0.6   0.8   1.0
```

Figure 2 visualises the MCC comparison between optimised models. The bar chart clearly shows that KNN and Random Forest achieve similar high performance, both significantly outperforming the linear LDA classifier. This visual representation emphasises that non-linear approaches are essential for the multi-class intrusion detection task where attack patterns exhibit complex, overlapping characteristics in the feature space.

### 2.2.2 MCC Per Attack Class

**Table 5: MCC Performance by Attack Category**

| Attack Class | LDA (Usama Fazal) | Random Forest (Imran Noble) | KNN (Sohel Rana) | Best |
|--------------|-------------------|----------------------------|------------------|------|
| Benign | 0.673 | 0.757 | **0.786** | KNN |
| DoS | 0.786 | **0.984** | 0.946 | RF |
| Probe | 0.575 | **0.911** | 0.850 | RF |
| R2L | 0.513 | 0.336 | **0.567** | KNN |
| U2R | 0.579 | **0.847** | 0.572 | RF |

Table 5 reveals important class-specific performance differences. Random Forest excels at detecting DoS (0.984), Probe (0.911), and U2R (0.847) attacks, benefiting from ensemble averaging and its ability to capture feature interactions. KNN performs best on Benign (0.786) and R2L (0.567) classifications, demonstrating its effectiveness in identifying local patterns in the feature space. The R2L class remains challenging for all classifiers due to the severe distribution shift between training (0.9%) and test (12.2%) sets, though KNN's instance-based approach provides the best generalisation.

**Figure 3: MCC Per Attack Class Comparison**

```
                     LDA        RF         KNN
                   (Usama)   (Imran)    (Sohel)
═══════════════════════════════════════════════════════════════════
Benign      ████▌     ██████     ███████      Best: KNN (0.786)
DoS         ██████▌   █████████▌ █████████    Best: RF  (0.984)
Probe       ████▌     █████████  ████████     Best: RF  (0.911)
R2L         ████      ██▌        ████▌        Best: KNN (0.567)
U2R         ████▌     ████████   ████▌        Best: RF  (0.847)
═══════════════════════════════════════════════════════════════════
```

Figure 3 illustrates the per-class MCC performance, highlighting that no single classifier dominates across all attack types. This finding suggests that a hybrid approach combining Random Forest for DoS/Probe/U2R detection with KNN for Benign/R2L classification could further improve overall detection rates in production deployments.

## 2.3 Cross-Validation Results

**Table 6: Cross-Validation Results (F1-Weighted, 5-Fold)**

| Classifier | Author | CV Mean | CV Std | 95% CI |
|------------|--------|---------|--------|--------|
| LDA | Muhammad Usama Fazal | 0.931 | 0.003 | 0.931 ± 0.005 |
| **Random Forest** | **Imran Shahadat Noble** | **0.996** | **0.000** | 0.996 ± 0.001 |
| KNN | Md Sohel Rana | 0.992 | 0.001 | 0.992 ± 0.001 |

Table 6 presents the cross-validation results, demonstrating model stability across different data folds. Random Forest achieves the highest CV score (0.996) with the lowest variance (0.000), indicating excellent consistency on the training data. However, the discrepancy between CV performance and test set results (CV: 0.996 vs Test MCC: 0.810) reveals potential overfitting to the training distribution. KNN shows better generalisation to the test set despite slightly lower CV scores, suggesting its instance-based approach adapts better to the distribution shift present in NSL-KDD.

**Figure 4: Cross-Validation Performance**

```
Cross-Validation F1-Score (5-Fold)
═══════════════════════════════════════════════════════════════════

Random Forest   ████████████████████████████████████████████████  0.996
KNN             ███████████████████████████████████████████████   0.992
LDA             ██████████████████████████████████████████        0.931

                0.90        0.92        0.94        0.96        0.98        1.00
```

Figure 4 shows that all three classifiers achieve high cross-validation scores on the training data. The minimal variance in Random Forest's CV results indicates robust ensemble averaging, while LDA's lower score reflects the limitation of linear decision boundaries. The key insight is that CV performance alone does not guarantee test set generalisation, particularly when distribution shift exists between training and test data.

## 2.4 Feature Selection Impact

**Table 7: Feature Reduction Summary**

| Classifier | Author | Original | Selected | Reduction | Method |
|------------|--------|----------|----------|-----------|--------|
| LDA | Muhammad Usama Fazal | 122 | 30 | 75.4% | Correlation (>0.1) |
| Random Forest | Imran Shahadat Noble | 122 | 38 | 68.9% | Importance (95%) |
| KNN | Md Sohel Rana | 122 | 30 | 75.4% | Correlation (>0.1) |

Table 7 summarises the feature selection strategies applied by each team member. All classifiers achieved substantial dimensionality reduction (69-75%) while maintaining or improving performance. LDA and KNN used correlation-based selection, removing features with weak target correlation (<0.1 threshold). Random Forest used its built-in feature importance, selecting features contributing to 95% cumulative importance. The consistent effectiveness of feature selection across all classifiers demonstrates that the NSL-KDD dataset contains significant redundancy, and removing noisy features improves model focus on discriminative patterns.

## 2.5 Key Findings and Recommendations

### 2.5.1 Final Classifier Ranking

**Table 8: Final Classifier Ranking**

| Rank | Classifier | Author | MCC | Accuracy | Key Strength |
|------|------------|--------|-----|----------|--------------|
| 1st | KNN | Md Sohel Rana (TP086217) | 0.816 | 87.5% | Best overall, highest accuracy |
| 2nd | Random Forest | Imran Shahadat Noble (TP087895) | 0.810 | 86.8% | Best DoS/Probe/U2R detection |
| 3rd | LDA | Muhammad Usama Fazal (TP086008) | 0.671 | 77.5% | Fast, interpretable |

Table 8 presents the final ranking based on overall MCC performance. KNN emerges as the best performer with MCC 0.816, closely followed by Random Forest at 0.810. The marginal difference (0.006) between the top two classifiers suggests both are viable for deployment, with selection depending on specific operational requirements. LDA's significantly lower performance (0.671) confirms that linear assumptions are inadequate for the complex, non-linear patterns inherent in network intrusion data.

### 2.5.2 Recommendations

**For Maximum Detection Accuracy:** Deploy KNN (k=3, Manhattan distance, distance weighting) with correlation-based feature selection (30 features). Trade-off: Higher prediction-time computation.

**For Real-Time Systems:** Deploy Random Forest (100 trees, balanced class weights) with importance-based feature selection (38 features). Trade-off: Slightly lower R2L detection.

**For Interpretability:** Deploy LDA with correlation-based feature selection for environments requiring explainable decisions, accepting the performance trade-off.

**Conclusion:** KNN (Md Sohel Rana - TP086217) achieved the best overall performance with MCC 0.816, demonstrating that a well-optimised instance-based learner can effectively detect multi-class network intrusions. Random Forest (Imran Shahadat Noble - TP087895) offers a strong alternative with superior detection of specific attack types. The substantial improvement of non-linear methods over linear LDA (Muhammad Usama Fazal - TP086008) confirms the necessity of capturing complex decision boundaries for effective intrusion detection.

---
