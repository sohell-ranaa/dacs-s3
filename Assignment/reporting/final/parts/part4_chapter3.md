<div style="page-break-after: always;"></div>

# 3. INDIVIDUAL REPORTS

---

## 3.1 Linear Discriminant Analysis (LDA)

**Author:** Muhammad Usama Fazal
**TP Number:** TP086008
**Classifier Category:** Linear

### 3.1.1 Baseline Model Comparison

Three linear classifiers were evaluated as baselines to select the best algorithm for optimisation.

**Table 9: Linear Classifier Baseline Comparison**

| Algorithm | Accuracy | F1 (Weighted) | F1 (Macro) | MCC | Train Time (s) |
|-----------|----------|---------------|------------|-----|----------------|
| LDA | 0.769 | 0.750 | 0.599 | 0.664 | 2.62 |
| **Logistic Regression** | **0.819** | **0.824** | **0.702** | **0.738** | 20.98 |
| Ridge Classifier | 0.771 | 0.789 | 0.645 | 0.676 | 0.69 |

Table 9 presents the baseline comparison of three linear classifiers. While Logistic Regression achieved the highest MCC (0.738), LDA was selected for optimisation due to its significantly faster training time (2.62s vs 20.98s) and native dimensionality reduction capability. LDA's computational efficiency makes it suitable for scenarios requiring frequent model retraining, and its projection-based approach provides insights into feature relationships that are valuable for security analysis.

**Figure 5: LDA Baseline Comparison**

```
MCC Score - Linear Classifiers (Baseline)
═══════════════════════════════════════════════════════════════════

Logistic Regression   █████████████████████████████████████  0.738
Ridge Classifier      ███████████████████████████████        0.676
LDA (Selected)        ██████████████████████████████         0.664

                      0.0   0.2   0.4   0.6   0.8   1.0
```

Figure 5 visualises the baseline MCC comparison. Although LDA has the lowest baseline MCC among linear classifiers, its selection is justified by the substantial training time advantage (8x faster than Logistic Regression) and the opportunity to demonstrate optimisation impact on a classifier with clear improvement potential.

### 3.1.2 Optimisation Strategy: Hyperparameter Tuning

**Table 10: LDA Hyperparameter Tuning Results**

| Parameter | Values Tested | Justification | Reference |
|-----------|--------------|---------------|-----------|
| solver | svd, lsqr, eigen | SVD stable for most cases; lsqr/eigen allow shrinkage | Hastie et al. (2009) |
| shrinkage | None, auto, 0.1, 0.5, 0.9 | Regularisation to prevent overfitting | Ledoit & Wolf (2004) |

GridSearchCV with 5-fold stratified cross-validation was used to identify optimal hyperparameters. The search evaluated 8 configurations across solver and shrinkage combinations. The SVD solver with no shrinkage achieved the best CV F1-score of 0.9612, indicating that the unregularised approach best preserves discriminative information in the projected space.

**Best Configuration:**
- **Solver:** svd (Singular Value Decomposition)
- **Shrinkage:** None

### 3.1.3 Optimisation Strategy: Feature Selection via Correlation

Correlation-based feature selection was implemented to reduce dimensionality and remove noisy features:

1. Computed Pearson correlation between each feature and the encoded target
2. Ranked features by absolute correlation value
3. Selected features with correlation > 0.1 threshold

**Top 5 Selected Features:**
1. dst_host_srv_count (correlation: 0.617)
2. logged_in (correlation: 0.570)
3. flag_SF (correlation: 0.537)
4. dst_host_same_srv_rate (correlation: 0.518)
5. service_http (correlation: 0.508)

**Result:** Feature space reduced from 122 to 30 features (75.4% reduction).

### 3.1.4 Baseline vs Optimised Model Comparison

**Table 11: LDA Baseline vs Optimised Performance**

| Metric | Baseline | Optimised | Change | % Change |
|--------|----------|-----------|--------|----------|
| Accuracy | 0.769 | 0.775 | +0.006 | +0.8% |
| F1 (Weighted) | 0.750 | 0.763 | +0.013 | +1.7% |
| F1 (Macro) | 0.599 | 0.671 | +0.072 | **+12.0%** |
| MCC | 0.664 | 0.671 | +0.007 | +1.1% |
| Train Time (s) | 2.62 | 0.34 | -2.28 | **-87.0%** |

Table 11 summarises the performance improvement achieved through optimisation. The most significant gains are observed in F1 (Macro) (+12.0%), indicating improved balance across all five attack classes. This improvement is attributed to feature selection removing noisy features that disproportionately affected minority class detection. The training time reduction of 87% demonstrates that feature selection not only improves performance but also dramatically reduces computational requirements.

**Figure 6: LDA Baseline vs Optimised MCC**

```
LDA Performance Comparison
═══════════════════════════════════════════════════════════════════

              Baseline                Optimised
MCC:          ██████████████████████  ██████████████████████▌
              0.664                   0.671 (+1.1%)

F1 (Macro):   ████████████████        ██████████████████████
              0.599                   0.671 (+12.0%)
═══════════════════════════════════════════════════════════════════
```

Figure 6 illustrates the improvement from baseline to optimised LDA. While the overall MCC improvement appears modest (+1.1%), the F1 (Macro) improvement of 12.0% reveals substantial gains in minority class detection. The optimised model correctly identified more U2R attacks (MCC improved from 0.314 to 0.579) and R2L attacks (MCC improved from 0.448 to 0.513), demonstrating that feature selection effectively removed features that hindered minority class recognition.

### 3.1.5 Analysis and Conclusions

The optimisation strategies successfully improved LDA's multi-class detection capability while dramatically reducing training time. However, LDA's linear assumption fundamentally limits its performance compared to non-linear classifiers (final MCC: 0.671 vs KNN: 0.816). LDA is recommended for:
- Real-time systems requiring fast prediction
- Scenarios needing interpretable decision boundaries
- Use as a first-stage filter in multi-stage detection systems

---

<div style="page-break-after: always;"></div>

## 3.2 Random Forest

**Author:** Imran Shahadat Noble
**TP Number:** TP087895
**Classifier Category:** Ensemble (Bagging)

### 3.2.1 Baseline Model Comparison

Three ensemble classifiers were evaluated as baselines.

**Table 12: Ensemble Classifier Baseline Comparison**

| Algorithm | Accuracy | F1 (Weighted) | F1 (Macro) | MCC | Train Time (s) |
|-----------|----------|---------------|------------|-----|----------------|
| **Random Forest** | **0.871** | 0.845 | 0.779 | **0.814** | 3.25 |
| Extra Trees | 0.872 | **0.847** | **0.795** | 0.814 | 3.75 |
| AdaBoost | 0.734 | 0.686 | 0.476 | 0.617 | 21.73 |

Table 12 shows that Random Forest and Extra Trees achieve identical MCC (0.814), substantially outperforming AdaBoost (0.617). Random Forest was selected due to faster training time (3.25s vs 3.75s) and better feature importance interpretability. AdaBoost's poor performance is attributed to its sensitivity to noise and the sequential boosting approach that amplifies errors on the severely imbalanced minority classes.

**Figure 7: Random Forest Baseline Comparison**

```
MCC Score - Ensemble Classifiers (Baseline)
═══════════════════════════════════════════════════════════════════

Random Forest (Selected)  █████████████████████████████████████████  0.814
Extra Trees               █████████████████████████████████████████  0.814
AdaBoost                  ██████████████████████████████             0.617

                          0.0   0.2   0.4   0.6   0.8   1.0
```

Figure 7 visualises the baseline comparison, clearly showing the performance gap between bagging methods (Random Forest, Extra Trees) and boosting (AdaBoost). The bagging approach's robustness to outliers and class imbalance makes it superior for the NSL-KDD dataset with its 0.09% U2R class representation.

### 3.2.2 Optimisation Strategy: Hyperparameter Tuning

**Table 13: Random Forest Hyperparameter Configuration**

| Parameter | Values Tested | Best Value | Justification |
|-----------|--------------|------------|---------------|
| n_estimators | 100, 150 | 100 | Sufficient trees for stability (Oshiro et al., 2012) |
| max_depth | 20, None | None | Full growth captures complex patterns (Breiman, 2001) |
| min_samples_split | 2, 5 | 2 | Allow fine-grained splits |
| min_samples_leaf | 1, 2 | 1 | Preserve minority class instances |
| class_weight | balanced | balanced | Address severe class imbalance |

RandomizedSearchCV with 3-fold cross-validation was employed to tune hyperparameters. The search evaluated 10 configurations, achieving a best CV F1-score of 0.9929. The optimal configuration allows trees to grow to full depth while using class weighting to address the severe imbalance between majority (Benign: 53.2%) and minority (U2R: 0.09%) classes.

### 3.2.3 Optimisation Strategy: Feature Selection (Importance-based)

Random Forest's built-in feature importance was used for selection:

1. Extracted mean decrease in impurity for all 122 features
2. Ranked features by importance score
3. Selected features contributing to 95% cumulative importance

**Top 5 Important Features:**
1. dst_host_srv_count (importance: 0.063)
2. dst_host_diff_srv_rate (importance: 0.046)
3. dst_host_same_src_port_rate (importance: 0.046)
4. count (importance: 0.043)
5. dst_host_same_srv_rate (importance: 0.042)

**Result:** Feature space reduced from 122 to 38 features (68.9% reduction).

### 3.2.4 Baseline vs Optimised Model Comparison

**Table 14: Random Forest Baseline vs Optimised Performance**

| Metric | Baseline | Optimised | Change | % Change |
|--------|----------|-----------|--------|----------|
| Accuracy | 0.871 | 0.868 | -0.003 | -0.3% |
| F1 (Weighted) | 0.845 | 0.840 | -0.005 | -0.6% |
| F1 (Macro) | 0.779 | 0.778 | -0.001 | -0.1% |
| MCC | 0.814 | 0.810 | -0.004 | -0.5% |
| Train Time (s) | 3.25 | 2.55 | -0.70 | **-21.5%** |

Table 14 reveals that Random Forest's baseline performance was already near-optimal. The minimal change (-0.5% MCC) after optimisation indicates that scikit-learn's default parameters are well-suited for this classification task. The 21.5% training time reduction from feature selection provides computational benefit without significant performance loss. Notably, the optimised model improved U2R detection (MCC: 0.820 → 0.847) while sacrificing slightly on R2L (MCC: 0.369 → 0.336).

**Figure 8: Random Forest Baseline vs Optimised MCC**

```
Random Forest Performance Comparison
═══════════════════════════════════════════════════════════════════

              Baseline                Optimised
MCC:          █████████████████████████████████████████  ████████████████████████████████████████
              0.814                   0.810 (-0.5%)

U2R MCC:      █████████████████████████████████████████  ██████████████████████████████████████████
              0.820                   0.847 (+3.3%)
═══════════════════════════════════════════════════════════════════
```

Figure 8 shows the stable overall performance with targeted improvement in U2R detection. The class weighting strategy effectively prioritised the smallest attack class (0.09% training samples) without substantially degrading overall metrics. This trade-off demonstrates that optimisation can be directed toward specific operational priorities, such as detecting privilege escalation attacks (U2R).

### 3.2.5 Analysis and Conclusions

Random Forest demonstrated excellent baseline performance with minimal optimisation benefit, suggesting its default parameters are robust for intrusion detection tasks. The key insights are:
- Class weighting effectively handles severe imbalance
- Feature importance-based selection maintains performance while reducing computation
- The ensemble averaging mechanism provides inherent regularisation

Random Forest is recommended for production deployment due to its consistent performance, interpretable feature importance rankings, and reasonable training time.

---

<div style="page-break-after: always;"></div>

## 3.3 K-Nearest Neighbors (KNN)

**Author:** Md Sohel Rana
**TP Number:** TP086217
**Classifier Category:** Non-Linear

### 3.3.1 Baseline Model Comparison

Three non-linear classifiers were evaluated as baselines.

**Table 15: Non-Linear Classifier Baseline Comparison**

| Algorithm | Accuracy | F1 (Weighted) | F1 (Macro) | MCC | Train Time (s) |
|-----------|----------|---------------|------------|-----|----------------|
| KNN | 0.837 | 0.812 | **0.756** | 0.760 | 6.99 |
| Decision Tree | 0.841 | 0.819 | 0.708 | 0.767 | 1.02 |
| **SVM-RBF** | **0.843** | **0.837** | 0.751 | **0.769** | 117.96 |

Table 15 shows SVM-RBF achieved the highest baseline MCC (0.769), but KNN was selected for optimisation due to its dramatically faster training time (6.99s vs 117.96s) and highest F1 (Macro) score (0.756). The F1 (Macro) metric indicates better balance across all classes, which is critical for detecting rare attack types. KNN's instance-based approach also offers greater potential for improvement through distance metric and neighbour weighting optimisation.

**Figure 9: KNN Baseline Comparison**

```
MCC Score - Non-Linear Classifiers (Baseline)
═══════════════════════════════════════════════════════════════════

SVM-RBF               ████████████████████████████████████████  0.769
Decision Tree         ████████████████████████████████████████  0.767
KNN (Selected)        ███████████████████████████████████████   0.760

                      0.0   0.2   0.4   0.6   0.8   1.0
```

Figure 9 illustrates the close baseline performance among non-linear classifiers. The selection of KNN over SVM-RBF is justified by the 17x training time advantage and KNN's superior F1 (Macro), indicating better minority class handling that is essential for detecting R2L and U2R attacks.

### 3.3.2 Optimisation Strategy: Hyperparameter Tuning

**Table 16: KNN Hyperparameter Tuning Results**

| Parameter | Values Tested | Best Value | Justification |
|-----------|--------------|------------|---------------|
| n_neighbors | 3, 5, 7, 9 | **3** | Captures local patterns; odd prevents ties (Cover & Hart, 1967) |
| weights | uniform, distance | **distance** | Closer neighbours have more influence (Hastie et al., 2009) |
| p | 1, 2 | **1 (Manhattan)** | More robust in high-dimensional spaces (Aggarwal et al., 2001) |

GridSearchCV with 3-fold cross-validation exhaustively searched 16 configurations. The optimal parameters (k=3, distance weighting, Manhattan distance) achieved the best CV F1-score of 0.9921.

**Key Finding:** Manhattan distance (p=1) outperformed Euclidean distance (p=2) because:
- It treats each dimension equally rather than squaring differences
- It is more robust to outliers common in network traffic data
- It performs better in high-dimensional spaces affected by the curse of dimensionality

### 3.3.3 Optimisation Strategy: Feature Selection via Correlation

Identical to LDA, correlation-based feature selection was applied:

1. Computed Pearson correlation between features and target
2. Selected features with |correlation| > 0.1

**Result:** Feature space reduced from 122 to 30 features (75.4% reduction).

Feature selection is particularly critical for KNN because distance calculations are sensitive to irrelevant features. Removing low-correlation features ensures that distance measurements reflect meaningful similarity in attack patterns rather than noise.

### 3.3.4 Baseline vs Optimised Model Comparison

**Table 17: KNN Baseline vs Optimised Performance**

| Metric | Baseline | Optimised | Change | % Change |
|--------|----------|-----------|--------|----------|
| Accuracy | 0.837 | 0.875 | +0.038 | **+4.5%** |
| F1 (Weighted) | 0.812 | 0.865 | +0.053 | **+6.5%** |
| F1 (Macro) | 0.756 | 0.770 | +0.014 | +1.9% |
| MCC | 0.760 | 0.816 | +0.056 | **+7.4%** |
| Train Time (s) | 6.99 | 13.37 | +6.38 | +91.3% |

Table 17 demonstrates substantial improvements across all metrics. The 7.4% MCC improvement is the largest among all three classifiers, validating the selection of KNN for optimisation. The accuracy gain of 4.5% (from 83.7% to 87.5%) represents over 1,000 additional correctly classified samples in the test set. The increased training time is acceptable given the significant performance gains.

**Per-Class MCC Improvements:**
- Benign: 0.713 → 0.786 (+10.2%)
- DoS: 0.936 → 0.946 (+1.1%)
- Probe: 0.796 → 0.850 (+6.8%)
- R2L: 0.349 → 0.567 (**+62.5%**)
- U2R: 0.863 → 0.572 (-33.7%)

**Figure 10: KNN Baseline vs Optimised MCC**

```
KNN Performance Comparison
═══════════════════════════════════════════════════════════════════

              Baseline                Optimised
MCC:          ███████████████████████████████████████   █████████████████████████████████████████
              0.760                   0.816 (+7.4%)

R2L MCC:      █████████████████                         ██████████████████████████
              0.349                   0.567 (+62.5%)
═══════════════════════════════════════════════════════════════════
```

Figure 10 highlights the dramatic improvement in KNN performance after optimisation. The R2L class MCC improvement of 62.5% is particularly significant given the severe distribution shift between training (0.9%) and test (12.2%) sets. This improvement demonstrates that the optimised KNN configuration better generalises to underrepresented attack patterns.

### 3.3.5 Analysis and Conclusions

KNN achieved the highest overall performance (MCC: 0.816) among all classifiers through careful optimisation of distance metric, neighbour count, and weighting scheme. The key success factors were:

1. **Manhattan Distance:** More robust than Euclidean in high-dimensional feature space
2. **Small k (k=3):** Captures local attack signature patterns
3. **Distance Weighting:** Increases confidence when query points are near known attack instances
4. **Feature Selection:** Removes noisy dimensions that distort distance calculations

**Trade-off:** U2R detection decreased (-33.7%) while R2L improved substantially (+62.5%). This trade-off may be acceptable since R2L comprises 12.2% of test data compared to U2R's 0.9%, and Random Forest can complement KNN for U2R detection in a hybrid system.

KNN is recommended as the primary classifier for maximum detection accuracy, with the caveat that prediction-time computation scales with training set size and may require optimisation (e.g., KD-trees, approximate nearest neighbours) for high-throughput deployment.

---
