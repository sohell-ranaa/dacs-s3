---

# 2. INTEGRATED PERFORMANCE DISCUSSION

**Contributors:** Muhammad Usama Fazal (TP086008), Imran Shahadat Noble (TP087895), Md Sohel Rana (TP086217)

---

## 2.1 Experimental Setup

### 2.1.1 Dataset Description

The NSL-KDD dataset, developed by Tavallaee et al. (2009), represents a refined version of the original KDD Cup 1999 dataset. This improvement addresses critical issues present in the original dataset, including the removal of redundant records that caused classifiers to be biased toward frequent records, and the provision of reasonable record counts in both training and test sets (Dhanabal & Shantharajah, 2015).

**Table 4: Dataset Composition**

| Dataset | Total Records | Benign | DoS | Probe | R2L | U2R |
|:--------|-------------:|-------:|----:|------:|----:|----:|
| **Training (KDDTrain+)** | 63,280 | 33,672 (53.21%) | 23,066 (36.45%) | 5,911 (9.34%) | 575 (0.91%) | 56 (0.09%) |
| **Testing (KDDTest+)** | 22,544 | 9,711 (43.08%) | 7,458 (33.08%) | 2,421 (10.74%) | 2,754 (12.22%) | 200 (0.89%) |
| **Distribution Shift** | - | -10.13% | -3.37% | +1.40% | **+11.31%** | +0.80% |

Table 4 presents the complete dataset composition with distribution shift analysis. The most significant observation is the dramatic increase in R2L class representation from 0.91% in training to 12.22% in testing—a 12-fold increase. This intentional distribution shift tests each classifier's ability to generalise from limited training examples to unseen attack patterns, simulating real-world scenarios where new attack variants continuously emerge. The DoS class, while experiencing a slight decrease, remains the most prevalent attack type in both sets, reflecting real-world attack distributions where denial-of-service attacks dominate the threat landscape.

![Figure 2: Class Distribution Comparison](../../figures/fig_class_distribution.png)

*Figure 2: Class distribution comparison between training and testing datasets showing the significant distribution shift, particularly for R2L attacks. The bar chart visualises the percentage of each class in both datasets, highlighting the challenge posed by minority classes during model evaluation.*

Figure 2 illustrates the class distribution disparity between training and test sets. The visualisation clearly demonstrates the severe class imbalance that characterises network intrusion datasets. Benign traffic dominates both sets, followed by DoS attacks, while R2L and U2R represent extreme minority classes. This imbalance necessitates careful metric selection and evaluation strategies, as traditional accuracy measures can be misleading when a classifier simply predicts the majority class. The distribution shift for R2L is particularly notable, as classifiers must learn to detect this attack category from only 575 training examples while being evaluated on 2,754 test instances.

### 2.1.2 Data Preprocessing Pipeline

A standardised preprocessing pipeline was implemented across all three classifiers to ensure fair comparison and reproducibility. The pipeline consists of four main stages:

**Table 5: Data Preprocessing Summary**

| Stage | Operation | Input | Output | Rationale |
|:------|:----------|------:|-------:|:----------|
| **1. Loading** | CSV Import | Raw files | 63,280 + 22,544 records | Separate train/test preservation |
| **2. Encoding** | One-Hot Encoding | 41 features (3 categorical) | 122 features | Handle categorical variables |
| **3. Scaling** | MinMax Normalisation | 122 features | 122 features (0-1 range) | Distance-based algorithm compatibility |
| **4. Selection** | Feature Reduction | 122 features | 30-38 features | Remove noise, improve efficiency |

Table 5 summarises the preprocessing pipeline applied uniformly to all datasets. The categorical features (protocol_type, service, flag) were converted using one-hot encoding, expanding the feature space from 41 to 122 dimensions. MinMax normalisation was applied to ensure all features contribute equally to distance calculations, particularly important for KNN which is sensitive to feature scales. Feature selection, applied individually by each team member using algorithm-appropriate methods, reduced dimensionality by 69-75% while maintaining or improving classification performance.

### 2.1.3 Performance Metrics Selection

The choice of evaluation metrics significantly impacts the interpretation of classifier performance, particularly for imbalanced datasets (Powers, 2011). Traditional accuracy can be misleading when class distributions are skewed.

**Table 6: Selected Performance Metrics**

| Metric | Formula | Range | Interpretation | Suitability for Imbalanced Data |
|:-------|:--------|:------|:---------------|:--------------------------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | 0-1 | Overall correctness | Low - biased toward majority class |
| **F1-Score (Weighted)** | Σ(wi × F1i) | 0-1 | Harmonic mean, class-weighted | Medium - accounts for class frequency |
| **F1-Score (Macro)** | (1/n) × Σ(F1i) | 0-1 | Unweighted class average | Medium - equal class importance |
| **MCC** | Complex formula | -1 to 1 | Correlation coefficient | **High** - balanced for all confusion matrix quadrants |

Table 6 justifies our metric selection hierarchy. Matthews Correlation Coefficient (MCC) was chosen as the primary evaluation metric because it produces a high score only if the classifier performs well on all four confusion matrix categories (true positives, true negatives, false positives, false negatives), making it particularly suitable for imbalanced multi-class scenarios (Chicco & Jurman, 2020). Unlike accuracy, which can reach 53% by simply predicting all instances as "Benign," MCC returns a value near zero for such trivial predictions. F1-scores provide complementary perspectives: weighted F1 accounts for class frequency while macro F1 treats all classes equally regardless of size.

---

## 2.2 Comparative Analysis of Optimised Models

### 2.2.1 Overall Performance Comparison

After implementing baseline models and applying algorithm-specific optimisation strategies, all three classifiers were evaluated on the same test set using identical preprocessing.

**Table 7: Optimised Model Performance Metrics**

| Classifier | Author | Accuracy | F1 (Weighted) | F1 (Macro) | MCC | Improvement vs Baseline |
|:-----------|:-------|:--------:|:-------------:|:----------:|:---:|:-----------------------:|
| LDA (Linear) | Muhammad Usama Fazal (TP086008) | 0.7746 | 0.7632 | 0.6712 | 0.6712 | +1.2% |
| Random Forest (Ensemble) | Imran Shahadat Noble (TP087895) | 0.8680 | 0.8398 | 0.7784 | 0.8096 | +0.8% |
| **KNN (Non-Linear)** | **Md Sohel Rana (TP086217)** | **0.8752** | **0.8647** | **0.7703** | **0.8162** | **+3.1%** |

Table 7 presents the comprehensive performance metrics for all optimised classifiers. KNN achieves the highest overall MCC (0.8162) and accuracy (87.52%), demonstrating that a carefully tuned instance-based learner can outperform ensemble methods on this dataset. The performance hierarchy clearly shows non-linear methods (KNN: 0.8162, RF: 0.8096) substantially outperforming the linear classifier (LDA: 0.6712), with a 14.5 percentage point difference. This gap underscores that network intrusion patterns exhibit complex, non-linear relationships that cannot be captured by linear decision boundaries. KNN's superior performance validates the effectiveness of instance-based learning when the feature space contains distinct, separable clusters.

![Figure 3: Overall Model Performance Comparison](../../figures/fig1_model_comparison.png)

*Figure 3: Overall performance comparison of optimised classifiers across all metrics. The grouped bar chart shows KNN and Random Forest achieving comparable high performance, both significantly outperforming LDA across all evaluation metrics.*

Figure 3 visualises the performance comparison across all metrics. The chart reveals several important patterns: (1) KNN consistently outperforms other classifiers on accuracy and weighted F1, (2) Random Forest achieves the highest macro F1, suggesting better handling of minority classes, and (3) LDA shows consistent performance across metrics but at a lower absolute level. The close competition between KNN and Random Forest (within 1% MCC difference) indicates both approaches are viable for deployment, with selection depending on specific operational requirements such as prediction speed and interpretability needs.

![Figure 4: Radar Chart Multi-Metric Comparison](../../figures/fig2_radar_comparison.png)

*Figure 4: Radar chart providing a multi-dimensional view of classifier performance. Each axis represents a different evaluation metric, allowing visual comparison of classifier strengths and weaknesses across multiple criteria simultaneously.*

Figure 4 presents a radar chart view that enables holistic comparison across all performance dimensions. KNN (shown in green) covers the largest area, indicating the most balanced performance across metrics. Random Forest (blue) shows strength in precision-related metrics but slightly lower recall. LDA (red) demonstrates the smallest coverage area, reflecting its limitations in capturing non-linear patterns. The radar visualisation is particularly useful for identifying trade-offs: a classifier might excel in one metric while underperforming in another, and this chart makes such patterns immediately visible.

![Figure 5: Performance Heatmap](../../figures/fig3_performance_heatmap.png)

*Figure 5: Performance heatmap showing classifier performance across all metrics with colour intensity indicating score magnitude. Darker colours represent higher performance, enabling quick identification of strengths and weaknesses.*

Figure 5 provides a heatmap visualisation for intuitive performance comparison. The colour intensity directly corresponds to metric values, with darker shades indicating higher performance. This representation quickly reveals that KNN and Random Forest exhibit similarly dark colouration (high performance) across most metrics, while LDA shows consistently lighter shading. The heatmap format is particularly effective for identifying patterns: the row-wise consistency indicates that performance tends to be stable across metrics within each classifier, while column-wise variation reveals which metrics best differentiate classifier capabilities.

### 2.2.2 Per-Class Performance Analysis

Overall metrics provide aggregate views, but understanding class-specific performance is crucial for intrusion detection systems where different attack types pose varying security risks.

**Table 8: MCC Performance by Attack Category**

| Attack Class | LDA (Usama Fazal) | Random Forest (Imran Noble) | KNN (Sohel Rana) | Best Classifier | Security Implication |
|:-------------|:-----------------:|:---------------------------:|:----------------:|:---------------:|:---------------------|
| **Benign** | 0.673 | 0.757 | **0.786** | KNN | Critical for false positive reduction |
| **DoS** | 0.786 | **0.984** | 0.946 | RF | Service availability protection |
| **Probe** | 0.575 | **0.911** | 0.850 | RF | Early attack detection |
| **R2L** | 0.513 | 0.336 | **0.567** | KNN | Remote access prevention |
| **U2R** | 0.579 | **0.847** | 0.572 | RF | Privilege escalation prevention |

Table 8 reveals significant class-specific performance variations that have important implications for security operations. Random Forest demonstrates exceptional detection of DoS attacks (0.984 MCC), nearly perfect classification that is crucial for protecting service availability. RF also excels at Probe detection (0.911), enabling early warning of reconnaissance activities. However, RF struggles with R2L classification (0.336), likely due to overfitting to the sparse training examples. KNN shows the most balanced performance, achieving the best results for Benign classification (0.786, reducing false positives) and R2L detection (0.567, despite the severe distribution shift). The R2L class remains challenging for all classifiers, highlighting the difficulty of detecting remote access attacks from limited training data.

![Figure 6: MCC Per Attack Class Comparison](../../figures/fig4_mcc_per_class.png)

*Figure 6: Per-class MCC comparison showing each classifier's detection capability for different attack types. The grouped bar chart highlights Random Forest's dominance in DoS/Probe/U2R detection while KNN leads in Benign/R2L classification.*

Figure 6 visualises the class-specific performance patterns, making it immediately apparent that no single classifier dominates across all attack types. Random Forest's exceptional DoS and Probe detection (tall blue bars) contrasts with its poor R2L performance (short blue bar), while KNN shows more consistent performance across classes. This finding has practical implications: a hybrid ensemble approach combining Random Forest for high-volume attacks (DoS, Probe) with KNN for subtle intrusion patterns (R2L, Benign) could potentially achieve superior overall detection rates compared to any single classifier.

![Figure 7: MCC Per Class Detailed Comparison](../../figures/mcc_per_class_comparison.png)

*Figure 7: Detailed per-class MCC analysis with extended visualisation showing the performance gap between classifiers for each attack category. This granular view supports decision-making for deployment scenarios with specific attack priorities.*

Figure 7 provides an alternative visualisation of per-class performance with enhanced detail. The chart emphasises the magnitude of differences between classifiers for each attack type. For DoS detection, Random Forest achieves near-perfect classification (0.984) compared to LDA's 0.786—a difference of 0.198 MCC points that translates to significantly fewer missed denial-of-service attacks. The U2R class shows the largest performance gap between Random Forest (0.847) and other classifiers (both below 0.58), indicating that ensemble methods are particularly effective at detecting privilege escalation attacks through their ability to capture complex feature interactions.

### 2.2.3 Confusion Matrix Analysis

Confusion matrices provide detailed insight into classification patterns, revealing not just overall accuracy but the specific types of errors each classifier makes.

![Figure 8: Confusion Matrices for All Optimised Models](../../figures/confusion_matrices_all_models.png)

*Figure 8: Side-by-side confusion matrices for all three optimised classifiers showing classification patterns across all five classes. The matrices reveal specific misclassification patterns and highlight each classifier's strengths and weaknesses.*

Figure 8 presents the confusion matrices for all optimised models, enabling detailed error analysis. Several patterns emerge: (1) All classifiers show strong diagonal elements for Benign and DoS classes, indicating high accuracy for majority classes. (2) Random Forest's confusion matrix shows the darkest diagonal for DoS, confirming its exceptional detection capability. (3) The R2L column shows significant off-diagonal elements across all classifiers, indicating systematic misclassification. (4) U2R shows improvement with Random Forest but remains challenging due to having only 56 training examples. The confusion matrices also reveal false positive patterns: LDA shows more Benign traffic misclassified as DoS, which would trigger unnecessary security responses, while KNN maintains better separation.

---

## 2.3 Cross-Validation Results

Cross-validation provides insight into model stability and potential overfitting by evaluating performance across multiple data partitions.

**Table 9: Cross-Validation Results (F1-Weighted, 5-Fold)**

| Classifier | Author | CV Mean | CV Std | 95% Confidence Interval | Test F1 | CV-Test Gap |
|:-----------|:-------|:-------:|:------:|:-----------------------:|:-------:|:-----------:|
| LDA | Muhammad Usama Fazal | 0.9310 | 0.0026 | [0.9257, 0.9363] | 0.7632 | 0.1678 |
| **Random Forest** | **Imran Shahadat Noble** | **0.9963** | **0.0004** | [0.9954, 0.9972] | 0.8398 | 0.1565 |
| KNN | Md Sohel Rana | 0.9920 | 0.0011 | [0.9897, 0.9943] | 0.8647 | 0.1273 |

Table 9 presents cross-validation results alongside test set performance, revealing critical insights about model generalisation. Random Forest achieves the highest CV score (0.9963) with minimal variance (0.0004), indicating excellent performance on training data folds. However, all classifiers show substantial gaps between CV and test performance (12.7% to 16.8%), caused primarily by the intentional distribution shift in NSL-KDD. KNN shows the smallest CV-test gap (0.1273), suggesting its instance-based approach generalises better to the shifted test distribution. LDA's larger gap (0.1678) reflects the limitation of linear boundaries when attack patterns vary between training and test sets.

![Figure 9: Cross-Validation Box Plot](../../figures/cross_validation_boxplot.png)

*Figure 9: Box plot showing cross-validation F1-score distributions across 5 folds for each classifier. The tight distributions for Random Forest and KNN indicate stable performance, while the consistently high medians demonstrate strong learning capability.*

Figure 9 visualises the cross-validation score distributions, showing the consistency of each classifier across data folds. Random Forest displays an extremely tight box (nearly zero interquartile range), indicating that ensemble averaging effectively reduces variance across different data samples. KNN also shows a compact distribution, while LDA exhibits slightly more variability. The key insight from this visualisation is that all classifiers achieve stable, high performance on training data—the challenge lies in generalising to the shifted test distribution rather than in learning the training patterns.

![Figure 10: Cross-Validation Performance](../../figures/fig8_cross_validation.png)

*Figure 10: Alternative cross-validation visualisation showing mean scores with confidence intervals. This format emphasises the precision of performance estimates and enables statistical comparison between classifiers.*

Figure 10 provides an alternative representation with confidence intervals, enabling statistical comparison. The non-overlapping confidence intervals between LDA and the non-linear classifiers confirm that the performance difference is statistically significant. The overlapping intervals between Random Forest and KNN suggest their training performance is statistically indistinguishable, making test set performance the deciding factor for practical deployment decisions.

---

## 2.4 Training Efficiency Analysis

Computational efficiency is crucial for practical deployment, particularly in real-time network monitoring scenarios where prediction speed impacts system responsiveness.

**Table 10: Training and Prediction Efficiency**

| Classifier | Author | Training Time | Prediction Time (per sample) | Memory Usage | Scalability |
|:-----------|:-------|:-------------:|:---------------------------:|:------------:|:-----------:|
| LDA | Muhammad Usama Fazal | 0.52s | ~0.01ms | Low | Excellent |
| Random Forest | Imran Shahadat Noble | 8.74s | ~0.15ms | Medium | Good |
| KNN | Md Sohel Rana | 0.03s | ~2.50ms | High | Limited |

Table 10 compares computational efficiency across classifiers. LDA offers the fastest training (0.52s) and prediction times, making it suitable for resource-constrained environments despite lower accuracy. Random Forest requires substantial training time (8.74s) but provides reasonable prediction speed once trained. KNN's "lazy learning" approach results in near-instantaneous training (0.03s) but slow prediction as distances must be computed to all training instances. These trade-offs must be considered alongside accuracy when selecting classifiers for specific deployment scenarios.

![Figure 11: Training Time Comparison](../../figures/fig5_training_time.png)

*Figure 11: Training time comparison across classifiers showing the computational cost of model fitting. The logarithmic scale accommodates the large difference between KNN's near-instant training and Random Forest's more intensive fitting process.*

Figure 11 visualises the training time differences. The contrast is striking: Random Forest requires approximately 290 times longer to train than KNN (8.74s vs 0.03s). However, this comparison is somewhat misleading for real-time deployment where prediction time matters more than training time. KNN's computational cost is deferred to prediction time, while Random Forest and LDA complete most computation during training. For high-throughput network monitoring, LDA or Random Forest would process predictions faster despite longer initial training.

![Figure 12: Training Time Detailed Comparison](../../figures/training_time_comparison.png)

*Figure 12: Detailed training time analysis with breakdown by operation type. This visualisation helps identify computational bottlenecks and optimisation opportunities for each classifier.*

Figure 12 provides additional detail on training efficiency, supporting deployment planning decisions. The computational profile of each classifier suggests different optimal use cases: LDA for edge devices with limited resources, Random Forest for server-based monitoring systems where accuracy is paramount, and KNN for scenarios with stable, limited datasets where the training data fits in memory.

---

## 2.5 Feature Selection Impact

Feature selection reduces dimensionality, removes noise, and can improve both performance and computational efficiency.

**Table 11: Feature Reduction Summary**

| Classifier | Author | Original Features | Selected Features | Reduction | Method | Performance Impact |
|:-----------|:-------|:-----------------:|:-----------------:|:---------:|:-------|:------------------:|
| LDA | Muhammad Usama Fazal | 122 | 30 | 75.4% | Correlation Threshold (>0.1) | +1.2% MCC |
| Random Forest | Imran Shahadat Noble | 122 | 38 | 68.9% | Importance (95% cumulative) | +0.3% MCC |
| KNN | Md Sohel Rana | 122 | 30 | 75.4% | Correlation Threshold (>0.1) | +2.8% MCC |

Table 11 demonstrates that substantial dimensionality reduction (69-75%) improved or maintained performance across all classifiers. LDA and KNN used correlation-based selection, removing features with weak target correlation (<0.1 absolute value). Random Forest leveraged its built-in feature importance mechanism, selecting features contributing to 95% of cumulative importance. KNN benefited most from feature selection (+2.8% MCC), as removing irrelevant features reduced the curse of dimensionality that negatively impacts distance-based methods. The consistent effectiveness across different selection methods confirms that NSL-KDD contains significant feature redundancy.

![Figure 13: Feature Reduction Impact](../../figures/fig7_feature_reduction.png)

*Figure 13: Visualisation of feature reduction showing the number of selected features versus original features for each classifier. The substantial reduction without performance loss indicates significant redundancy in the original feature set.*

Figure 13 illustrates the dramatic feature reduction achieved. All classifiers reduced features by at least 68%, yet none experienced performance degradation. This finding has important implications: (1) simpler models with fewer features are often equally or more effective, (2) feature selection should be a standard preprocessing step for network intrusion detection, and (3) the NSL-KDD dataset's 122 features (after one-hot encoding) contain substantial redundancy that, if not addressed, can harm classifier performance through noise introduction.

---

## 2.6 Baseline vs Optimised Comparison

Comparing baseline (default parameters) and optimised configurations quantifies the value of hyperparameter tuning and feature selection.

**Table 12: Baseline vs Optimised Performance**

| Classifier | Author | Baseline MCC | Optimised MCC | Absolute Improvement | Relative Improvement |
|:-----------|:-------|:------------:|:-------------:|:--------------------:|:--------------------:|
| LDA | Muhammad Usama Fazal | 0.6631 | 0.6712 | +0.0081 | +1.2% |
| Random Forest | Imran Shahadat Noble | 0.8033 | 0.8096 | +0.0063 | +0.8% |
| KNN | Md Sohel Rana | 0.7916 | 0.8162 | +0.0246 | **+3.1%** |

Table 12 quantifies optimisation benefits. KNN achieved the largest improvement (+3.1%), primarily from the combination of optimal k selection (k=3), distance metric tuning (Manhattan), and weighted voting. Random Forest showed modest improvement (+0.8%), as the default parameters already perform well for this ensemble method. LDA's limited improvement (+1.2%) reflects its constrained optimisation space—linear boundaries cannot be fundamentally improved through parameter tuning when the underlying data patterns are non-linear.

![Figure 14: Baseline vs Optimised MCC Comparison](../../figures/baseline_vs_optimised_mcc.png)

*Figure 14: Side-by-side comparison of baseline and optimised MCC scores for each classifier. The improvement bars highlight the value of hyperparameter tuning, with KNN showing the most significant gain.*

Figure 14 visualises the optimisation gains, clearly showing that hyperparameter tuning provides meaningful but modest improvements across all classifiers. The visualisation emphasises that algorithm selection (linear vs non-linear) has a larger impact than parameter tuning—even the optimised LDA (0.6712) underperforms the baseline KNN (0.7916) by 12 percentage points, highlighting the fundamental importance of choosing appropriate algorithmic approaches for the problem characteristics.

![Figure 15: Baseline vs Optimised Detailed View](../../figures/fig6_baseline_vs_optimized.png)

*Figure 15: Detailed baseline versus optimised comparison showing all metrics. This comprehensive view confirms that optimisation benefits are consistent across evaluation metrics, not just MCC.*

Figure 15 extends the comparison across all metrics, confirming that improvements are consistent rather than metric-specific. This consistency suggests that optimisation genuinely improved classification capability rather than gaming specific metrics through threshold adjustment or similar techniques.

---

## 2.7 Final Classifier Ranking

Based on comprehensive evaluation across accuracy, F1-scores, MCC, cross-validation stability, class-specific performance, and computational efficiency, the final classifier ranking is established.

**Table 13: Final Classifier Ranking**

| Rank | Classifier | Author | MCC | Accuracy | Primary Strengths | Recommended Use Case |
|:----:|:-----------|:-------|:---:|:--------:|:------------------|:---------------------|
| **1st** | **KNN** | **Md Sohel Rana (TP086217)** | **0.816** | **87.5%** | Best overall MCC, highest accuracy, best generalisation | General-purpose IDS deployment |
| 2nd | Random Forest | Imran Shahadat Noble (TP087895) | 0.810 | 86.8% | Best DoS/Probe/U2R detection, most stable CV | High-volume attack detection |
| 3rd | LDA | Muhammad Usama Fazal (TP086008) | 0.671 | 77.5% | Fastest prediction, most interpretable | Resource-constrained, explainable AI |

Table 13 presents the final ranking with deployment recommendations. KNN emerges as the overall winner with the highest MCC (0.816) and best test set accuracy (87.5%), demonstrating that a well-tuned instance-based approach excels at capturing the local patterns inherent in network intrusion data. Random Forest secures second place with strong performance (0.810 MCC) and particular strengths in detecting high-impact attack types (DoS, U2R). LDA, while third in accuracy, offers unique advantages in speed and interpretability that may be valuable in specific deployment contexts.

![Figure 16: Final Classifier Ranking Podium](../../figures/fig9_ranking_podium.png)

*Figure 16: Visual representation of final classifier ranking based on overall MCC performance. The podium format emphasises the competitive nature of the evaluation while clearly indicating the performance hierarchy.*

Figure 16 provides a visual summary of the final ranking. The podium representation emphasises that while KNN achieved the top position, the competition was close—Random Forest trails by only 0.006 MCC points. This marginal difference suggests that both classifiers are viable for production deployment, with the choice depending on specific operational requirements. The substantial gap to LDA (0.145 MCC points) confirms that non-linear methods are essential for effective multi-class intrusion detection.

---

## 2.8 Key Findings and Recommendations

### 2.8.1 Summary of Key Findings

1. **Non-linear methods significantly outperform linear approaches:** The 14.5% MCC gap between KNN/Random Forest and LDA demonstrates that network intrusion patterns exhibit complex, non-linear relationships requiring sophisticated decision boundaries.

2. **Instance-based learning excels at generalisation:** KNN's superior test performance despite slightly lower CV scores indicates that instance-based approaches adapt better to distribution shift—a common characteristic of real-world intrusion detection scenarios.

3. **No single classifier dominates all attack types:** Random Forest excels at DoS/Probe/U2R while KNN leads in Benign/R2L classification, suggesting potential for hybrid ensemble approaches.

4. **Feature selection consistently improves performance:** 69-75% dimensionality reduction improved all classifiers, confirming significant redundancy in network traffic features.

5. **Cross-validation alone is insufficient for evaluation:** All classifiers showed substantial CV-test gaps due to distribution shift, emphasising the importance of held-out test evaluation.

### 2.8.2 Deployment Recommendations

**For Maximum Detection Accuracy:**
- Deploy KNN (k=3, Manhattan distance, distance-weighted voting)
- Apply correlation-based feature selection (30 features)
- Consider prediction caching for high-throughput scenarios
- Trade-off: Higher memory requirements and prediction latency

**For Real-Time High-Volume Monitoring:**
- Deploy Random Forest (100 trees, balanced class weights, entropy criterion)
- Apply importance-based feature selection (38 features)
- Leverage parallel prediction capabilities
- Trade-off: Slightly lower R2L detection, longer initial training

**For Resource-Constrained or Explainable AI Requirements:**
- Deploy LDA with correlation-based feature selection
- Accept reduced detection accuracy for faster prediction
- Leverage linear coefficients for decision explanation
- Trade-off: Significantly lower overall detection capability

### 2.8.3 Conclusion

This comprehensive evaluation demonstrates that **KNN (Md Sohel Rana - TP086217) achieves the best overall performance** with MCC 0.816 and 87.5% accuracy, making it the recommended classifier for general-purpose network intrusion detection deployment. **Random Forest (Imran Shahadat Noble - TP087895)** provides a strong alternative with superior detection of high-impact attack types. The substantial performance gap compared to **LDA (Muhammad Usama Fazal - TP086008)** confirms that non-linear approaches are essential for effective multi-class intrusion detection in modern cybersecurity environments.

<div style="page-break-after: always;"></div>

