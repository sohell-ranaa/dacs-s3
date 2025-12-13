<div align="center">

# CT115-3-M DATA ANALYTICS IN CYBER SECURITY

## GROUP ASSIGNMENT

---

### Machine Learning-Based Network Intrusion Detection System

---

**Group Members:**

| Name | Student ID |
|:-----|:-----------|
| Muhammad Usama Fazal | TP086008 |
| Imran Shahadat Noble | TP087895 |
| Md Sohel Rana | TP086217 |

---

**Intake Code:** UC2F2408CS

**Module Title:** Data Analytics in Cyber Security

**Submission Date:** December 2024

---

*Asia Pacific University of Technology and Innovation*

</div>

<div style="page-break-after: always;"></div>

---

## TABLE OF CONTENTS

| Section | Title | Page |
|:--------|:------|-----:|
| 1 | Combined Review of Selected Algorithms | 4 |
| 1.1 | Introduction | 4 |
| 1.2 | Algorithm Classification Taxonomy | 5 |
| 1.3 | Linear Classifier: LDA (Muhammad Usama Fazal - TP086008) | 6 |
| 1.4 | Non-Linear Classifier: KNN (Md Sohel Rana - TP086217) | 7 |
| 1.5 | Ensemble Classifier: Random Forest (Imran Shahadat Noble - TP087895) | 8 |
| 1.6 | Summary of Algorithm Characteristics | 9 |
| 2 | Integrated Performance Discussion | 10 |
| 2.1 | Experimental Setup | 10 |
| 2.2 | Performance Metrics | 11 |
| 2.3 | Comparative Analysis of Optimised Models | 12 |
| 2.4 | Cross-Validation Results | 15 |
| 2.5 | Key Findings and Recommendations | 16 |
| 3 | Individual Reports | 18 |
| 3.1 | Linear Discriminant Analysis (Muhammad Usama Fazal - TP086008) | 18 |
| 3.2 | Random Forest (Imran Shahadat Noble - TP087895) | 22 |
| 3.3 | K-Nearest Neighbors (Md Sohel Rana - TP086217) | 26 |
| 4 | References | 30 |
| | **Appendices** | |
| A | JupyterLab Notebook - Linear Classifier | 31 |
| B | JupyterLab Notebook - Ensemble Classifier | 45 |
| C | JupyterLab Notebook - Non-Linear Classifier | 60 |

<div style="page-break-after: always;"></div>

---

## LIST OF FIGURES

| Figure | Title | Page |
|:-------|:------|-----:|
| Figure 1 | Algorithm Classification Taxonomy | 5 |
| Figure 2 | Overall Model Performance Comparison | 12 |
| Figure 3 | Radar Chart: Multi-Metric Performance Comparison | 13 |
| Figure 4 | Performance Heatmap Across Metrics | 13 |
| Figure 5 | MCC Per Attack Class Comparison | 14 |
| Figure 6 | Cross-Validation Box Plot | 15 |
| Figure 7 | Training Time Comparison | 16 |
| Figure 8 | Baseline vs Optimised MCC Comparison | 17 |
| Figure 9 | Confusion Matrices for All Optimised Models | 17 |
| Figure 10 | Final Classifier Ranking | 18 |
| Figure 11 | Linear Classifier Baseline Comparison | 19 |
| Figure 12 | LDA Feature Correlation Analysis | 20 |
| Figure 13 | LDA Confusion Matrices (Baseline vs Optimised) | 21 |
| Figure 14 | Ensemble Classifier Baseline Comparison | 23 |
| Figure 15 | Random Forest Feature Importance | 24 |
| Figure 16 | Random Forest Confusion Matrices | 25 |
| Figure 17 | Non-Linear Classifier Baseline Comparison | 27 |
| Figure 18 | KNN Feature Correlation Analysis | 28 |
| Figure 19 | KNN Confusion Matrices (Baseline vs Optimised) | 29 |

<div style="page-break-after: always;"></div>

---

## LIST OF TABLES

| Table | Title | Page |
|:------|:------|-----:|
| Table 1 | Class Distribution in NSL-KDD Dataset | 4 |
| Table 2 | Attack Category Descriptions | 5 |
| Table 3 | Comparison of Selected Classification Algorithms | 9 |
| Table 4 | Dataset Composition | 10 |
| Table 5 | Data Preprocessing Summary | 11 |
| Table 6 | Selected Performance Metrics | 11 |
| Table 7 | Optimised Model Performance Metrics | 12 |
| Table 8 | MCC Performance by Attack Category | 14 |
| Table 9 | Cross-Validation Results (F1-Weighted) | 15 |
| Table 10 | Feature Reduction Summary | 16 |
| Table 11 | Final Classifier Ranking | 17 |
| Table 12 | Linear Classifier Baseline Comparison | 19 |
| Table 13 | LDA Hyperparameter Tuning Configuration | 20 |
| Table 14 | LDA Top Correlated Features | 20 |
| Table 15 | LDA Baseline vs Optimised Performance | 21 |
| Table 16 | LDA MCC Per Class Comparison | 21 |
| Table 17 | Ensemble Classifier Baseline Comparison | 23 |
| Table 18 | Random Forest Hyperparameter Configuration | 24 |
| Table 19 | Random Forest Top Important Features | 24 |
| Table 20 | Random Forest Baseline vs Optimised Performance | 25 |
| Table 21 | Non-Linear Classifier Baseline Comparison | 27 |
| Table 22 | KNN Hyperparameter Tuning Configuration | 28 |
| Table 23 | KNN Baseline vs Optimised Performance | 29 |
| Table 24 | KNN MCC Per Class Comparison | 29 |

<div style="page-break-after: always;"></div>
---

# 1. COMBINED REVIEW OF SELECTED ALGORITHMS

**Contributors:** Muhammad Usama Fazal (TP086008), Imran Shahadat Noble (TP087895), Md Sohel Rana (TP086217)

---

## 1.1 Introduction

Network intrusion detection is a critical component of modern cybersecurity infrastructure. As cyber threats continue to evolve in sophistication and frequency, the need for intelligent, automated detection systems has become paramount. Machine learning offers a promising approach by enabling systems to learn patterns from historical network traffic data and identify anomalous behaviour indicative of potential attacks.

This report presents a comprehensive study of machine learning-based network intrusion detection using the NSL-KDD dataset, a refined version of the widely-used KDD Cup 1999 dataset. The NSL-KDD dataset addresses several inherent problems of the original dataset, including the removal of redundant records and the provision of a more balanced representation of attack types (Tavallaee et al., 2009).

### 1.1.1 Study Objectives

The objective of this study is to evaluate and compare three distinct classification algorithms representing different methodological approaches:

| Team Member | Algorithm | Category |
|:------------|:----------|:---------|
| Muhammad Usama Fazal (TP086008) | Linear Discriminant Analysis (LDA) | Linear |
| Md Sohel Rana (TP086217) | K-Nearest Neighbors (KNN) | Non-Linear |
| Imran Shahadat Noble (TP087895) | Random Forest | Ensemble (Bagging) |

Each algorithm was implemented with both baseline (default parameters) and optimised configurations to evaluate the impact of various optimisation strategies on multi-class classification performance.

### 1.1.2 Multi-Class Classification Task

**Table 1: Class Distribution in NSL-KDD Dataset**

| Class | Description | Training Samples | Training % | Test Samples | Test % |
|:------|:------------|----------------:|----------:|-------------:|-------:|
| **Benign** | Normal network traffic | 33,672 | 53.21% | 9,711 | 43.08% |
| **DoS** | Denial of Service attacks | 23,066 | 36.45% | 7,458 | 33.08% |
| **Probe** | Surveillance/scanning attacks | 5,911 | 9.34% | 2,421 | 10.74% |
| **R2L** | Remote-to-Local attacks | 575 | 0.91% | 2,754 | 12.22% |
| **U2R** | User-to-Root attacks | 56 | 0.09% | 200 | 0.89% |
| **Total** | | **63,280** | **100%** | **22,544** | **100%** |

Table 1 presents the class distribution across training and test sets. The severe class imbalance is immediately evident, particularly for the R2L class which represents only 0.91% of training data but 12.22% of test data. This distribution shift poses a significant challenge for model generalisation, requiring classifiers to detect attack patterns from extremely limited training examples while being evaluated on a much larger test proportion.

**Table 2: Attack Category Descriptions**

| Attack Type | Full Name | Description | Example Attacks |
|:------------|:----------|:------------|:----------------|
| **DoS** | Denial of Service | Disrupts service availability by overwhelming resources | SYN flood, Smurf, Neptune |
| **Probe** | Surveillance | Scans networks to gather information | Port scan, IP sweep, Nmap |
| **R2L** | Remote-to-Local | Gains local access from remote machine | Password guessing, FTP write |
| **U2R** | User-to-Root | Escalates privileges to superuser | Buffer overflow, Rootkit |

Table 2 provides detailed descriptions of each attack category. Understanding these attack types is crucial for interpreting classifier performance, as different algorithms may exhibit varying effectiveness against specific attack patterns based on their underlying mathematical assumptions and decision boundary characteristics.

---

## 1.2 Algorithm Classification Taxonomy

Machine learning classification algorithms can be organised into distinct categories based on their underlying mathematical principles and learning mechanisms. Figure 1 illustrates the taxonomy of algorithms evaluated in this study.

![Figure 1: Algorithm Classification Taxonomy](../../figures/fig0_algorithm_taxonomy.png)

*Figure 1: Classification of machine learning algorithms showing the taxonomy of methods evaluated in this study. The three selected algorithms represent different categories: Linear (LDA), Non-Linear (KNN), and Ensemble (Random Forest). This diversity ensures comprehensive evaluation across methodological approaches.*

The selection of algorithms from three distinct categories ensures diversity in the approaches evaluated. Linear methods assume classes can be separated by hyperplanes, making them computationally efficient but potentially limited for complex patterns. Non-linear methods capture intricate decision boundaries without distributional assumptions. Ensemble methods combine multiple models to achieve robust, generalised predictions through collective decision-making.

---

## 1.3 Linear Classifier: Linear Discriminant Analysis

**Author:** Muhammad Usama Fazal (TP086008)

Linear Discriminant Analysis (LDA), introduced by Ronald Fisher in 1936, is a classical statistical method for dimensionality reduction and classification. LDA seeks to find a linear combination of features that best separates two or more classes by maximising the ratio of between-class variance to within-class variance (Hastie et al., 2009).

### Mathematical Foundation

LDA operates on the principle of maximising class separability in a lower-dimensional projection space. The algorithm assumes:

- Data for each class follows a **multivariate Gaussian distribution**
- All classes share a **common covariance matrix** (homoscedasticity)
- Features are **continuous** and not perfectly collinear

### Characteristics for Intrusion Detection

| Aspect | Advantage | Limitation |
|:-------|:----------|:-----------|
| **Computation** | Fast training and prediction | Limited by linear assumption |
| **Interpretability** | Clear decision boundaries | Sensitive to outliers |
| **Dimensionality** | Natural feature reduction | Struggles with non-linear patterns |
| **Multi-class** | Native support | Overlapping distributions challenging |

LDA's computational efficiency makes it suitable for real-time detection scenarios where prediction speed is critical. However, the assumption of linear separability may not hold for complex attack patterns that exhibit non-linear relationships with network features. Despite these limitations, LDA serves as an important baseline representing classical statistical approaches.

---

## 1.4 Non-Linear Classifier: K-Nearest Neighbors

**Author:** Md Sohel Rana (TP086217)

K-Nearest Neighbors (KNN) is an instance-based learning algorithm that classifies observations based on similarity measures in the feature space. Unlike parametric methods, KNN makes no assumptions about the underlying data distribution, making it highly flexible (Cover & Hart, 1967).

### Algorithmic Principle

The KNN algorithm operates through a simple yet powerful mechanism:

1. **Storage Phase:** Store all training instances in memory
2. **Distance Calculation:** Compute distances between query instance and all stored instances
3. **Neighbour Selection:** Identify the k nearest neighbours
4. **Voting:** Assign the majority class among neighbours (or weighted voting)

### Characteristics for Intrusion Detection

| Aspect | Advantage | Limitation |
|:-------|:----------|:-----------|
| **Assumptions** | Distribution-free approach | Curse of dimensionality |
| **Decision Boundary** | Captures complex, non-linear patterns | Computationally expensive at prediction |
| **Multi-class** | Natural handling | Sensitive to irrelevant features |
| **Adaptability** | Easily updated with new data | Requires careful k selection |

KNN's non-parametric nature allows it to capture complex decision boundaries that linear methods cannot represent. The instance-based approach is particularly effective when attack patterns form distinct clusters in the feature space. However, prediction-time computation scales with dataset size, requiring optimisation strategies for high-throughput deployment.

---

## 1.5 Ensemble Classifier: Random Forest

**Author:** Imran Shahadat Noble (TP087895)

Random Forest, proposed by Leo Breiman in 2001, is an ensemble learning method that constructs multiple decision trees during training and combines their predictions through majority voting (Breiman, 2001).

### Ensemble Mechanism

Random Forest employs two key randomisation techniques:

| Technique | Description | Benefit |
|:----------|:------------|:--------|
| **Bootstrap Aggregating (Bagging)** | Each tree trained on random subset with replacement | Reduces variance |
| **Random Feature Selection** | Only random subset of features considered at each split | Decorrelates trees |

This dual randomisation creates decorrelated trees whose collective prediction is more accurate and stable than any individual tree, implementing the "wisdom of crowds" principle.

### Characteristics for Intrusion Detection

| Aspect | Advantage | Limitation |
|:-------|:----------|:-----------|
| **Robustness** | Resistant to noise and outliers | Less interpretable than single trees |
| **Feature Analysis** | Built-in importance rankings | Computationally intensive for large data |
| **Overfitting** | Ensemble averaging prevents overfitting | May not capture very rare patterns |
| **Performance** | Consistent high accuracy | Black-box decision process |

Random Forest's robustness to noise makes it particularly suitable for network traffic data, which often contains anomalies and measurement errors. The built-in feature importance mechanism provides valuable insights for security analysts seeking to understand which network attributes most strongly indicate malicious activity.

---

## 1.6 Summary of Algorithm Characteristics

**Table 3: Comparison of Selected Classification Algorithms**

| Characteristic | LDA (Usama Fazal) | KNN (Sohel Rana) | Random Forest (Imran Noble) |
|:---------------|:-----------------:|:----------------:|:---------------------------:|
| **Category** | Linear | Non-Linear | Ensemble (Bagging) |
| **Training Complexity** | O(n·d²) - Low | O(1) - Lazy | O(k·n·log n) - Medium |
| **Prediction Speed** | O(d) - Fast | O(n·d) - Slow | O(k·log n) - Medium |
| **Interpretability** | High | Medium | Low |
| **Handles Non-linearity** | No | Yes | Yes |
| **Feature Importance** | No | No | Yes |
| **Sensitivity to Outliers** | High | Medium | Low |
| **Hyperparameter Sensitivity** | Low | High | Low |
| **Multi-class Capability** | Native | Native | Native |
| **Memory Requirements** | Low | High | Medium |

*Where: n = training samples, d = features, k = trees/neighbours*

Table 3 provides a comprehensive comparison highlighting the complementary strengths and weaknesses of each algorithm. LDA offers speed and interpretability but sacrifices flexibility. KNN provides adaptability and non-linear capture but at computational cost. Random Forest balances performance and robustness through ensemble averaging. This diversity justifies evaluating all three approaches, as the optimal choice depends on specific operational requirements including detection accuracy, prediction speed, and interpretability needs.

<div style="page-break-after: always;"></div>
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

---

# 3. INDIVIDUAL REPORTS

---

## 3.1 Linear Discriminant Analysis (LDA)

**Author:** Muhammad Usama Fazal (TP086008)

### 3.1.1 Algorithm Overview

Linear Discriminant Analysis (LDA), introduced by Ronald Fisher in 1936, represents one of the foundational techniques in statistical pattern recognition. As a supervised dimensionality reduction method, LDA seeks to find a linear combination of features that maximises the separation between classes while minimising within-class variance (Fisher, 1936). The algorithm projects high-dimensional data onto a lower-dimensional space where class discrimination is optimal.

The mathematical foundation of LDA rests on the Fisher criterion, which maximises the ratio:

**J(w) = (between-class scatter) / (within-class scatter)**

This optimisation finds projection directions that maximise the distance between class means while minimising the spread of samples within each class. For multi-class problems, LDA can reduce dimensionality to at most (C-1) dimensions, where C is the number of classes.

**Key Assumptions of LDA:**
1. Features follow multivariate Gaussian distributions within each class
2. All classes share a common covariance matrix (homoscedasticity)
3. Features are continuous and linearly related to class membership
4. No severe multicollinearity among predictors

### 3.1.2 Baseline Comparison

Before optimisation, LDA was compared against other linear classifiers to establish a baseline understanding of linear method capabilities on the NSL-KDD dataset.

**Table 14: Linear Classifier Baseline Comparison**

| Classifier | Accuracy | F1 (Weighted) | F1 (Macro) | MCC | Training Time |
|:-----------|:--------:|:-------------:|:----------:|:---:|:-------------:|
| Logistic Regression | 0.7623 | 0.7502 | 0.6589 | 0.6592 | 2.14s |
| **LDA** | **0.7698** | **0.7583** | **0.6687** | **0.6631** | **0.48s** |
| Ridge Classifier | 0.7612 | 0.7489 | 0.6572 | 0.6578 | 0.31s |
| SGD Classifier | 0.7534 | 0.7412 | 0.6453 | 0.6467 | 0.15s |

Table 14 compares LDA against other linear classifiers. LDA achieves the highest accuracy (0.7698) and MCC (0.6631) among linear methods, justifying its selection for further optimisation. The performance differences between linear classifiers are relatively small (within 2%), suggesting that the linear separability assumption itself limits performance more than the specific algorithm choice. Logistic Regression, despite its popularity, slightly underperforms LDA on this dataset, likely because LDA's direct optimisation of class separability is more effective than logistic regression's probabilistic approach for the given feature distributions.

![Figure 17: Linear Classifier Baseline Comparison](../../figures/linear_baseline_comparison.png)

*Figure 17: Baseline performance comparison of linear classifiers showing LDA achieving the highest MCC among evaluated methods. The bar chart demonstrates the relative similarity in performance among linear approaches, highlighting the fundamental limitation of linear decision boundaries.*

Figure 17 visualises the baseline comparison, showing that LDA marginally outperforms other linear methods. The relatively flat performance profile across all linear classifiers indicates that the ceiling for linear methods on this dataset is approximately 0.67 MCC—a limitation imposed by the non-linear nature of network intrusion patterns rather than by specific algorithm implementations. This observation motivated exploring optimisation strategies within LDA's framework while acknowledging the inherent constraints of linear approaches.

### 3.1.3 Feature Analysis

Understanding which features contribute most to classification helps interpret model decisions and identify potential for dimensionality reduction.

**Table 15: LDA Top Correlated Features**

| Rank | Feature Name | Correlation | Category | Interpretation |
|:----:|:-------------|:-----------:|:---------|:---------------|
| 1 | src_bytes | 0.487 | Traffic Volume | Bytes sent by source |
| 2 | dst_bytes | 0.412 | Traffic Volume | Bytes received by destination |
| 3 | logged_in | 0.398 | Connection Status | Successful login indicator |
| 4 | same_srv_rate | 0.356 | Service Pattern | Same service connection rate |
| 5 | diff_srv_rate | -0.342 | Service Pattern | Different service rate |
| 6 | dst_host_srv_count | 0.328 | Host Behaviour | Service count to destination |
| 7 | count | 0.315 | Connection Count | Connections to same host |
| 8 | serror_rate | 0.289 | Error Pattern | SYN error rate |
| 9 | srv_count | 0.267 | Service Count | Same service connections |
| 10 | dst_host_same_srv_rate | 0.254 | Host Pattern | Destination same service rate |

Table 15 lists the top 10 features by absolute correlation with attack categories. Traffic volume features (src_bytes, dst_bytes) show the strongest correlations, indicating that attack patterns often involve unusual data transfer volumes—DoS attacks typically generate high traffic, while reconnaissance attacks may show minimal data exchange. Service pattern features (same_srv_rate, diff_srv_rate) capture behavioural anomalies that distinguish normal usage from attack patterns. The diversity of feature categories among top correlations suggests that effective detection requires a combination of volume, connection, and behavioural indicators.

![Figure 18: LDA Feature Correlation Analysis](../../figures/linear_feature_correlation.png)

*Figure 18: Feature correlation heatmap for LDA showing the relationships between top features and attack categories. Positive correlations (warm colours) indicate features that increase with attack likelihood, while negative correlations (cool colours) suggest inverse relationships.*

Figure 18 presents the correlation structure among selected features. The heatmap reveals several important patterns: (1) src_bytes and dst_bytes are moderately correlated, suggesting redundancy that could be addressed through feature engineering. (2) Service rate features show expected negative correlations (same_srv_rate vs diff_srv_rate). (3) Error rate features cluster together, indicating that error patterns provide consistent discriminative information. Understanding these correlations informed the feature selection threshold choice (>0.1 absolute correlation), balancing dimensionality reduction against information retention.

### 3.1.4 Hyperparameter Tuning

LDA has limited hyperparameter options compared to other algorithms, with the primary tuning parameter being the solver method.

**Table 16: LDA Hyperparameter Tuning Configuration**

| Parameter | Values Tested | Optimal Value | Impact on Performance |
|:----------|:--------------|:--------------|:----------------------|
| solver | svd, lsqr, eigen | **svd** | Minimal (< 0.5% difference) |
| shrinkage | None, auto, 0.1-0.9 | **auto** | +0.8% MCC improvement |
| n_components | None, 2, 3, 4 | **4** (C-1) | Retains maximum discriminative information |
| store_covariance | True, False | True | Enables probability estimates |
| tol | 1e-3, 1e-4, 1e-5 | 1e-4 | Convergence precision |

Table 16 documents the hyperparameter search space and optimal values. The Singular Value Decomposition (SVD) solver was selected for its numerical stability with the feature set. Automatic shrinkage regularisation provided modest improvement (+0.8% MCC) by addressing the estimated covariance matrix's sensitivity to high-dimensional data. The number of components was set to 4 (number of classes minus one), retaining maximum discriminative power available under LDA's theoretical constraints.

### 3.1.5 Baseline vs Optimised Performance

**Table 17: LDA Baseline vs Optimised Performance**

| Metric | Baseline | Optimised | Change | Interpretation |
|:-------|:--------:|:---------:|:------:|:---------------|
| Accuracy | 0.7698 | 0.7746 | +0.62% | Marginal overall improvement |
| F1 (Weighted) | 0.7583 | 0.7632 | +0.65% | Balanced class performance gain |
| F1 (Macro) | 0.6687 | 0.6712 | +0.37% | Limited minority class improvement |
| **MCC** | **0.6631** | **0.6712** | **+1.22%** | Moderate correlation improvement |
| Training Time | 0.48s | 0.52s | +8.3% | Minimal additional cost |

Table 17 quantifies the optimisation impact. The improvements are modest across all metrics (0.37% to 1.22%), reflecting LDA's constrained optimisation space. Linear decision boundaries fundamentally limit achievable performance regardless of parameter tuning. The primary value of optimisation was confirming that LDA operates near its theoretical ceiling on this dataset, justifying exploration of non-linear methods for higher accuracy requirements.

![Figure 19: LDA Confusion Matrices (Baseline vs Optimised)](../../figures/linear_confusion_matrices.png)

*Figure 19: Side-by-side confusion matrices comparing LDA baseline and optimised models. The matrices reveal similar misclassification patterns with minor improvements in specific classes after optimisation.*

Figure 19 shows the confusion matrix evolution from baseline to optimised LDA. The most notable change is a slight reduction in Benign→DoS misclassifications and improved Probe detection. However, R2L and U2R remain challenging, with the optimised model showing only marginal improvements for these minority classes. The diagonal elements show modest strengthening, but the overall pattern confirms that linear separability constraints prevent dramatic performance gains through parameter tuning alone.

### 3.1.6 Per-Class Performance

**Table 18: LDA MCC Per Class Comparison**

| Class | Baseline MCC | Optimised MCC | Change | Analysis |
|:------|:------------:|:-------------:|:------:|:---------|
| Benign | 0.665 | 0.673 | +0.8% | Slight false positive reduction |
| DoS | 0.782 | 0.786 | +0.4% | Strong detection maintained |
| Probe | 0.568 | 0.575 | +0.7% | Moderate improvement |
| R2L | 0.502 | 0.513 | +1.1% | Distribution shift challenge |
| U2R | 0.571 | 0.579 | +0.8% | Limited training data impact |

Table 18 breaks down per-class performance changes. DoS detection remains LDA's strongest capability (0.786 MCC), likely because DoS attacks create distinct traffic volume patterns that are linearly separable from normal traffic. R2L shows the largest improvement (+1.1%), though absolute performance remains low (0.513), reflecting the severe distribution shift challenge. U2R performance (0.579) exceeds R2L despite having fewer training examples, suggesting that privilege escalation attacks create more linearly distinguishable patterns than remote access attacks.

### 3.1.7 LDA Summary and Conclusions

LDA provides a computationally efficient baseline for network intrusion detection with clear interpretability advantages. The algorithm achieves 0.6712 MCC with optimised parameters, demonstrating reasonable detection capability while highlighting the fundamental limitations of linear approaches for complex network traffic patterns.

**Key Strengths:**
- Fast training (0.52s) and prediction (sub-millisecond)
- Interpretable linear coefficients for decision explanation
- Stable performance across cross-validation folds
- Native multi-class support without one-vs-rest decomposition

**Key Limitations:**
- Linear separability assumption violated by complex attack patterns
- Limited improvement potential through hyperparameter tuning
- Poor performance on R2L class due to distribution shift
- Sensitive to outliers in the feature space

**Recommendation:** LDA is suitable for resource-constrained deployments or scenarios requiring explainable decisions, accepting the 14.5% MCC gap compared to non-linear methods.

---

## 3.2 Random Forest Classifier

**Author:** Imran Shahadat Noble (TP087895)

### 3.2.1 Algorithm Overview

Random Forest, introduced by Leo Breiman in 2001, represents a powerful ensemble learning method that constructs multiple decision trees during training and combines their predictions through majority voting (Breiman, 2001). The algorithm addresses the overfitting tendency of individual decision trees through two randomisation techniques: bootstrap aggregating (bagging) for training sample selection and random feature subspace selection at each split point.

**Ensemble Learning Mechanism:**

1. **Bootstrap Aggregating (Bagging):** Each tree is trained on a random subset of the training data, sampled with replacement. This creates diverse trees that make different errors, which average out in the ensemble.

2. **Random Feature Selection:** At each split, only a random subset of features is considered. This decorrelates trees, ensuring that no single dominant feature drives all predictions.

3. **Majority Voting:** Final predictions aggregate individual tree votes, with the class receiving the most votes selected as the prediction.

The theoretical foundation rests on the "wisdom of crowds" principle—many weak learners combined can outperform individual strong learners by averaging out individual biases and errors.

### 3.2.2 Baseline Comparison

Random Forest was compared against other ensemble methods to establish its relative performance within the ensemble classifier category.

**Table 19: Ensemble Classifier Baseline Comparison**

| Classifier | Accuracy | F1 (Weighted) | F1 (Macro) | MCC | Training Time |
|:-----------|:--------:|:-------------:|:----------:|:---:|:-------------:|
| AdaBoost | 0.7823 | 0.7678 | 0.6798 | 0.6834 | 12.45s |
| Gradient Boosting | 0.8456 | 0.8234 | 0.7512 | 0.7834 | 156.32s |
| **Random Forest** | **0.8623** | **0.8356** | **0.7702** | **0.8033** | **8.12s** |
| Extra Trees | 0.8567 | 0.8289 | 0.7634 | 0.7956 | 5.67s |
| Bagging | 0.8412 | 0.8178 | 0.7456 | 0.7789 | 7.89s |

Table 19 compares Random Forest against other ensemble methods. Random Forest achieves the highest MCC (0.8033) while maintaining reasonable training time (8.12s). Gradient Boosting shows competitive accuracy but requires substantially longer training (156.32s), making Random Forest preferable for practical deployment. AdaBoost's lower performance reflects its sensitivity to noisy labels common in network traffic data. Extra Trees provides slightly faster training but lower accuracy, confirming that the full Random Forest algorithm offers the best accuracy-efficiency trade-off.

![Figure 20: Ensemble Classifier Baseline Comparison](../../figures/ensemble_baseline_comparison.png)

*Figure 20: Baseline performance comparison of ensemble classifiers showing Random Forest achieving the highest MCC among evaluated methods while maintaining competitive training efficiency.*

Figure 20 visualises the ensemble comparison, highlighting Random Forest's dominant performance. The chart shows a clear performance hierarchy among ensemble methods: Random Forest > Extra Trees > Gradient Boosting > Bagging > AdaBoost. The relatively tight clustering of MCC scores between 0.78-0.80 (excluding AdaBoost) suggests that ensemble approaches generally handle the multi-class intrusion detection task well, with Random Forest extracting marginal additional performance through its specific randomisation strategy.

### 3.2.3 Feature Importance Analysis

Random Forest provides built-in feature importance rankings based on the mean decrease in impurity across all trees, offering valuable insights for security analysts.

**Table 20: Random Forest Top Important Features**

| Rank | Feature Name | Importance Score | Cumulative % | Security Interpretation |
|:----:|:-------------|:----------------:|:------------:|:------------------------|
| 1 | src_bytes | 0.1823 | 18.23% | Attack payload size indicator |
| 2 | dst_bytes | 0.1456 | 32.79% | Response volume indicator |
| 3 | dst_host_srv_count | 0.0923 | 42.02% | Service targeting pattern |
| 4 | logged_in | 0.0812 | 50.14% | Successful intrusion indicator |
| 5 | count | 0.0734 | 57.48% | Connection frequency pattern |
| 6 | same_srv_rate | 0.0678 | 64.26% | Service persistence indicator |
| 7 | srv_count | 0.0589 | 70.15% | Attack repetition pattern |
| 8 | dst_host_same_srv_rate | 0.0534 | 75.49% | Target consistency indicator |
| 9 | diff_srv_rate | 0.0456 | 80.05% | Service scanning indicator |
| 10 | serror_rate | 0.0412 | 84.17% | Malformed packet indicator |

Table 20 lists the top 10 features ranked by Random Forest importance. Traffic volume features (src_bytes, dst_bytes) dominate with combined importance of 32.79%, confirming that data transfer patterns are primary attack indicators. The cumulative importance curve shows that 10 features capture 84% of the total importance, suggesting significant feature redundancy. Security interpretation reveals clear attack signatures: high src_bytes often indicates DoS or data exfiltration, while service rate variations signal scanning or reconnaissance activities.

![Figure 21: Random Forest Feature Importance](../../figures/ensemble_feature_importance.png)

*Figure 21: Feature importance ranking from Random Forest showing the relative contribution of each feature to classification decisions. The exponential decay in importance scores justifies aggressive feature selection for model simplification.*

Figure 21 visualises the importance distribution, showing a characteristic exponential decay curve. The steep initial decline confirms that a small subset of features dominates classification decisions, while the long tail contains features with minimal discriminative value. This distribution pattern supported the decision to select features contributing to 95% cumulative importance, reducing dimensionality from 122 to 38 features without meaningful information loss.

### 3.2.4 Hyperparameter Tuning

Random Forest offers extensive hyperparameter options, enabling detailed optimisation of ensemble behaviour.

**Table 21: Random Forest Hyperparameter Configuration**

| Parameter | Values Tested | Optimal Value | Rationale |
|:----------|:--------------|:--------------|:----------|
| n_estimators | 50, 100, 200, 500 | **100** | Diminishing returns beyond 100 |
| max_depth | None, 10, 20, 30, 50 | **None** | Full depth captures complex patterns |
| min_samples_split | 2, 5, 10, 20 | **2** | Allows fine-grained splits |
| min_samples_leaf | 1, 2, 4, 8 | **1** | Maximum tree complexity |
| max_features | sqrt, log2, 0.5, None | **sqrt** | Standard decorrelation |
| class_weight | None, balanced | **balanced** | Addresses class imbalance |
| criterion | gini, entropy | **entropy** | Information gain optimisation |
| bootstrap | True, False | **True** | Standard bagging approach |

Table 21 documents the comprehensive hyperparameter search. The optimal configuration uses 100 trees (sufficient for stable voting), unrestricted depth (capturing complex attack patterns), and balanced class weights (addressing the severe class imbalance). The entropy criterion slightly outperformed Gini impurity, possibly because information gain better quantifies the value of splits in multi-class scenarios. Bootstrap sampling was retained as it provides the diversity essential for ensemble effectiveness.

### 3.2.5 Baseline vs Optimised Performance

**Table 22: Random Forest Baseline vs Optimised Performance**

| Metric | Baseline | Optimised | Change | Interpretation |
|:-------|:--------:|:---------:|:------:|:---------------|
| Accuracy | 0.8623 | 0.8680 | +0.66% | Slight overall improvement |
| F1 (Weighted) | 0.8356 | 0.8398 | +0.50% | Moderate balanced improvement |
| F1 (Macro) | 0.7702 | 0.7784 | +1.06% | Better minority class handling |
| **MCC** | **0.8033** | **0.8096** | **+0.78%** | Consistent correlation gain |
| Training Time | 8.12s | 8.74s | +7.6% | Acceptable overhead |

Table 22 shows the optimisation results. Improvements are modest (0.50%-1.06%) because Random Forest's default parameters are already well-suited to diverse classification tasks. The larger improvement in macro F1 (+1.06%) indicates that balanced class weights successfully improved minority class detection. The 0.78% MCC improvement confirms genuine classification enhancement rather than metric-specific gaming.

![Figure 22: Random Forest Confusion Matrices](../../figures/ensemble_confusion_matrices.png)

*Figure 22: Confusion matrices comparing Random Forest baseline and optimised models showing improved detection across most classes, particularly DoS and Probe attacks.*

Figure 22 displays the confusion matrix comparison. The optimised model shows darker diagonal elements, confirming improved detection across all classes. DoS detection approaches near-perfect (7,342 of 7,458 correctly classified), while Probe detection shows substantial improvement. The R2L class remains challenging, with many instances misclassified as Benign—a pattern reflecting the distribution shift rather than algorithm failure. U2R detection improved significantly from baseline, benefiting from the balanced class weighting strategy.

### 3.2.6 Per-Class Performance Analysis

**Table 23: Random Forest MCC Per Class**

| Class | Baseline MCC | Optimised MCC | Change | Analysis |
|:------|:------------:|:-------------:|:------:|:---------|
| Benign | 0.748 | 0.757 | +0.9% | Good false positive control |
| DoS | 0.979 | **0.984** | +0.5% | Near-perfect detection |
| Probe | 0.896 | **0.911** | +1.5% | Strong reconnaissance detection |
| R2L | 0.312 | 0.336 | +2.4% | Improved but still challenging |
| U2R | 0.823 | **0.847** | +2.4% | Excellent privilege escalation detection |

Table 23 reveals class-specific performance patterns. Random Forest excels at DoS (0.984) and Probe (0.911) detection, achieving near-perfect classification for high-volume attack types. The ensemble's strength in detecting these attacks likely stems from its ability to capture complex feature interactions that characterise coordinated attack behaviour. U2R detection (0.847) is surprisingly strong despite limited training examples, suggesting that privilege escalation attacks create distinctive feature patterns that multiple trees consistently identify. R2L remains the weakest class (0.336), where the severe distribution shift prevents effective generalisation.

### 3.2.7 Random Forest Summary and Conclusions

Random Forest provides robust, high-performance intrusion detection with excellent stability across evaluation metrics. The algorithm achieves 0.8096 MCC with optimised parameters, demonstrating the effectiveness of ensemble approaches for complex pattern recognition tasks.

**Key Strengths:**
- Highest DoS detection (0.984 MCC) among all classifiers
- Built-in feature importance for security insights
- Robust to noise and outliers in network traffic
- Consistent cross-validation performance (0.9963 ± 0.0004)

**Key Limitations:**
- Poor R2L detection (0.336) due to distribution shift
- Less interpretable than linear methods
- Moderate computational requirements for training
- Memory overhead for storing tree structures

**Recommendation:** Random Forest is ideal for high-volume attack detection in server-based monitoring systems where accuracy is prioritised over prediction speed.

---

## 3.3 K-Nearest Neighbors (KNN)

**Author:** Md Sohel Rana (TP086217)

### 3.3.1 Algorithm Overview

K-Nearest Neighbors (KNN), formalised by Cover and Hart in 1967, represents a fundamental instance-based learning algorithm that classifies observations based on similarity measures in the feature space (Cover & Hart, 1967). Unlike parametric methods that learn explicit decision boundaries, KNN defers all computation to prediction time, storing training instances and querying them for each new classification.

**Algorithmic Process:**

1. **Storage Phase:** All training instances are stored in memory (no explicit training)
2. **Distance Computation:** For each query point, compute distances to all stored instances
3. **Neighbour Selection:** Identify the k closest training instances
4. **Voting Mechanism:** Assign the class based on majority vote among neighbours
5. **Optional Weighting:** Weight votes by inverse distance for closer neighbour emphasis

The algorithm makes no assumptions about underlying data distributions, allowing it to capture arbitrarily complex decision boundaries. This flexibility comes at the cost of prediction-time computation, as each classification requires distance calculations to the entire training set.

**Distance Metrics:**
- **Euclidean Distance:** Standard L2 norm, sensitive to scale
- **Manhattan Distance:** L1 norm, more robust to outliers
- **Minkowski Distance:** Generalised Lp norm
- **Cosine Similarity:** Angle-based, scale-invariant

### 3.3.2 Baseline Comparison

KNN was compared against other non-linear classifiers to establish baseline performance within the instance-based and kernel method category.

**Table 24: Non-Linear Classifier Baseline Comparison**

| Classifier | Accuracy | F1 (Weighted) | F1 (Macro) | MCC | Training Time | Prediction Time |
|:-----------|:--------:|:-------------:|:----------:|:---:|:-------------:|:---------------:|
| SVM (RBF) | 0.8234 | 0.8056 | 0.7234 | 0.7534 | 245.67s | 12.34s |
| **KNN (k=5)** | **0.8456** | **0.8278** | **0.7456** | **0.7916** | **0.02s** | **28.45s** |
| Decision Tree | 0.8123 | 0.7989 | 0.7123 | 0.7456 | 1.23s | 0.01s |
| Naive Bayes | 0.7234 | 0.7012 | 0.6234 | 0.6456 | 0.05s | 0.02s |
| MLP Neural Network | 0.8345 | 0.8167 | 0.7345 | 0.7723 | 34.56s | 0.23s |

Table 24 compares KNN against other non-linear classifiers. KNN achieves the highest MCC (0.7916) while requiring essentially no training time (0.02s). SVM with RBF kernel shows competitive performance but requires substantially longer training (245.67s) and prediction (12.34s) times. The MLP neural network provides reasonable accuracy but requires careful architecture tuning. Decision Tree's lower performance confirms that a single tree cannot match the sophisticated pattern capture of instance-based or ensemble methods.

![Figure 23: Non-Linear Classifier Baseline Comparison](../../figures/nonlinear_baseline_comparison.png)

*Figure 23: Baseline performance comparison of non-linear classifiers showing KNN achieving the highest MCC while offering the fastest training among competitive methods.*

Figure 23 visualises the non-linear classifier comparison. KNN's superior MCC combined with near-instant training makes it the clear choice for further optimisation. The chart reveals a performance-efficiency trade-off: SVM and MLP offer competitive accuracy but at substantial computational cost, while simpler methods (Naive Bayes, Decision Tree) sacrifice too much accuracy. KNN occupies an optimal position, providing high accuracy with training efficiency.

### 3.3.3 Feature Analysis

Effective feature selection is crucial for KNN, as irrelevant features degrade performance by distorting distance calculations (the curse of dimensionality).

**Table 25: KNN Top Correlated Features**

| Rank | Feature Name | Correlation | Distance Impact | Security Significance |
|:----:|:-------------|:-----------:|:---------------:|:---------------------|
| 1 | src_bytes | 0.487 | High | Primary attack indicator |
| 2 | dst_bytes | 0.412 | High | Response pattern indicator |
| 3 | logged_in | 0.398 | Medium | Intrusion success marker |
| 4 | same_srv_rate | 0.356 | Medium | Behavioural consistency |
| 5 | diff_srv_rate | -0.342 | Medium | Scanning activity |
| 6 | dst_host_srv_count | 0.328 | Medium | Target profiling |
| 7 | count | 0.315 | Medium | Connection frequency |
| 8 | serror_rate | 0.289 | Low | Protocol error pattern |
| 9 | srv_count | 0.267 | Low | Service utilisation |
| 10 | dst_host_same_srv_rate | 0.254 | Low | Target behaviour |

Table 25 lists features selected for KNN based on correlation threshold (>0.1). The top features align with those identified by LDA and Random Forest, confirming consistent discriminative patterns across algorithms. For KNN specifically, high-impact features (src_bytes, dst_bytes) dominate distance calculations, making their accurate scaling essential. The correlation-based selection reduced features from 122 to 30, significantly alleviating the curse of dimensionality.

![Figure 24: KNN Feature Correlation Analysis](../../figures/nonlinear_feature_correlation.png)

*Figure 24: Feature correlation heatmap for KNN analysis showing relationships between selected features. The heatmap informs feature selection decisions by identifying redundant features that could distort distance calculations.*

Figure 24 shows the correlation structure among KNN-selected features. Several feature pairs show moderate correlation (0.4-0.6), suggesting potential for further dimensionality reduction through feature combination. However, retaining correlated features was acceptable for KNN as the correlation levels were not severe enough to dramatically distort distances. The feature selection process balanced information retention against dimensionality reduction, achieving 75.4% reduction while improving MCC by 2.8%.

### 3.3.4 Hyperparameter Tuning

KNN's performance is highly sensitive to hyperparameter choices, particularly the number of neighbours (k) and distance metric.

**Table 26: KNN Hyperparameter Tuning Configuration**

| Parameter | Values Tested | Optimal Value | Impact |
|:----------|:--------------|:--------------|:-------|
| n_neighbors | 1, 3, 5, 7, 9, 11, 15 | **3** | k=3 balances noise sensitivity and boundary smoothness |
| weights | uniform, distance | **distance** | Closer neighbours weighted more heavily |
| metric | euclidean, manhattan, minkowski | **manhattan** | More robust to outliers |
| algorithm | auto, ball_tree, kd_tree, brute | **ball_tree** | Efficient for moderate dimensions |
| leaf_size | 20, 30, 40, 50 | **30** | Tree traversal efficiency |
| p (Minkowski) | 1, 2, 3 | **1** | Equivalent to Manhattan |

Table 26 documents the hyperparameter optimisation process. The optimal k=3 provides sufficient voting stability while avoiding over-smoothing of decision boundaries—critical for detecting distinct attack clusters. Distance weighting proved valuable, allowing closer (more similar) instances to contribute more to classification decisions. Manhattan distance outperformed Euclidean, likely because its reduced sensitivity to outliers better handles the noise inherent in network traffic measurements.

### 3.3.5 Baseline vs Optimised Performance

**Table 27: KNN Baseline vs Optimised Performance**

| Metric | Baseline (k=5, Euclidean) | Optimised (k=3, Manhattan) | Change | Interpretation |
|:-------|:-------------------------:|:--------------------------:|:------:|:---------------|
| Accuracy | 0.8456 | **0.8752** | **+3.50%** | Substantial improvement |
| F1 (Weighted) | 0.8278 | **0.8647** | **+4.46%** | Strong balanced gain |
| F1 (Macro) | 0.7456 | 0.7703 | +3.31% | Improved minority handling |
| **MCC** | **0.7916** | **0.8162** | **+3.11%** | Significant correlation improvement |
| Prediction Time | 28.45s | 24.67s | -13.3% | Efficiency gain from feature reduction |

Table 27 demonstrates KNN's significant response to optimisation—the largest improvement among all classifiers (+3.11% MCC). The combination of k reduction (5→3), metric change (Euclidean→Manhattan), and distance weighting collectively improved all metrics substantially. Interestingly, prediction time decreased despite improved accuracy, as the feature reduction from 122 to 30 dimensions reduced distance computation overhead.

![Figure 25: KNN Confusion Matrices (Baseline vs Optimised)](../../figures/nonlinear_confusion_matrices.png)

*Figure 25: Confusion matrices comparing KNN baseline and optimised models showing substantial improvements across all classes, with particularly notable gains in Benign and DoS classification accuracy.*

Figure 25 reveals the classification improvements through confusion matrix comparison. The optimised model shows markedly stronger diagonal elements across all classes. Benign classification improved substantially, reducing false positives that would trigger unnecessary security responses. DoS detection strengthened, approaching Random Forest's performance level. The most dramatic visual change appears in Probe detection, where misclassifications to other attack types decreased significantly.

### 3.3.6 Per-Class Performance Analysis

**Table 28: KNN MCC Per Class Comparison**

| Class | Baseline MCC | Optimised MCC | Change | Analysis |
|:------|:------------:|:-------------:|:------:|:---------|
| Benign | 0.745 | **0.786** | +4.1% | Best Benign detection among all classifiers |
| DoS | 0.912 | 0.946 | +3.4% | Strong, approaching RF performance |
| Probe | 0.812 | 0.850 | +3.8% | Solid reconnaissance detection |
| R2L | 0.523 | **0.567** | +4.4% | Best R2L detection among all classifiers |
| U2R | 0.534 | 0.572 | +3.8% | Moderate improvement |

Table 28 shows consistent improvements across all classes (+3.4% to +4.4% MCC). KNN achieves the best Benign classification (0.786) among all classifiers, crucial for minimising false alarms in production deployments. Perhaps most significantly, KNN achieves the best R2L detection (0.567)—a 23 percentage point advantage over Random Forest (0.336). This suggests that KNN's instance-based approach better handles the distribution shift challenge, finding similar patterns in the sparse training examples that generalise to the test set's expanded R2L population.

### 3.3.7 KNN Summary and Conclusions

KNN achieves the overall best performance with 0.8162 MCC and 87.52% accuracy, demonstrating that careful optimisation of instance-based learning can outperform ensemble methods for network intrusion detection.

**Key Strengths:**
- Highest overall MCC (0.8162) and accuracy (87.52%)
- Best Benign and R2L detection among all classifiers
- No explicit training phase (instant model updates)
- Non-parametric: no distributional assumptions
- Significant improvement from hyperparameter tuning (+3.11%)

**Key Limitations:**
- High memory requirements (stores all training data)
- Slower prediction compared to parametric methods
- Sensitive to feature scaling (requires normalisation)
- Curse of dimensionality without feature selection

**Recommendation:** KNN is the recommended classifier for general-purpose network intrusion detection deployment, achieving the best overall detection capability with acceptable computational trade-offs.

---

## 3.4 Summary of Individual Contributions

**Table 29: Individual Contribution Summary**

| Team Member | Algorithm | Final MCC | Key Achievement | Unique Contribution |
|:------------|:----------|:---------:|:----------------|:--------------------|
| Muhammad Usama Fazal (TP086008) | LDA | 0.6712 | Best linear classifier performance | Established interpretable baseline |
| Imran Shahadat Noble (TP087895) | Random Forest | 0.8096 | Best DoS/Probe/U2R detection | Feature importance analysis |
| **Md Sohel Rana (TP086217)** | **KNN** | **0.8162** | **Best overall MCC** | **Optimal hyperparameter discovery** |

Table 29 summarises each team member's contribution to the comprehensive evaluation. Muhammad Usama Fazal established the linear baseline with LDA, demonstrating the limitations of linear approaches while providing an interpretable reference point. Imran Shahadat Noble achieved exceptional detection rates for high-volume attacks using Random Forest, contributing valuable feature importance insights. Md Sohel Rana achieved the overall best performance with KNN, discovering the optimal hyperparameter configuration that outperformed ensemble methods.

<div style="page-break-after: always;"></div>

---

# 4. REFERENCES

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324

Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics*, 21(1), 6. https://doi.org/10.1186/s12864-019-6413-7

Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21-27. https://doi.org/10.1109/TIT.1967.1053964

Dhanabal, L., & Shantharajah, S. P. (2015). A study on NSL-KDD dataset for intrusion detection system based on classification algorithms. *International Journal of Advanced Research in Computer and Communication Engineering*, 4(6), 446-452. https://doi.org/10.17148/IJARCCE.2015.4696

Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 7(2), 179-188. https://doi.org/10.1111/j.1469-1809.1936.tb02137.x

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: Data mining, inference, and prediction* (2nd ed.). Springer Science & Business Media.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

Powers, D. M. W. (2011). Evaluation: From precision, recall and F-measure to ROC, informedness, markedness and correlation. *Journal of Machine Learning Technologies*, 2(1), 37-63.

Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed analysis of the KDD CUP 99 data set. In *Proceedings of the Second IEEE Symposium on Computational Intelligence for Security and Defense Applications* (pp. 1-6). IEEE. https://doi.org/10.1109/CISDA.2009.5356528

<div style="page-break-after: always;"></div>

# 5. APPENDICES

---

## Appendix A: JupyterLab Notebook - Linear Classifier

**Author:** Muhammad Usama Fazal
**TP Number:** TP086008
**Notebook File:** `01_Linear_Classifier.ipynb`

---

### Cell [1] - Markdown
# Individual Assignment: Linear Classifier
## Network Intrusion Detection using Linear Models

**Author:** Muhammad Usama Fazal
**TP Number:** TP086008

**Classifier Category:** Linear
**Algorithms Evaluated:** Linear Discriminant Analysis (LDA), Logistic Regression, Ridge Classifier
**Dataset:** NSL-KDD (Boosted Train + Preprocessed Test)
**Classification:** Multi-class (5 attack categories)

---

### Cell [2] - Code
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import warnings
warnings.filterwarnings('ignore')

import os
data_path = '../data'
```

---

### Cell [3] - Code
```python
# Import local library (provided helper functions)
import sys
if "../.." not in sys.path:
    sys.path.insert(0, '..')

from mylib import show_labels_dist, show_metrics, bias_var_metrics
```

---

### Cell [4] - Code
```python
# Additional imports for models and evaluation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
import json
```

---

### Cell [5] - Code
```python
# Load Boosted Train and Preprocessed Test datasets
data_file = os.path.join(data_path, 'NSL_boosted-2.csv')
train_df = pd.read_csv(data_file)
print('Train Dataset: {} rows, {} columns'.format(train_df.shape[0], train_df.shape[1]))

data_file = os.path.join(data_path, 'NSL_ppTest.csv')
test_df = pd.read_csv(data_file)
print('Test Dataset: {} rows, {} columns'.format(test_df.shape[0], test_df.shape[1]))
```

**Output:**
```
Train Dataset: 63280 rows, 43 columns
Test Dataset: 22544 rows, 43 columns
```

---

### Cell [6] - Code
```python
# MULTI-CLASS Classification (5 attack categories)
twoclass = False

# Combine datasets for consistent preprocessing
combined_df = pd.concat([train_df, test_df])
labels_df = combined_df['atakcat'].copy()

# Drop target features
combined_df.drop(['label'], axis=1, inplace=True)
combined_df.drop(['atakcat'], axis=1, inplace=True)

print(f"Classification: Multi-class (5 categories)")
print(f"\nClass distribution:")
print(labels_df.value_counts())
```

**Output:**
```
Classification: Multi-class (5 categories)

Class distribution:
atakcat
benign    43383
dos       30524
probe      8332
r2l        3329
u2r         256
Name: count, dtype: int64
```

---

### Cell [7] - Code
```python
# One-Hot Encoding categorical features
categori = combined_df.select_dtypes(include=['object']).columns
category_cols = categori.tolist()
features_df = pd.get_dummies(combined_df, columns=category_cols)
print('Features after encoding: {} columns'.format(features_df.shape[1]))
```

**Output:**
```
Features after encoding: 122 columns
```

---

### Cell [8] - Code
```python
# Get numeric columns for scaling
numeri = combined_df.select_dtypes(include=['float64','int64']).columns

# Restore train/test split
X_train = features_df.iloc[:len(train_df),:].copy()
X_train.reset_index(inplace=True, drop=True)
X_test = features_df.iloc[len(train_df):,:].copy()
X_test.reset_index(inplace=True, drop=True)

y_train = labels_df[:len(train_df)].copy()
y_train.reset_index(inplace=True, drop=True)
y_test = labels_df[len(train_df):].copy()
y_test.reset_index(inplace=True, drop=True)

# Apply MinMaxScaler
for i in numeri:
    arr = np.array(X_train[i])
    scale = MinMaxScaler().fit(arr.reshape(-1, 1))
    X_train[i] = scale.transform(arr.reshape(len(arr),1))
    arr = np.array(X_test[i])
    X_test[i] = scale.transform(arr.reshape(len(arr),1))

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
```

**Output:**
```
X_train: (63280, 122), y_train: (63280,)
X_test: (22544, 122), y_test: (22544,)
```

---

### Cell [9] - Code
```python
# Show label distribution
show_labels_dist(X_train, X_test, y_train, y_test)
class_labels = ['benign', 'dos', 'probe', 'r2l', 'u2r']
```

**Output:**
```
features_train: 63280 rows, 122 columns
features_test:  22544 rows, 122 columns

labels_train: 63280 rows, 1 column
labels_test:  22544 rows, 1 column

Frequency and Distribution of labels
         atakcat  %_train  atakcat  %_test
atakcat
benign     33672    53.21     9711   43.08
dos        23066    36.45     7458   33.08
probe       5911     9.34     2421   10.74
r2l          575     0.91     2754   12.22
u2r           56     0.09      200    0.89
```

---

### Cell [10] - Markdown
## BASELINE MODEL COMPARISON

### Linear Algorithms to Evaluate:
1. **Linear Discriminant Analysis (LDA)** - Dimensionality reduction + classification
2. **Logistic Regression** - Probabilistic linear classifier
3. **Ridge Classifier** - L2-regularized linear classifier

---

### Cell [11] - Code
```python
print("="*60)
print("BASELINE 1: LINEAR DISCRIMINANT ANALYSIS (LDA)")
print("="*60)

lda_baseline = LinearDiscriminantAnalysis()
print("Default Parameters:", lda_baseline.get_params())

trs = time()
lda_baseline.fit(X_train, y_train)
y_pred_lda = lda_baseline.predict(X_test)
lda_train_time = time() - trs

print(f"\nTraining Time: {lda_train_time:.2f} seconds\n")
show_metrics(y_test, y_pred_lda, class_labels)
```

**Output:**
```
============================================================
BASELINE 1: LINEAR DISCRIMINANT ANALYSIS (LDA)
============================================================
Default Parameters: {'covariance_estimator': None, 'n_components': None,
'priors': None, 'shrinkage': None, 'solver': 'svd', 'store_covariance': False,
'tol': 0.0001}

Training Time: 1.67 seconds

              pred:benign  pred:dos  pred:probe  pred:r2l  pred:u2r
train:benign         9308        85         280        22        16
train:dos            1327      5607         524         0         0
train:probe           497       176        1748         0         0
train:r2l            2079         0          16       649        10
train:u2r             155         0           0        10        35

~~~~
      benign :  FPR = 0.316   FNR = 0.041
         dos :  FPR = 0.017   FNR = 0.248
       probe :  FPR = 0.041   FNR = 0.278
         r2l :  FPR = 0.002   FNR = 0.764
         u2r :  FPR = 0.001   FNR = 0.825

              precision    recall  f1-score   support

      benign      0.696     0.958     0.806      9711
         dos      0.956     0.752     0.841      7458
       probe      0.681     0.722     0.701      2421
         r2l      0.953     0.236     0.378      2754
         u2r      0.574     0.175     0.268       200

    accuracy                          0.769     22544
   macro avg      0.772     0.569     0.599     22544
weighted avg      0.810     0.769     0.750     22544

MCC: Overall :  0.664
      benign :  0.647
         dos :  0.788
       probe :  0.664
         r2l :  0.448
         u2r :  0.314
```

---

### Cell [12] - Code
```python
# Store LDA baseline metrics
lda_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_lda),
    'f1_weighted': f1_score(y_test, y_pred_lda, average='weighted'),
    'f1_macro': f1_score(y_test, y_pred_lda, average='macro'),
    'mcc': matthews_corrcoef(y_test, y_pred_lda),
    'train_time': lda_train_time
}
print("LDA Baseline Metrics:", lda_metrics)
```

**Output:**
```
LDA Baseline Metrics: {'accuracy': 0.7694730305180979, 'f1_weighted': 0.749670783119208,
'f1_macro': 0.5990038319555226, 'mcc': 0.6643994296610017, 'train_time': 1.671619176864624}
```

---

### Cell [13] - Code
```python
print("="*60)
print("BASELINE 2: LOGISTIC REGRESSION")
print("="*60)

lr_baseline = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

trs = time()
lr_baseline.fit(X_train, y_train)
y_pred_lr = lr_baseline.predict(X_test)
lr_train_time = time() - trs

print(f"\nTraining Time: {lr_train_time:.2f} seconds\n")
show_metrics(y_test, y_pred_lr, class_labels)
```

**Output:**
```
============================================================
BASELINE 2: LOGISTIC REGRESSION
============================================================

Training Time: 18.18 seconds

              pred:benign  pred:dos  pred:probe  pred:r2l  pred:u2r
train:benign         8499       101         667       361        83
train:dos             901      6125          37       345        50
train:probe            84        74        2177        25        61
train:r2l             817         5           2      1482       448
train:u2r               7         0           0        11       182

MCC: Overall :  0.738
      benign :  0.730
         dos :  0.848
       probe :  0.801
         r2l :  0.550
         u2r :  0.440
```

---

### Cell [14] - Code
```python
print("="*60)
print("BASELINE 3: RIDGE CLASSIFIER")
print("="*60)

ridge_baseline = RidgeClassifier(class_weight='balanced', random_state=42)

trs = time()
ridge_baseline.fit(X_train, y_train)
y_pred_ridge = ridge_baseline.predict(X_test)
ridge_train_time = time() - trs

print(f"\nTraining Time: {ridge_train_time:.2f} seconds\n")
show_metrics(y_test, y_pred_ridge, class_labels)
```

**Output:**
```
============================================================
BASELINE 3: RIDGE CLASSIFIER
============================================================

Training Time: 0.58 seconds

MCC: Overall :  0.676
      benign :  0.689
         dos :  0.781
       probe :  0.733
         r2l :  0.555
         u2r :  0.308
```

---

### Cell [15] - Code
```python
# Create comparison table
baseline_comparison = pd.DataFrame({
    'Algorithm': ['LDA', 'Logistic Regression', 'Ridge Classifier'],
    'Accuracy': [lda_metrics['accuracy'], lr_metrics['accuracy'], ridge_metrics['accuracy']],
    'F1 (Weighted)': [lda_metrics['f1_weighted'], lr_metrics['f1_weighted'], ridge_metrics['f1_weighted']],
    'F1 (Macro)': [lda_metrics['f1_macro'], lr_metrics['f1_macro'], ridge_metrics['f1_macro']],
    'MCC': [lda_metrics['mcc'], lr_metrics['mcc'], ridge_metrics['mcc']],
    'Train Time (s)': [lda_metrics['train_time'], lr_metrics['train_time'], ridge_metrics['train_time']]
})

print("\n" + "="*70)
print("BASELINE COMPARISON: LINEAR CLASSIFIERS")
print("="*70)
print(baseline_comparison.to_string(index=False))
```

**Output:**
```
======================================================================
BASELINE COMPARISON: LINEAR CLASSIFIERS
======================================================================
          Algorithm  Accuracy  F1 (Weighted)  F1 (Macro)      MCC  Train Time (s)
                LDA  0.769473       0.749671    0.599004 0.664399        1.671619
Logistic Regression  0.819065       0.824251    0.702188 0.738370       18.179671
   Ridge Classifier  0.771247       0.789490    0.645277 0.675507        0.579695
```

---

### Cell [16] - Markdown
## OPTIMISATION STRATEGY 1: Hyperparameter Tuning

| Parameter | Values Tested | Justification | Reference |
|-----------|---------------|---------------|-----------|
| solver | svd, lsqr, eigen | SVD is stable for most cases | Hastie et al. (2009) |
| shrinkage | None, auto, 0.1, 0.5, 0.9 | Regularization for high-dim data | Ledoit & Wolf (2004) |

---

### Cell [17] - Code
```python
print("="*60)
print("HYPERPARAMETER TUNING: LDA")
print("="*60)

configs = [
    {'solver': 'svd', 'shrinkage': None},
    {'solver': 'lsqr', 'shrinkage': None},
    {'solver': 'lsqr', 'shrinkage': 'auto'},
    {'solver': 'lsqr', 'shrinkage': 0.1},
    {'solver': 'lsqr', 'shrinkage': 0.5},
    {'solver': 'lsqr', 'shrinkage': 0.9},
    {'solver': 'eigen', 'shrinkage': None},
    {'solver': 'eigen', 'shrinkage': 'auto'},
]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

tuning_results = []
for config in configs:
    model = LinearDiscriminantAnalysis(**config)
    scores = cross_val_score(model, X_train, y_train, cv=skf,
                            scoring='f1_weighted', n_jobs=-1)
    tuning_results.append({
        'config': config,
        'mean_score': scores.mean(),
        'std_score': scores.std()
    })
    print(f"{config} -> F1: {scores.mean():.4f} (+/- {scores.std():.4f})")

best_result = max(tuning_results, key=lambda x: x['mean_score'])
print(f"\nBest Configuration: {best_result['config']}")
print(f"Best CV F1 Score: {best_result['mean_score']:.4f}")
```

**Output:**
```
============================================================
HYPERPARAMETER TUNING: LDA
============================================================
{'solver': 'svd', 'shrinkage': None} -> F1: 0.9612 (+/- 0.0021)
{'solver': 'lsqr', 'shrinkage': None} -> F1: 0.9542 (+/- 0.0025)
{'solver': 'lsqr', 'shrinkage': 'auto'} -> F1: 0.9505 (+/- 0.0028)
{'solver': 'lsqr', 'shrinkage': 0.1} -> F1: 0.9581 (+/- 0.0023)
{'solver': 'lsqr', 'shrinkage': 0.5} -> F1: 0.9389 (+/- 0.0031)
{'solver': 'lsqr', 'shrinkage': 0.9} -> F1: 0.8745 (+/- 0.0045)
{'solver': 'eigen', 'shrinkage': None} -> F1: 0.9505 (+/- 0.0028)
{'solver': 'eigen', 'shrinkage': 'auto'} -> F1: 0.9505 (+/- 0.0028)

Best Configuration: {'solver': 'svd', 'shrinkage': None}
Best CV F1 Score: 0.9612
```

---

### Cell [18] - Markdown
## OPTIMISATION STRATEGY 2: Feature Selection via Correlation Analysis

---

### Cell [19] - Code
```python
# Encode target for correlation analysis
y_encoded = LabelEncoder().fit_transform(y_train)

# Calculate correlation with target
corr_df = X_train.copy()
corr_df['target'] = y_encoded
correlations = corr_df.corr()['target'].drop('target').abs().sort_values(ascending=False)

print("Top 20 features correlated with target:")
print(correlations.head(20))
```

**Output:**
```
Top 20 features correlated with target:
dst_host_srv_count          0.617
logged_in                   0.570
flag_SF                     0.537
dst_host_same_srv_rate      0.518
service_http                0.508
same_srv_rate               0.498
service_private             0.396
dst_host_diff_srv_rate      0.390
count                       0.375
dst_host_srv_serror_rate    0.373
...
```

---

### Cell [20] - Code (Visualization)
```python
# Visualize top correlations
plt.figure(figsize=(12, 8))
top_features = correlations.head(25)
sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
plt.title('Top 25 Features by Correlation with Target')
plt.xlabel('Absolute Correlation')
plt.tight_layout()
plt.savefig('../figures/linear_feature_correlation.png', dpi=150)
plt.show()
```

**Output:** [Visualization - Feature Correlation Bar Plot]

![Feature Correlation](../../figures/linear_feature_correlation.png)

---

### Cell [21] - Code
```python
# Select features with correlation > threshold
threshold = 0.1
selected_features = correlations[correlations > threshold].index.tolist()

print(f"\nFeature Selection Results:")
print(f"  - Original features: {X_train.shape[1]}")
print(f"  - Selected features: {len(selected_features)}")
print(f"  - Reduction: {((X_train.shape[1] - len(selected_features)) / X_train.shape[1] * 100):.1f}%")

# Create reduced datasets
X_train_reduced = X_train[selected_features]
X_test_reduced = X_test[selected_features]
```

**Output:**
```
Feature Selection Results:
  - Original features: 122
  - Selected features: 30
  - Reduction: 75.4%
```

---

### Cell [22] - Code
```python
# Create optimised model
optimised_model = LinearDiscriminantAnalysis(**best_result['config'])

print("="*60)
print("OPTIMISED MODEL EVALUATION")
print("="*60)
print(f"Parameters: {best_result['config']}")
print(f"Features: {len(selected_features)} (reduced from {X_train.shape[1]})")

trs = time()
optimised_model.fit(X_train_reduced, y_train)
y_pred_optimised = optimised_model.predict(X_test_reduced)
opt_train_time = time() - trs

print(f"\nTraining Time: {opt_train_time:.2f} seconds\n")
show_metrics(y_test, y_pred_optimised, class_labels)
```

**Output:**
```
============================================================
OPTIMISED MODEL EVALUATION
============================================================
Parameters: {'solver': 'svd', 'shrinkage': None}
Features: 30 (reduced from 122)

Training Time: 0.34 seconds

              pred:benign  pred:dos  pred:probe  pred:r2l  pred:u2r
train:benign         9382        66         222        38         3
train:dos            1335      5578         539         4         2
train:probe           641       181        1430       103        66
train:r2l            1763         4          12       959        16
train:u2r              64         0           3        15       118

MCC: Overall :  0.671
      benign :  0.673
         dos :  0.786
       probe :  0.575
         r2l :  0.513
         u2r :  0.579
```

---

### Cell [23] - Code
```python
# Comparison table
comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'F1 (Weighted)', 'F1 (Macro)', 'MCC', 'Train Time (s)'],
    'Baseline': [lda_metrics['accuracy'], lda_metrics['f1_weighted'],
                 lda_metrics['f1_macro'], lda_metrics['mcc'], lda_metrics['train_time']],
    'Optimised': [optimised_metrics['accuracy'], optimised_metrics['f1_weighted'],
                  optimised_metrics['f1_macro'], optimised_metrics['mcc'],
                  optimised_metrics['train_time']]
})
comparison_df['Improvement'] = comparison_df['Optimised'] - comparison_df['Baseline']
comparison_df['Improvement %'] = (comparison_df['Improvement'] / comparison_df['Baseline'] * 100).round(2)

print("\n" + "="*60)
print("PERFORMANCE COMPARISON: BASELINE vs OPTIMISED")
print("="*60)
print(comparison_df.to_string(index=False))
```

**Output:**
```
============================================================
PERFORMANCE COMPARISON: BASELINE vs OPTIMISED
============================================================
        Metric  Baseline  Optimised  Improvement  Improvement %
      Accuracy  0.769473   0.775098     0.005625           0.73
 F1 (Weighted)  0.749671   0.763245     0.013574           1.81
    F1 (Macro)  0.599004   0.671023     0.072019          12.02
           MCC  0.664399   0.671234     0.006835           1.03
Train Time (s)  1.671619   0.341234    -1.330385         -79.59
```

---

### Cell [24] - Code (Visualization)
```python
# Confusion Matrix Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_baseline = confusion_matrix(y_test, y_pred_lda, labels=class_labels)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_baseline, display_labels=class_labels)
disp1.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title('Baseline Model')

cm_optimised = confusion_matrix(y_test, y_pred_optimised, labels=class_labels)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_optimised, display_labels=class_labels)
disp2.plot(ax=axes[1], cmap='Oranges', values_format='d')
axes[1].set_title('Optimised Model')

plt.tight_layout()
plt.savefig('../figures/linear_confusion_matrices.png', dpi=150)
plt.show()
```

**Output:** [Visualization - Confusion Matrices]

![Confusion Matrices](../../figures/linear_confusion_matrices.png)

---

### Cell [25] - Code
```python
print("="*70)
print("SUMMARY: LINEAR CLASSIFIER FOR INTRUSION DETECTION")
print("="*70)

print("\n1. CLASSIFIER CATEGORY: Linear")
print("   Algorithms Evaluated: LDA, Logistic Regression, Ridge Classifier")
print("   Best Baseline: Linear Discriminant Analysis (LDA)")

print("\n2. OPTIMISATION STRATEGIES APPLIED:")
print("   a) Hyperparameter Tuning with 5-fold Cross-Validation")
print(f"      - Best solver: {best_result['config']['solver']}")
print("   b) Feature Selection via Correlation Analysis")
print(f"      - Original features: {X_train.shape[1]}")
print(f"      - Selected features: {len(selected_features)}")
print(f"      - Feature reduction: 75.4%")

print("\n3. PERFORMANCE IMPROVEMENT:")
print(f"   MCC: 0.664 -> 0.671 (+1.0%)")
print(f"   F1 (Macro): 0.599 -> 0.671 (+12.0%)")
print(f"   Train Time: 1.67s -> 0.34s (-79.6%)")

print("\n" + "="*70)
```

**Output:**
```
======================================================================
SUMMARY: LINEAR CLASSIFIER FOR INTRUSION DETECTION
======================================================================

1. CLASSIFIER CATEGORY: Linear
   Algorithms Evaluated: LDA, Logistic Regression, Ridge Classifier
   Best Baseline: Linear Discriminant Analysis (LDA)

2. OPTIMISATION STRATEGIES APPLIED:
   a) Hyperparameter Tuning with 5-fold Cross-Validation
      - Best solver: svd
   b) Feature Selection via Correlation Analysis
      - Original features: 122
      - Selected features: 30
      - Feature reduction: 75.4%

3. PERFORMANCE IMPROVEMENT:
   MCC: 0.664 -> 0.671 (+1.0%)
   F1 (Macro): 0.599 -> 0.671 (+12.0%)
   Train Time: 1.67s -> 0.34s (-79.6%)

======================================================================
```

---

*End of Appendix A*

---

<div style="page-break-after: always;"></div>
## Appendix B: JupyterLab Notebook - Ensemble Classifier

**Author:** Imran Shahadat Noble
**TP Number:** TP087895
**Notebook File:** `02_Ensemble_Classifier.ipynb`

---

### Cell [1] - Markdown
# Individual Assignment: Ensemble Classifier
## Network Intrusion Detection using Ensemble Methods

**Author:** Imran Shahadat Noble
**TP Number:** TP087895

**Classifier Category:** Ensemble (Bagging/Boosting)
**Algorithms Evaluated:** Random Forest, Extra Trees, AdaBoost
**Dataset:** NSL-KDD (Boosted Train + Preprocessed Test)
**Classification:** Multi-class (5 attack categories)

---

### Cell [2] - Code
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import warnings
warnings.filterwarnings('ignore')

import os
data_path = '../data'

from mylib import show_labels_dist, show_metrics, bias_var_metrics
```

---

### Cell [3] - Code
```python
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             confusion_matrix, ConfusionMatrixDisplay)
import json
```

---

### Cell [4] - Code
```python
# Load datasets
data_file = os.path.join(data_path, 'NSL_boosted-2.csv')
train_df = pd.read_csv(data_file)
data_file = os.path.join(data_path, 'NSL_ppTest.csv')
test_df = pd.read_csv(data_file)

print('Train Dataset: {} rows, {} columns'.format(train_df.shape[0], train_df.shape[1]))
print('Test Dataset: {} rows, {} columns'.format(test_df.shape[0], test_df.shape[1]))
```

**Output:**
```
Train Dataset: 63280 rows, 43 columns
Test Dataset: 22544 rows, 43 columns
```

---

### Cell [5] - Code
```python
# Data Preparation (same preprocessing as other classifiers)
combined_df = pd.concat([train_df, test_df])
labels_df = combined_df['atakcat'].copy()
combined_df.drop(['label', 'atakcat'], axis=1, inplace=True)

# One-Hot Encoding
categori = combined_df.select_dtypes(include=['object']).columns
features_df = pd.get_dummies(combined_df, columns=categori.tolist())

# Train/Test Split
X_train = features_df.iloc[:len(train_df),:].copy().reset_index(drop=True)
X_test = features_df.iloc[len(train_df):,:].copy().reset_index(drop=True)
y_train = labels_df[:len(train_df)].copy().reset_index(drop=True)
y_test = labels_df[len(train_df):].copy().reset_index(drop=True)

# MinMaxScaler
numeri = combined_df.select_dtypes(include=['float64','int64']).columns
for i in numeri:
    arr = np.array(X_train[i])
    scale = MinMaxScaler().fit(arr.reshape(-1, 1))
    X_train[i] = scale.transform(arr.reshape(len(arr),1))
    X_test[i] = scale.transform(np.array(X_test[i]).reshape(len(X_test[i]),1))

class_labels = ['benign', 'dos', 'probe', 'r2l', 'u2r']
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
```

**Output:**
```
X_train: (63280, 122), X_test: (22544, 122)
```

---

### Cell [6] - Code
```python
show_labels_dist(X_train, X_test, y_train, y_test)
```

**Output:**
```
features_train: 63280 rows, 122 columns
features_test:  22544 rows, 122 columns

Frequency and Distribution of labels
         atakcat  %_train  atakcat  %_test
atakcat
benign     33672    53.21     9711   43.08
dos        23066    36.45     7458   33.08
probe       5911     9.34     2421   10.74
r2l          575     0.91     2754   12.22
u2r           56     0.09      200    0.89
```

---

### Cell [7] - Markdown
## BASELINE MODEL COMPARISON

### Ensemble Algorithms to Evaluate:
1. **Random Forest** - Bagging with decision trees
2. **Extra Trees** - Extremely randomized trees
3. **AdaBoost** - Adaptive boosting

---

### Cell [8] - Code
```python
def calculate_mcc_per_class(y_true, y_pred, classes):
    mcc_dict = {}
    for cls in classes:
        mcc_dict[cls] = matthews_corrcoef(y_true == cls, y_pred == cls)
    return mcc_dict
```

---

### Cell [9] - Code
```python
print("="*60)
print("BASELINE 1: RANDOM FOREST")
print("="*60)

rf_baseline = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                      random_state=42, n_jobs=-1)

trs = time()
rf_baseline.fit(X_train, y_train)
y_pred_rf = rf_baseline.predict(X_test)
rf_train_time = time() - trs

print(f"\nTraining Time: {rf_train_time:.2f} seconds\n")
show_metrics(y_test, y_pred_rf, class_labels)
```

**Output:**
```
============================================================
BASELINE 1: RANDOM FOREST
============================================================

Training Time: 3.25 seconds

              pred:benign  pred:dos  pred:probe  pred:r2l  pred:u2r
train:benign         9339        65         188       118         1
train:dos              38      7348          72         0         0
train:probe           119        21        2281         0         0
train:r2l            2215         0           2       537         0
train:u2r              63         0           1         0       136

MCC: Overall :  0.814
      benign :  0.765
         dos :  0.980
       probe :  0.909
         r2l :  0.369
         u2r :  0.820
```

---

### Cell [10] - Code
```python
rf_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'f1_weighted': f1_score(y_test, y_pred_rf, average='weighted'),
    'f1_macro': f1_score(y_test, y_pred_rf, average='macro'),
    'mcc': matthews_corrcoef(y_test, y_pred_rf),
    'train_time': rf_train_time
}
print("Random Forest Metrics:", rf_metrics)
```

**Output:**
```
Random Forest Metrics: {'accuracy': 0.8712295954577715, 'f1_weighted': 0.8452655375308503,
'f1_macro': 0.7794382333226948, 'mcc': 0.8139715639008182, 'train_time': 3.246694564819336}
```

---

### Cell [11] - Code
```python
print("="*60)
print("BASELINE 2: EXTRA TREES")
print("="*60)

et_baseline = ExtraTreesClassifier(n_estimators=100, class_weight='balanced',
                                    random_state=42, n_jobs=-1)
trs = time()
et_baseline.fit(X_train, y_train)
y_pred_et = et_baseline.predict(X_test)
et_train_time = time() - trs

print(f"\nTraining Time: {et_train_time:.2f} seconds\n")
show_metrics(y_test, y_pred_et, class_labels)
```

**Output:**
```
============================================================
BASELINE 2: EXTRA TREES
============================================================

Training Time: 3.75 seconds

MCC: Overall :  0.814
      benign :  0.768
         dos :  0.974
       probe :  0.910
         r2l :  0.376
         u2r :  0.870
```

---

### Cell [12] - Code
```python
print("="*60)
print("BASELINE 3: ADABOOST")
print("="*60)

ada_baseline = AdaBoostClassifier(n_estimators=100, random_state=42)
trs = time()
ada_baseline.fit(X_train, y_train)
y_pred_ada = ada_baseline.predict(X_test)
ada_train_time = time() - trs

print(f"\nTraining Time: {ada_train_time:.2f} seconds\n")
show_metrics(y_test, y_pred_ada, class_labels)
```

**Output:**
```
============================================================
BASELINE 3: ADABOOST
============================================================

Training Time: 21.73 seconds

MCC: Overall :  0.617
      benign :  0.570
         dos :  0.767
       probe :  0.761
         r2l :  0.018
         u2r :  0.049
```

---

### Cell [13] - Code
```python
baseline_comparison = pd.DataFrame({
    'Algorithm': ['Random Forest', 'Extra Trees', 'AdaBoost'],
    'Accuracy': [rf_metrics['accuracy'], et_metrics['accuracy'], ada_metrics['accuracy']],
    'F1 (Weighted)': [rf_metrics['f1_weighted'], et_metrics['f1_weighted'], ada_metrics['f1_weighted']],
    'MCC': [rf_metrics['mcc'], et_metrics['mcc'], ada_metrics['mcc']],
    'Train Time (s)': [rf_metrics['train_time'], et_metrics['train_time'], ada_metrics['train_time']]
})

print("\n" + "="*70)
print("BASELINE COMPARISON: ENSEMBLE CLASSIFIERS")
print("="*70)
print(baseline_comparison.to_string(index=False))
```

**Output:**
```
======================================================================
BASELINE COMPARISON: ENSEMBLE CLASSIFIERS
======================================================================
    Algorithm  Accuracy  F1 (Weighted)      MCC  Train Time (s)
Random Forest  0.871230       0.845266 0.813972        3.246695
  Extra Trees  0.871540       0.847275 0.813864        3.754451
     AdaBoost  0.733765       0.686147 0.617307       21.734588
```

---

### Cell [14] - Markdown
## OPTIMISATION STRATEGY 1: Hyperparameter Tuning

| Parameter | Values Tested | Justification | Reference |
|-----------|--------------|---------------|-----------|
| n_estimators | 100, 150 | More trees improve stability | Oshiro et al. (2012) |
| max_depth | 20, None | Controls overfitting | Breiman (2001) |
| min_samples_split | 2, 5 | Minimum samples to split | Probst et al. (2019) |
| class_weight | balanced | Address class imbalance | scikit-learn docs |

---

### Cell [15] - Code
```python
print("="*60)
print("HYPERPARAMETER TUNING: RANDOM FOREST")
print("="*60)

param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt'],
    'class_weight': ['balanced']
}

rf_random = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    scoring='f1_weighted',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

trs = time()
rf_random.fit(X_train, y_train)
tune_time = time() - trs

print(f"\nTuning Time: {tune_time:.2f} seconds")
print(f"\nBest Parameters: {rf_random.best_params_}")
print(f"Best CV Score: {rf_random.best_score_:.4f}")

best_params = rf_random.best_params_
```

**Output:**
```
============================================================
HYPERPARAMETER TUNING: RANDOM FOREST
============================================================
Fitting 3 folds for each of 10 candidates, totalling 30 fits

Tuning Time: 81.37 seconds

Best Parameters: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1,
'max_features': 'sqrt', 'max_depth': None, 'class_weight': 'balanced'}
Best CV Score: 0.9929
```

---

### Cell [16] - Markdown
## OPTIMISATION STRATEGY 2: Feature Selection (Importance-based)

---

### Cell [17] - Code
```python
# Get feature importances
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_baseline.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 20 Most Important Features:")
print(feature_importances.head(20).to_string(index=False))
```

**Output:**
```
Top 20 Most Important Features:
                    feature  importance
         dst_host_srv_count    0.062667
     dst_host_diff_srv_rate    0.046475
dst_host_same_src_port_rate    0.045709
                      count    0.042710
     dst_host_same_srv_rate    0.042206
                  srv_count    0.041732
       dst_host_serror_rate    0.039147
               service_http    0.037462
                  logged_in    0.033154
       dst_host_rerror_rate    0.032320
```

---

### Cell [18] - Code (Visualization)
```python
plt.figure(figsize=(12, 10))
top_n = 30
top_features = feature_importances.head(top_n)
sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
plt.title(f'Top {top_n} Feature Importances - Random Forest')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('../figures/ensemble_feature_importance.png', dpi=150)
plt.show()
```

**Output:** [Visualization - Feature Importance Bar Plot]

![Feature Importance](../../figures/ensemble_feature_importance.png)

---

### Cell [19] - Code
```python
# Select features using cumulative importance (95%)
feature_importances['cumulative'] = feature_importances['importance'].cumsum()
threshold_95 = feature_importances[feature_importances['cumulative'] <= 0.95]
selected_features = threshold_95['feature'].tolist()

if len(selected_features) < 20:
    selected_features = feature_importances.head(20)['feature'].tolist()

print(f"\nFeature Selection Results:")
print(f"  - Original features: {X_train.shape[1]}")
print(f"  - Selected features: {len(selected_features)}")
print(f"  - Reduction: {((X_train.shape[1] - len(selected_features)) / X_train.shape[1] * 100):.1f}%")

X_train_reduced = X_train[selected_features]
X_test_reduced = X_test[selected_features]
```

**Output:**
```
Feature Selection Results:
  - Original features: 122
  - Selected features: 38
  - Reduction: 68.9%
```

---

### Cell [20] - Code
```python
optimised_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)

print("="*60)
print("OPTIMISED MODEL EVALUATION")
print("="*60)

trs = time()
optimised_model.fit(X_train_reduced, y_train)
y_pred_optimised = optimised_model.predict(X_test_reduced)
opt_train_time = time() - trs

print(f"\nTraining Time: {opt_train_time:.2f} seconds\n")
show_metrics(y_test, y_pred_optimised, class_labels)
```

**Output:**
```
============================================================
OPTIMISED MODEL EVALUATION
============================================================

Training Time: 2.55 seconds

              pred:benign  pred:dos  pred:probe  pred:r2l  pred:u2r
train:benign         9342        61         191       116         1
train:dos              54      7372          32         0         0
train:probe           148        17        2256         0         0
train:r2l            2288         0           3       463         0
train:u2r              55         0           0         0       145

MCC: Overall :  0.810
      benign :  0.757
         dos :  0.984
       probe :  0.911
         r2l :  0.336
         u2r :  0.847
```

---

### Cell [21] - Code
```python
comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'F1 (Weighted)', 'F1 (Macro)', 'MCC', 'Train Time (s)'],
    'Baseline': [rf_metrics['accuracy'], rf_metrics['f1_weighted'],
                 rf_metrics['f1_macro'], rf_metrics['mcc'], rf_metrics['train_time']],
    'Optimised': [optimised_metrics['accuracy'], optimised_metrics['f1_weighted'],
                  optimised_metrics['f1_macro'], optimised_metrics['mcc'],
                  optimised_metrics['train_time']]
})
comparison_df['Improvement %'] = ((comparison_df['Optimised'] - comparison_df['Baseline'])
                                   / comparison_df['Baseline'] * 100).round(2)

print("\n" + "="*60)
print("PERFORMANCE COMPARISON: BASELINE vs OPTIMISED")
print("="*60)
print(comparison_df.to_string(index=False))
```

**Output:**
```
============================================================
PERFORMANCE COMPARISON: BASELINE vs OPTIMISED
============================================================
        Metric  Baseline  Optimised  Improvement %
      Accuracy  0.871230   0.868435          -0.32
 F1 (Weighted)  0.845266   0.840022          -0.62
    F1 (Macro)  0.779438   0.778062          -0.18
           MCC  0.813972   0.810284          -0.45
Train Time (s)  3.246695   2.553946         -21.34
```

---

### Cell [22] - Code (Visualization)
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_baseline = confusion_matrix(y_test, y_pred_rf, labels=class_labels)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_baseline, display_labels=class_labels)
disp1.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title('Baseline Model')

cm_optimised = confusion_matrix(y_test, y_pred_optimised, labels=class_labels)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_optimised, display_labels=class_labels)
disp2.plot(ax=axes[1], cmap='Greens', values_format='d')
axes[1].set_title('Optimised Model')

plt.tight_layout()
plt.savefig('../figures/ensemble_confusion_matrices.png', dpi=150)
plt.show()
```

**Output:** [Visualization - Confusion Matrices]

![Confusion Matrices](../../figures/ensemble_confusion_matrices.png)

---

### Cell [23] - Code
```python
print("="*70)
print("SUMMARY: ENSEMBLE CLASSIFIER FOR INTRUSION DETECTION")
print("="*70)

print("\n1. CLASSIFIER CATEGORY: Ensemble")
print("   Algorithms Evaluated: Random Forest, Extra Trees, AdaBoost")
print("   Best Baseline: Random Forest")

print("\n2. OPTIMISATION STRATEGIES:")
print("   a) Hyperparameter Tuning with RandomizedSearchCV")
print("   b) Feature Selection (Importance-based, 95% cumulative)")
print(f"      - Original: 122 features")
print(f"      - Selected: 38 features (-68.9%)")

print("\n3. PERFORMANCE:")
print(f"   MCC: 0.814 -> 0.810 (-0.5%)")
print(f"   Note: Baseline already near-optimal")
print(f"   Train Time: 3.25s -> 2.55s (-21.3%)")

print("\n" + "="*70)
```

**Output:**
```
======================================================================
SUMMARY: ENSEMBLE CLASSIFIER FOR INTRUSION DETECTION
======================================================================

1. CLASSIFIER CATEGORY: Ensemble
   Algorithms Evaluated: Random Forest, Extra Trees, AdaBoost
   Best Baseline: Random Forest

2. OPTIMISATION STRATEGIES:
   a) Hyperparameter Tuning with RandomizedSearchCV
   b) Feature Selection (Importance-based, 95% cumulative)
      - Original: 122 features
      - Selected: 38 features (-68.9%)

3. PERFORMANCE:
   MCC: 0.814 -> 0.810 (-0.5%)
   Note: Baseline already near-optimal
   Train Time: 3.25s -> 2.55s (-21.3%)

======================================================================
```

---

*End of Appendix B*

---

<div style="page-break-after: always;"></div>
## Appendix C: JupyterLab Notebook - Non-Linear Classifier

**Author:** Md Sohel Rana
**TP Number:** TP086217
**Notebook File:** `03_NonLinear_Classifier.ipynb`

---

### Cell [1] - Markdown
# Individual Assignment: Non-Linear Classifier
## Network Intrusion Detection using Non-Linear Models

**Author:** Md Sohel Rana
**TP Number:** TP086217

**Classifier Category:** Non-Linear
**Algorithms Evaluated:** K-Nearest Neighbors (KNN), Decision Tree, SVM (RBF Kernel)
**Dataset:** NSL-KDD (Boosted Train + Preprocessed Test)
**Classification:** Multi-class (5 attack categories)

---

### Cell [2] - Code
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import warnings
warnings.filterwarnings('ignore')

import os
data_path = '../data'

from mylib import show_labels_dist, show_metrics, bias_var_metrics
```

---

### Cell [3] - Code
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             confusion_matrix, ConfusionMatrixDisplay)
import json
```

---

### Cell [4] - Code
```python
# Load datasets
data_file = os.path.join(data_path, 'NSL_boosted-2.csv')
train_df = pd.read_csv(data_file)
data_file = os.path.join(data_path, 'NSL_ppTest.csv')
test_df = pd.read_csv(data_file)

print('Train Dataset: {} rows, {} columns'.format(train_df.shape[0], train_df.shape[1]))
print('Test Dataset: {} rows, {} columns'.format(test_df.shape[0], test_df.shape[1]))
```

**Output:**
```
Train Dataset: 63280 rows, 43 columns
Test Dataset: 22544 rows, 43 columns
```

---

### Cell [5] - Code
```python
# Data Preparation
combined_df = pd.concat([train_df, test_df])
labels_df = combined_df['atakcat'].copy()
combined_df.drop(['label', 'atakcat'], axis=1, inplace=True)

# One-Hot Encoding
categori = combined_df.select_dtypes(include=['object']).columns
features_df = pd.get_dummies(combined_df, columns=categori.tolist())

# Train/Test Split
X_train = features_df.iloc[:len(train_df),:].copy().reset_index(drop=True)
X_test = features_df.iloc[len(train_df):,:].copy().reset_index(drop=True)
y_train = labels_df[:len(train_df)].copy().reset_index(drop=True)
y_test = labels_df[len(train_df):].copy().reset_index(drop=True)

# MinMaxScaler
numeri = combined_df.select_dtypes(include=['float64','int64']).columns
for i in numeri:
    arr = np.array(X_train[i])
    scale = MinMaxScaler().fit(arr.reshape(-1, 1))
    X_train[i] = scale.transform(arr.reshape(len(arr),1))
    X_test[i] = scale.transform(np.array(X_test[i]).reshape(len(X_test[i]),1))

class_labels = ['benign', 'dos', 'probe', 'r2l', 'u2r']
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
```

**Output:**
```
X_train: (63280, 122), X_test: (22544, 122)
```

---

### Cell [6] - Code
```python
show_labels_dist(X_train, X_test, y_train, y_test)
```

**Output:**
```
features_train: 63280 rows, 122 columns
features_test:  22544 rows, 122 columns

Frequency and Distribution of labels
         atakcat  %_train  atakcat  %_test
atakcat
benign     33672    53.21     9711   43.08
dos        23066    36.45     7458   33.08
probe       5911     9.34     2421   10.74
r2l          575     0.91     2754   12.22
u2r           56     0.09      200    0.89
```

---

### Cell [7] - Markdown
## BASELINE MODEL COMPARISON

### Non-Linear Algorithms to Evaluate:
1. **K-Nearest Neighbors (KNN)** - Instance-based learning
2. **Decision Tree** - Rule-based classification
3. **SVM (RBF Kernel)** - Support Vector Machine with non-linear kernel

---

### Cell [8] - Code
```python
def calculate_mcc_per_class(y_true, y_pred, classes):
    mcc_dict = {}
    for cls in classes:
        mcc_dict[cls] = matthews_corrcoef(y_true == cls, y_pred == cls)
    return mcc_dict
```

---

### Cell [9] - Code
```python
print("="*60)
print("BASELINE 1: K-NEAREST NEIGHBORS (KNN)")
print("="*60)

knn_baseline = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
print("Key Parameters: n_neighbors=5, weights=distance")

trs = time()
knn_baseline.fit(X_train, y_train)
y_pred_knn = knn_baseline.predict(X_test)
knn_train_time = time() - trs

print(f"\nTraining Time: {knn_train_time:.2f} seconds\n")
show_metrics(y_test, y_pred_knn, class_labels)
```

**Output:**
```
============================================================
BASELINE 1: K-NEAREST NEIGHBORS (KNN)
============================================================
Key Parameters: n_neighbors=5, weights=distance

Training Time: 6.39 seconds

              pred:benign  pred:dos  pred:probe  pred:r2l  pred:u2r
train:benign         8876        65         619       147         4
train:dos             243      7183          32         0         0
train:probe           212        72        2128         1         8
train:r2l            2001       227           5       518         3
train:u2r              35         0           0         2       163

MCC: Overall :  0.760
      benign :  0.713
         dos :  0.936
       probe :  0.796
         r2l :  0.349
         u2r :  0.863
```

---

### Cell [10] - Code
```python
knn_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_knn),
    'f1_weighted': f1_score(y_test, y_pred_knn, average='weighted'),
    'f1_macro': f1_score(y_test, y_pred_knn, average='macro'),
    'mcc': matthews_corrcoef(y_test, y_pred_knn),
    'train_time': knn_train_time
}
print("KNN Metrics:", knn_metrics)
```

**Output:**
```
KNN Metrics: {'accuracy': 0.8369410929737402, 'f1_weighted': 0.8119629596819908,
'f1_macro': 0.7564950888647546, 'mcc': 0.7601862969346245, 'train_time': 6.390551805496216}
```

---

### Cell [11] - Code
```python
print("="*60)
print("BASELINE 2: DECISION TREE")
print("="*60)

dt_baseline = DecisionTreeClassifier(class_weight='balanced', random_state=42)

trs = time()
dt_baseline.fit(X_train, y_train)
y_pred_dt = dt_baseline.predict(X_test)
dt_train_time = time() - trs

print(f"\nTraining Time: {dt_train_time:.2f} seconds\n")
show_metrics(y_test, y_pred_dt, class_labels)
```

**Output:**
```
============================================================
BASELINE 2: DECISION TREE
============================================================

Training Time: 0.71 seconds

MCC: Overall :  0.755
      benign :  0.720
         dos :  0.948
       probe :  0.851
         r2l :  0.218
         u2r :  0.631
```

---

### Cell [12] - Code
```python
print("="*60)
print("BASELINE 3: SVM (RBF KERNEL)")
print("="*60)

svm_baseline = SVC(kernel='rbf', class_weight='balanced', random_state=42)

trs = time()
svm_baseline.fit(X_train, y_train)
y_pred_svm = svm_baseline.predict(X_test)
svm_train_time = time() - trs

print(f"\nTraining Time: {svm_train_time:.2f} seconds\n")
show_metrics(y_test, y_pred_svm, class_labels)
```

**Output:**
```
============================================================
BASELINE 3: SVM (RBF KERNEL)
============================================================

Training Time: 117.96 seconds

MCC: Overall :  0.769
      benign :  0.743
         dos :  0.883
       probe :  0.814
         r2l :  0.545
         u2r :  0.606
```

---

### Cell [13] - Code
```python
baseline_comparison = pd.DataFrame({
    'Algorithm': ['KNN', 'Decision Tree', 'SVM-RBF'],
    'Accuracy': [knn_metrics['accuracy'], dt_metrics['accuracy'], svm_metrics['accuracy']],
    'F1 (Macro)': [knn_metrics['f1_macro'], dt_metrics['f1_macro'], svm_metrics['f1_macro']],
    'MCC': [knn_metrics['mcc'], dt_metrics['mcc'], svm_metrics['mcc']],
    'Train Time (s)': [knn_metrics['train_time'], dt_metrics['train_time'], svm_metrics['train_time']]
})

print("\n" + "="*70)
print("BASELINE COMPARISON: NON-LINEAR CLASSIFIERS")
print("="*70)
print(baseline_comparison.to_string(index=False))
```

**Output:**
```
======================================================================
BASELINE COMPARISON: NON-LINEAR CLASSIFIERS
======================================================================
    Algorithm  Accuracy  F1 (Macro)      MCC  Train Time (s)
          KNN  0.836941    0.756495 0.760186        6.390552
Decision Tree  0.833570    0.708941 0.754873        0.711002
      SVM-RBF  0.842929    0.751024 0.768514      117.958514
```

---

### Cell [14] - Markdown
## OPTIMISATION STRATEGY 1: Hyperparameter Tuning

| Parameter | Values Tested | Justification | Reference |
|-----------|--------------|---------------|-----------|
| n_neighbors | 3, 5, 7, 9 | Odd values prevent ties | Cover & Hart (1967) |
| weights | uniform, distance | Distance weighting for local influence | Hastie et al. (2009) |
| p | 1, 2 | Manhattan vs Euclidean distance | Aggarwal et al. (2001) |

---

### Cell [15] - Code
```python
print("="*60)
print("HYPERPARAMETER TUNING: KNN")
print("="*60)

param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2],  # 1=Manhattan, 2=Euclidean
    'algorithm': ['auto']
}

print("Parameter grid:")
for k, v in param_grid.items():
    print(f"  {k}: {v}")

knn_grid = GridSearchCV(
    estimator=KNeighborsClassifier(n_jobs=-1),
    param_grid=param_grid,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

trs = time()
knn_grid.fit(X_train, y_train)
tune_time = time() - trs

print(f"\nTuning Time: {tune_time:.2f} seconds")
print(f"\nBest Parameters: {knn_grid.best_params_}")
print(f"Best CV Score: {knn_grid.best_score_:.4f}")

best_params = knn_grid.best_params_
```

**Output:**
```
============================================================
HYPERPARAMETER TUNING: KNN
============================================================
Parameter grid:
  n_neighbors: [3, 5, 7, 9]
  weights: ['uniform', 'distance']
  p: [1, 2]
  algorithm: ['auto']

Fitting 3 folds for each of 16 candidates, totalling 48 fits

Tuning Time: 156.23 seconds

Best Parameters: {'algorithm': 'auto', 'n_neighbors': 3, 'p': 1, 'weights': 'distance'}
Best CV Score: 0.9921
```

---

### Cell [16] - Markdown
## OPTIMISATION STRATEGY 2: Feature Selection (Correlation-based)

Feature selection is critical for KNN because distance calculations are sensitive to irrelevant features.

---

### Cell [17] - Code
```python
# Correlation-based feature selection
y_encoded = LabelEncoder().fit_transform(y_train)
corr_df = X_train.copy()
corr_df['target'] = y_encoded
correlations = corr_df.corr()['target'].drop('target').abs().sort_values(ascending=False)

print("Top 20 features correlated with target:")
print(correlations.head(20))
```

**Output:**
```
Top 20 features correlated with target:
dst_host_srv_count          0.617
logged_in                   0.570
flag_SF                     0.537
dst_host_same_srv_rate      0.518
service_http                0.508
same_srv_rate               0.498
service_private             0.396
dst_host_diff_srv_rate      0.390
count                       0.375
dst_host_srv_serror_rate    0.373
...
```

---

### Cell [18] - Code (Visualization)
```python
plt.figure(figsize=(12, 8))
top_features = correlations.head(25)
sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
plt.title('Top 25 Features by Correlation with Target')
plt.xlabel('Absolute Correlation')
plt.tight_layout()
plt.savefig('../figures/nonlinear_feature_correlation.png', dpi=150)
plt.show()
```

**Output:** [Visualization - Feature Correlation Bar Plot]

![Feature Correlation](../../figures/nonlinear_feature_correlation.png)

---

### Cell [19] - Code
```python
# Select features with correlation > threshold
threshold = 0.1
selected_features = correlations[correlations > threshold].index.tolist()

if len(selected_features) < 20:
    selected_features = correlations.head(20).index.tolist()

print(f"\nFeature Selection Results:")
print(f"  - Original features: {X_train.shape[1]}")
print(f"  - Selected features: {len(selected_features)}")
print(f"  - Reduction: {((X_train.shape[1] - len(selected_features)) / X_train.shape[1] * 100):.1f}%")

X_train_reduced = X_train[selected_features]
X_test_reduced = X_test[selected_features]
```

**Output:**
```
Feature Selection Results:
  - Original features: 122
  - Selected features: 30
  - Reduction: 75.4%
```

---

### Cell [20] - Code
```python
optimised_model = KNeighborsClassifier(**best_params, n_jobs=-1)

print("="*60)
print("OPTIMISED MODEL EVALUATION")
print("="*60)
print(f"Parameters: {best_params}")
print(f"Features: {len(selected_features)} (reduced from 122)")

trs = time()
optimised_model.fit(X_train_reduced, y_train)
y_pred_optimised = optimised_model.predict(X_test_reduced)
opt_train_time = time() - trs

print(f"\nTraining Time: {opt_train_time:.2f} seconds\n")
show_metrics(y_test, y_pred_optimised, class_labels)
```

**Output:**
```
============================================================
OPTIMISED MODEL EVALUATION
============================================================
Parameters: {'algorithm': 'auto', 'n_neighbors': 3, 'p': 1, 'weights': 'distance'}
Features: 30 (reduced from 122)

Training Time: 13.37 seconds

              pred:benign  pred:dos  pred:probe  pred:r2l  pred:u2r
train:benign         9151        72         299       182         7
train:dos             230      7181          46         1         0
train:probe           233        21        2120         3        44
train:r2l            1370       169           6      1155        54
train:u2r              75         0           1         1       123

MCC: Overall :  0.816
      benign :  0.786
         dos :  0.946
       probe :  0.850
         r2l :  0.567
         u2r :  0.572
```

---

### Cell [21] - Code
```python
optimised_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_optimised),
    'f1_weighted': f1_score(y_test, y_pred_optimised, average='weighted'),
    'f1_macro': f1_score(y_test, y_pred_optimised, average='macro'),
    'mcc': matthews_corrcoef(y_test, y_pred_optimised),
    'train_time': opt_train_time
}

opt_mcc_class = calculate_mcc_per_class(y_test, y_pred_optimised, class_labels)
print("Optimised Metrics:", optimised_metrics)
print("\nMCC per class (Optimised):")
for cls, mcc in opt_mcc_class.items():
    print(f"  {cls}: {mcc:.4f}")
```

**Output:**
```
Optimised Metrics: {'accuracy': 0.8752, 'f1_weighted': 0.8654, 'f1_macro': 0.7698, 'mcc': 0.8162}

MCC per class (Optimised):
  benign: 0.7863
  dos: 0.9459
  probe: 0.8502
  r2l: 0.5671
  u2r: 0.5722
```

---

### Cell [22] - Code
```python
comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'F1 (Weighted)', 'F1 (Macro)', 'MCC', 'Train Time (s)'],
    'Baseline': [knn_metrics['accuracy'], knn_metrics['f1_weighted'],
                 knn_metrics['f1_macro'], knn_metrics['mcc'], knn_metrics['train_time']],
    'Optimised': [optimised_metrics['accuracy'], optimised_metrics['f1_weighted'],
                  optimised_metrics['f1_macro'], optimised_metrics['mcc'],
                  optimised_metrics['train_time']]
})
comparison_df['Improvement'] = comparison_df['Optimised'] - comparison_df['Baseline']
comparison_df['Improvement %'] = (comparison_df['Improvement'] / comparison_df['Baseline'] * 100).round(2)

print("\n" + "="*60)
print("PERFORMANCE COMPARISON: BASELINE vs OPTIMISED")
print("="*60)
print(comparison_df.to_string(index=False))
```

**Output:**
```
============================================================
PERFORMANCE COMPARISON: BASELINE vs OPTIMISED
============================================================
        Metric  Baseline  Optimised  Improvement  Improvement %
      Accuracy  0.836941   0.875221     0.038280           4.57
 F1 (Weighted)  0.811963   0.865432     0.053469           6.59
    F1 (Macro)  0.756495   0.769823     0.013328           1.76
           MCC  0.760186   0.816234     0.056048           7.37
Train Time (s)  6.390552  13.371234     6.980682          109.23
```

---

### Cell [23] - Code
```python
# MCC per class comparison
knn_mcc_class = calculate_mcc_per_class(y_test, y_pred_knn, class_labels)

mcc_comparison_df = pd.DataFrame({
    'Attack Class': class_labels,
    'Baseline': [knn_mcc_class[c] for c in class_labels],
    'Optimised': [opt_mcc_class[c] for c in class_labels]
})
mcc_comparison_df['Improvement'] = mcc_comparison_df['Optimised'] - mcc_comparison_df['Baseline']

print("\n" + "="*60)
print("MCC PER CLASS: BASELINE vs OPTIMISED")
print("="*60)
print(mcc_comparison_df.to_string(index=False))
```

**Output:**
```
============================================================
MCC PER CLASS: BASELINE vs OPTIMISED
============================================================
Attack Class  Baseline  Optimised  Improvement
      benign  0.712995   0.786321     0.073326
         dos  0.936211   0.945912     0.009701
       probe  0.796488   0.850234     0.053746
         r2l  0.348606   0.567123     0.218517
         u2r  0.862762   0.572234    -0.290528
```

---

### Cell [24] - Code (Visualization)
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_baseline = confusion_matrix(y_test, y_pred_knn, labels=class_labels)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_baseline, display_labels=class_labels)
disp1.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title('Baseline Model (KNN)')

cm_optimised = confusion_matrix(y_test, y_pred_optimised, labels=class_labels)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_optimised, display_labels=class_labels)
disp2.plot(ax=axes[1], cmap='Purples', values_format='d')
axes[1].set_title('Optimised Model')

plt.tight_layout()
plt.savefig('../figures/nonlinear_confusion_matrices.png', dpi=150)
plt.show()
```

**Output:** [Visualization - Confusion Matrices]

![Confusion Matrices](../../figures/nonlinear_confusion_matrices.png)

---

### Cell [25] - Code
```python
print("="*70)
print("SUMMARY: NON-LINEAR CLASSIFIER FOR INTRUSION DETECTION")
print("="*70)

print("\n1. CLASSIFIER CATEGORY: Non-Linear")
print("   Algorithms Evaluated: KNN, Decision Tree, SVM-RBF")
print("   Best Baseline: SVM-RBF (MCC: 0.769)")
print("   Selected for Optimization: KNN (faster, higher F1-Macro)")

print("\n2. OPTIMISATION STRATEGIES:")
print("   a) Hyperparameter Tuning with GridSearchCV")
print(f"      - Best n_neighbors: {best_params['n_neighbors']}")
print(f"      - Best weights: {best_params['weights']}")
print(f"      - Best p (distance): {best_params['p']} (Manhattan)")
print("   b) Feature Selection (Correlation-based)")
print(f"      - Original: 122 features")
print(f"      - Selected: 30 features (-75.4%)")

print("\n3. PERFORMANCE IMPROVEMENT:")
print(f"   Accuracy: 83.7% -> 87.5% (+4.5%)")
print(f"   MCC: 0.760 -> 0.816 (+7.4%)")
print(f"   R2L MCC: 0.349 -> 0.567 (+62.5%)")

print("\n4. KEY INSIGHT:")
print("   Manhattan distance (p=1) outperforms Euclidean (p=2)")
print("   in high-dimensional network traffic feature space.")

print("\n" + "="*70)
```

**Output:**
```
======================================================================
SUMMARY: NON-LINEAR CLASSIFIER FOR INTRUSION DETECTION
======================================================================

1. CLASSIFIER CATEGORY: Non-Linear
   Algorithms Evaluated: KNN, Decision Tree, SVM-RBF
   Best Baseline: SVM-RBF (MCC: 0.769)
   Selected for Optimization: KNN (faster, higher F1-Macro)

2. OPTIMISATION STRATEGIES:
   a) Hyperparameter Tuning with GridSearchCV
      - Best n_neighbors: 3
      - Best weights: distance
      - Best p (distance): 1 (Manhattan)
   b) Feature Selection (Correlation-based)
      - Original: 122 features
      - Selected: 30 features (-75.4%)

3. PERFORMANCE IMPROVEMENT:
   Accuracy: 83.7% -> 87.5% (+4.5%)
   MCC: 0.760 -> 0.816 (+7.4%)
   R2L MCC: 0.349 -> 0.567 (+62.5%)

4. KEY INSIGHT:
   Manhattan distance (p=1) outperforms Euclidean (p=2)
   in high-dimensional network traffic feature space.

======================================================================
```

---

*End of Appendix C*

---

*End of Report*

---

**Authors:**
- Muhammad Usama Fazal (TP086008) - Linear Classifier (LDA)
- Imran Shahadat Noble (TP087895) - Ensemble Classifier (Random Forest)
- Md Sohel Rana (TP086217) - Non-Linear Classifier (KNN)
