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
