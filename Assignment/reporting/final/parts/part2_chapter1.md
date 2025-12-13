# 1. COMBINED REVIEW OF SELECTED ALGORITHMS

**Contributors:** Muhammad Usama Fazal (TP086008), Imran Shahadat Noble (TP087895), Md Sohel Rana (TP086217)

## 1.1 Introduction

Network intrusion detection is a critical component of modern cybersecurity infrastructure. As cyber threats continue to evolve, machine learning offers a promising approach by enabling systems to learn patterns from historical network traffic data and identify anomalous behaviour indicative of potential attacks.

This report presents a comprehensive study of machine learning-based network intrusion detection using the NSL-KDD dataset. The study evaluates three distinct classification algorithms representing different methodological approaches:

1. **Linear Discriminant Analysis (LDA)** - A linear classification method - **Muhammad Usama Fazal (TP086008)**
2. **K-Nearest Neighbors (KNN)** - A non-linear, instance-based learning method - **Md Sohel Rana (TP086217)**
3. **Random Forest** - An ensemble method using bagging - **Imran Shahadat Noble (TP087895)**

Each algorithm was implemented with both baseline (default parameters) and optimised configurations to evaluate the impact of various optimisation strategies on multi-class classification performance.

### 1.1.1 Multi-Class Classification Task

This study addresses the multi-class classification problem with five categories as shown in Table 1.

**Table 1: Class Distribution in NSL-KDD Dataset**

| Class | Description | Training % | Test % |
|-------|-------------|------------|--------|
| Benign | Normal network traffic | 53.21% | 43.08% |
| DoS | Denial of Service attacks | 36.45% | 33.08% |
| Probe | Surveillance/scanning attacks | 9.34% | 10.74% |
| R2L | Remote-to-Local attacks | 0.91% | 12.22% |
| U2R | User-to-Root attacks | 0.09% | 0.89% |

Table 1 presents the class distribution across training and test sets. The severe class imbalance is evident, particularly for R2L (0.91% training vs 12.22% test) and U2R (0.09% training vs 0.89% test) classes. This distribution shift between training and test sets poses a significant challenge for model generalisation, as classifiers trained on majority classes may struggle to correctly identify minority attack types during testing.

## 1.2 Algorithm Classification Overview

Machine learning classification algorithms can be organised into distinct categories based on their underlying mathematical principles. Figure 1 illustrates the taxonomy of algorithms evaluated in this study.

**Figure 1: Algorithm Classification Overview**

```
┌─────────────────────────────────────────────────────────────────┐
│                 CLASSIFICATION ALGORITHMS                        │
├─────────────────┬─────────────────────┬─────────────────────────┤
│     LINEAR      │     NON-LINEAR      │       ENSEMBLE          │
├─────────────────┼─────────────────────┼─────────────────────────┤
│ • LDA           │ • KNN               │ • Random Forest         │
│ • Logistic Reg. │ • Decision Tree     │ • Extra Trees           │
│ • Ridge         │ • SVM (RBF)         │ • AdaBoost              │
├─────────────────┼─────────────────────┼─────────────────────────┤
│   SELECTED:     │     SELECTED:       │      SELECTED:          │
│   LDA           │     KNN             │   Random Forest         │
│  (Usama Fazal)  │   (Sohel Rana)      │  (Imran Noble)          │
└─────────────────┴─────────────────────┴─────────────────────────┘
```

The selection of algorithms from different categories ensures diversity in the approaches evaluated. Linear methods assume separable classes via hyperplanes, non-linear methods capture complex decision boundaries, and ensemble methods combine multiple models for robust predictions. This diversity is essential because network traffic data often contains complex, non-linear patterns and class imbalances that different algorithms handle with varying effectiveness.

## 1.3 Linear Classifier: Linear Discriminant Analysis

**Author:** Muhammad Usama Fazal (TP086008)

Linear Discriminant Analysis (LDA), introduced by Ronald Fisher in 1936, is a classical statistical method for dimensionality reduction and classification. LDA seeks to find a linear combination of features that best separates two or more classes by maximising the ratio of between-class variance to within-class variance (Hastie et al., 2009).

**Key Assumptions:**
- Data for each class follows a multivariate Gaussian distribution
- All classes share a common covariance matrix (homoscedasticity)
- Features are continuous and not perfectly collinear

**Advantages for Intrusion Detection:**
- Computational efficiency suitable for real-time detection
- Interpretable decision boundaries for security analysts
- Natural dimensionality reduction capability

**Limitations:**
- Linear separability assumption may not hold for complex attack patterns
- Sensitive to outliers common in network traffic data
- Struggles with multi-class problems involving overlapping distributions

## 1.4 Non-Linear Classifier: K-Nearest Neighbors

**Author:** Md Sohel Rana (TP086217)

K-Nearest Neighbors (KNN) is an instance-based learning algorithm that classifies observations based on similarity measures in the feature space. Unlike parametric methods, KNN makes no assumptions about the underlying data distribution (Cover & Hart, 1967).

**Algorithmic Principle:**
1. Store all training instances in memory
2. Compute distances between new instance and all stored instances
3. Identify the k nearest neighbours
4. Assign the majority class among neighbours

**Advantages for Intrusion Detection:**
- No distribution assumptions - robust to various attack patterns
- Naturally handles multi-class classification
- Captures complex, non-linear decision boundaries

**Limitations:**
- Computationally expensive at prediction time
- Sensitive to irrelevant features affecting distance calculations
- Requires careful selection of k parameter and distance metric

## 1.5 Ensemble Classifier: Random Forest

**Author:** Imran Shahadat Noble (TP087895)

Random Forest, proposed by Leo Breiman in 2001, is an ensemble learning method that constructs multiple decision trees during training and combines their predictions through majority voting (Breiman, 2001).

**Ensemble Mechanism:**
1. **Bootstrap Aggregating (Bagging):** Each tree trained on random subset with replacement
2. **Random Feature Selection:** Only random subset of features considered at each split

**Advantages for Intrusion Detection:**
- Robust to noise and outliers through ensemble averaging
- Provides feature importance rankings for identifying key network attributes
- Resistant to overfitting due to averaging of multiple trees

**Limitations:**
- Less interpretable than single decision trees
- Can be computationally intensive for very large datasets

## 1.6 Summary of Algorithm Characteristics

Table 2 provides a comprehensive comparison of the three selected algorithms.

**Table 2: Comparison of Selected Classification Algorithms**

| Characteristic | LDA (Usama Fazal) | KNN (Sohel Rana) | Random Forest (Imran Noble) |
|---------------|-------------------|------------------|----------------------------|
| Category | Linear | Non-Linear | Ensemble (Bagging) |
| Training Complexity | Low | None (Lazy) | Medium |
| Prediction Speed | Fast | Slow | Medium |
| Interpretability | High | Medium | Low |
| Handles Non-linearity | No | Yes | Yes |
| Feature Importance | No | No | Yes |
| Sensitivity to Outliers | High | Medium | Low |
| Hyperparameter Sensitivity | Low | High | Low |

Table 2 summarises the key characteristics differentiating the three algorithms. LDA offers speed and interpretability but is limited by its linear assumption. KNN provides flexibility through its non-parametric nature but at the cost of prediction-time computation. Random Forest balances performance and robustness through ensemble averaging, making it particularly suitable for the noisy, imbalanced network intrusion detection task. These diverse characteristics justify the selection of all three algorithms for comprehensive evaluation.

---
