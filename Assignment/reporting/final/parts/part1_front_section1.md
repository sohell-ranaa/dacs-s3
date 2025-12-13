# FRONT COVER

---

**CT115-3-M DATA ANALYTICS IN CYBER SECURITY**

**GROUP ASSIGNMENT**

---

**Group Members:**

| Name | Student ID |
|------|------------|
| Muhammad Usama Fazal | TP086008 |
| Imran Shahadat Noble | TP087895 |
| Md Sohel Rana | TP086217 |

**Intake Code:** UC2F2408CS

**Module Title:** Data Analytics in Cyber Security

**Assignment Title:** Machine Learning-Based Network Intrusion Detection System

**Submission Date:** December 2024

---

<div style="page-break-after: always;"></div>

## TABLE OF CONTENTS

1. [Combined Review of Selected Algorithms](#1-combined-review-of-selected-algorithms)
   - 1.1 Introduction
   - 1.2 Algorithm Classification Taxonomy
   - 1.3 Linear Classifier: Linear Discriminant Analysis (Muhammad Usama Fazal - TP086008)
   - 1.4 Non-Linear Classifier: K-Nearest Neighbors (Md Sohel Rana - TP086217)
   - 1.5 Ensemble Classifier: Random Forest (Imran Shahadat Noble - TP087895)
   - 1.6 Summary of Algorithm Characteristics

2. [Integrated Performance Discussion](#2-integrated-performance-discussion)
   - 2.1 Experimental Setup
   - 2.2 Performance Metrics Selection
   - 2.3 Comparative Analysis of Optimised Models
   - 2.4 Key Findings and Insights
   - 2.5 Recommendations for Deployment

3. [Individual Reports](#3-individual-reports)
   - 3.1 Linear Discriminant Analysis (Muhammad Usama Fazal - TP086008)
   - 3.2 Random Forest (Imran Shahadat Noble - TP087895)
   - 3.3 K-Nearest Neighbors (Md Sohel Rana - TP086217)

4. [Conclusions and Recommendations](#4-conclusions-and-recommendations)

5. [References](#5-references)

6. [Appendices](#6-appendices)
   - Appendix A: JupyterLab Notebook - Linear Classifier (Muhammad Usama Fazal)
   - Appendix B: JupyterLab Notebook - Ensemble Classifier (Imran Shahadat Noble)
   - Appendix C: JupyterLab Notebook - Non-Linear Classifier (Md Sohel Rana)

---

<div style="page-break-after: always;"></div>

# 1. COMBINED REVIEW OF SELECTED ALGORITHMS

**Contributors:** Muhammad Usama Fazal (TP086008), Imran Shahadat Noble (TP087895), Md Sohel Rana (TP086217)

## 1.1 Introduction

Network intrusion detection is a critical component of modern cybersecurity infrastructure. As cyber threats continue to evolve in sophistication and frequency, the need for intelligent, automated detection systems has become paramount. Machine learning offers a promising approach to this challenge by enabling systems to learn patterns from historical network traffic data and identify anomalous behaviour indicative of potential attacks.

### 1.1.1 The Cybersecurity Context

Modern organisations face an increasingly hostile threat landscape characterised by:

- **Volume:** Billions of network packets traverse enterprise networks daily, making manual inspection infeasible
- **Velocity:** Zero-day attacks and advanced persistent threats (APTs) require rapid detection and response
- **Variety:** Attack vectors range from simple port scans to sophisticated multi-stage intrusions
- **Evasion:** Attackers actively modify techniques to bypass signature-based detection systems

Traditional rule-based Intrusion Detection Systems (IDS) rely on predefined signatures to identify known attacks. While effective against documented threats, these systems struggle with novel attack patterns and generate high false positive rates. Machine learning-based approaches address these limitations by learning generalised patterns from data rather than relying on explicit rules.

### 1.1.2 Study Objectives

This report presents a comprehensive study of machine learning-based network intrusion detection using the NSL-KDD dataset, a refined version of the widely-used KDD Cup 1999 dataset. The NSL-KDD dataset addresses several inherent problems of the original dataset, including the removal of redundant records and the provision of a more balanced representation of attack types (Tavallaee et al., 2009).

The objective of this study is to evaluate and compare three distinct classification algorithms representing different methodological approaches to pattern recognition:

1. **Linear Discriminant Analysis (LDA)** - A linear classification method - **Muhammad Usama Fazal (TP086008)**
2. **K-Nearest Neighbors (KNN)** - A non-linear, instance-based learning method - **Md Sohel Rana (TP086217)**
3. **Random Forest** - An ensemble method using bagging - **Imran Shahadat Noble (TP087895)**

Each algorithm was implemented with both baseline (default parameters) and optimised configurations to evaluate the impact of various optimisation strategies on **multi-class classification performance** across five attack categories.

### 1.1.3 Multi-Class Classification Task

Unlike binary classification (Normal vs Attack), this study addresses the more challenging **multi-class classification problem** with five categories:

**Table 1: Class Distribution in NSL-KDD Dataset**

| Class | Description | Training % | Test % |
|-------|-------------|------------|--------|
| **Benign** | Normal network traffic | 53.21% | 43.08% |
| **DoS** | Denial of Service attacks | 36.45% | 33.08% |
| **Probe** | Surveillance/scanning attacks | 9.34% | 10.74% |
| **R2L** | Remote-to-Local attacks | 0.91% | 12.22% |
| **U2R** | User-to-Root attacks | 0.09% | 0.89% |

The severe class imbalance, particularly for R2L and U2R classes, presents a significant challenge for all classifiers.

## 1.2 Algorithm Classification Taxonomy

Machine learning classification algorithms can be organised into distinct categories based on their underlying mathematical principles and learning mechanisms.

![Figure 1: Algorithm Classification Taxonomy](../../figures/fig0_algorithm_taxonomy.png)

*Figure 1: Classification of machine learning algorithms showing the taxonomy of methods evaluated in this study. The three selected algorithms represent different categories: Linear (LDA), Non-Linear (KNN), and Ensemble (Random Forest).*

The selection of algorithms from different categories ensures diversity in the approaches evaluated:
- **Linear Methods:** Linear Discriminant Analysis (LDA) - **Muhammad Usama Fazal (TP086008)**
- **Non-Linear Methods:** K-Nearest Neighbors (KNN) - **Md Sohel Rana (TP086217)**
- **Ensemble Methods (Bagging):** Random Forest - **Imran Shahadat Noble (TP087895)**

This diversity is essential because different algorithm types may exhibit varying strengths and weaknesses when applied to network intrusion detection, where the data often contains complex, non-linear patterns and class imbalances.

## 1.3 Linear Classifier: Linear Discriminant Analysis

**Author:** Muhammad Usama Fazal (TP086008)

Linear Discriminant Analysis, introduced by Ronald Fisher in 1936, is a classical statistical method for dimensionality reduction and classification. LDA seeks to find a linear combination of features that best separates two or more classes of objects (Hastie et al., 2009).

### Mathematical Foundation

LDA operates on the principle of maximising the ratio of between-class variance to within-class variance. Given a set of observations with known class labels, LDA projects the data onto a lower-dimensional space that maximises class separability. The algorithm assumes that:

- The data for each class follows a multivariate Gaussian distribution
- All classes share a common covariance matrix (homoscedasticity)
- The features are continuous and not perfectly collinear

### Characteristics for Intrusion Detection

**Advantages:**
- Computational efficiency makes it suitable for real-time detection scenarios
- Provides interpretable decision boundaries that can be examined by security analysts
- Effective when class distributions are approximately Gaussian
- Natural dimensionality reduction capability helps handle high-dimensional network data

**Limitations:**
- The assumption of linear separability may not hold for complex attack patterns
- Sensitive to outliers, which are common in network traffic data
- May underperform when dealing with non-linear decision boundaries between normal and attack traffic
- Struggles with multi-class problems involving overlapping class distributions

## 1.4 Non-Linear Classifier: K-Nearest Neighbors

**Author:** Md Sohel Rana (TP086217)

K-Nearest Neighbors is an instance-based learning algorithm that classifies new observations based on similarity measures in the feature space. Unlike parametric methods, KNN makes no assumptions about the underlying data distribution, making it highly flexible (Cover & Hart, 1967).

### Algorithmic Principle

The KNN algorithm operates by:
1. Storing all training instances in memory
2. Computing distances between a new instance and all stored instances
3. Identifying the k nearest neighbours
4. Assigning the majority class among these neighbours to the new instance

### Characteristics for Intrusion Detection

**Advantages:**
- No assumptions about data distribution make it robust to various attack patterns
- Naturally handles multi-class classification problems
- Can capture complex, non-linear decision boundaries
- Simple to understand and implement

**Limitations:**
- Computationally expensive at prediction time, requiring distance calculations with all training instances
- Sensitive to irrelevant features, which can distort distance calculations
- The curse of dimensionality affects performance in high-dimensional spaces
- Requires careful selection of the k parameter and distance metric

## 1.5 Ensemble Classifier (Bagging): Random Forest

**Author:** Imran Shahadat Noble (TP087895)

Random Forest, proposed by Leo Breiman in 2001, is an ensemble learning method that constructs multiple decision trees during training and combines their predictions through majority voting (Breiman, 2001).

### Ensemble Mechanism

Random Forest employs two key randomisation techniques:
1. **Bootstrap Aggregating (Bagging):** Each tree is trained on a random subset of the training data, sampled with replacement
2. **Random Feature Selection:** At each node split, only a random subset of features is considered

This dual randomisation creates decorrelated trees whose collective prediction is more accurate and stable than any individual tree.

### Characteristics for Intrusion Detection

**Advantages:**
- Robust to noise and outliers through ensemble averaging
- Provides feature importance rankings useful for identifying key network attributes
- Handles high-dimensional data without requiring feature selection
- Resistant to overfitting due to the averaging of multiple trees

**Limitations:**
- Less interpretable than single decision trees
- Can be computationally intensive for very large datasets
- May not perform optimally when the number of relevant features is very small relative to the total number of features

## 1.6 Summary of Algorithm Characteristics

**Table 2: Comparison of Selected Classification Algorithms**

| Characteristic | LDA (Muhammad Usama Fazal) | KNN (Md Sohel Rana) | Random Forest (Imran Shahadat Noble) |
|---------------|-----|-----|---------------|
| **Category** | Linear | Non-Linear | Ensemble (Bagging) |
| **Training Complexity** | Low | None (Lazy) | Medium |
| **Prediction Speed** | Fast | Slow | Medium |
| **Interpretability** | High | Medium | Low |
| **Handles Non-linearity** | No | Yes | Yes |
| **Feature Importance** | No | No | Yes |
| **Sensitivity to Outliers** | High | Medium | Low |
| **Hyperparameter Sensitivity** | Low | High | Low |
| **Multi-class Capability** | Native | Native | Native |

The diversity in these characteristics justifies the selection of these three algorithms, as each brings unique strengths to the task of network intrusion detection.

---
