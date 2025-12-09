# ASIA PACIFIC UNIVERSITY OF TECHNOLOGY AND INNOVATION

---

## FRONT COVER

---

**CT115-3-M DATA ANALYTICS IN CYBER SECURITY**

**GROUP ASSIGNMENT**

**Machine Learning-Based Network Intrusion Detection System**

---

**Group Members:**

| No. | Name | Student ID |
|-----|------|------------|
| 1 | [Member 1 Name] | [TP000000] |
| 2 | [Member 2 Name] | [TP000000] |
| 3 | [Member 3 Name] | [TP000000] |

**Intake Code:** [UC2F0000CS]

**Module Title:** Data Analytics in Cyber Security

**Assignment Title:** Group Assignment - Network Intrusion Detection

**Submission Date:** [Date]

---

<div style="page-break-after: always;"></div>

## EXECUTIVE SUMMARY

This report presents a comprehensive evaluation of machine learning-based approaches for network intrusion detection using the NSL-KDD benchmark dataset. Three classification algorithms representing different methodological paradigms were implemented and compared: Linear Discriminant Analysis (linear method), K-Nearest Neighbors (non-linear method), and Random Forest (ensemble method).

### Key Findings

| Metric | Best Performer | Score |
|--------|----------------|-------|
| **Overall (F1-Score)** | K-Nearest Neighbors | 0.902 |
| **Precision** | Linear Discriminant Analysis | 0.969 |
| **Recall** | K-Nearest Neighbors | 0.855 |
| **Consistency** | Random Forest | 0.880 |

### Principal Conclusions

1. **KNN emerged as the top performer** with an F1-Score of 0.902 after optimisation, demonstrating that proper hyperparameter tuning (k=7, Manhattan distance, distance weighting) can yield substantial improvements (+5.0%).

2. **Non-linear methods significantly outperform linear approaches** for intrusion detection, with KNN and Random Forest achieving recall rates of 85.5% and 80.9% respectively, compared to LDA's 64.6%.

3. **Algorithm-specific optimisation is essential** - strategies effective for one classifier may not transfer to others. Feature selection thresholds and hyperparameter sensitivity vary significantly across methods.

### Practical Recommendations

- **For maximum detection accuracy:** Deploy optimised KNN (F1: 0.90)
- **For real-time requirements:** Use LDA with shrinkage (F1: 0.77)
- **For balanced deployment:** Random Forest with defaults (F1: 0.88)

---

<div style="page-break-after: always;"></div>

## TABLE OF CONTENTS

1. [Combined Review of Selected Algorithms](#chapter-1-combined-review-of-selected-algorithms)
   - 1.1 Introduction
   - 1.2 Algorithm Classification Taxonomy
   - 1.3 Linear Classifier: Linear Discriminant Analysis
   - 1.4 Non-Linear Classifier: K-Nearest Neighbors
   - 1.5 Ensemble Classifier: Random Forest
   - 1.6 Summary of Algorithm Characteristics

2. [Integrated Performance Discussion](#chapter-2-integrated-performance-discussion)
   - 2.1 Experimental Setup
   - 2.2 Performance Metrics Selection
   - 2.3 Comparative Analysis of Optimised Models
   - 2.4 Key Findings and Insights
   - 2.5 Recommendations for Deployment

3. [Individual Chapter: Linear Discriminant Analysis](#chapter-3-individual-chapter---linear-discriminant-analysis-lda)
   - 3.1 Algorithm Overview
   - 3.2 Optimisation Strategies Applied
   - 3.3 Baseline vs Optimised Model Comparison
   - 3.4 Analysis and Discussion

4. [Individual Chapter: Random Forest](#chapter-4-individual-chapter---random-forest)
   - 4.1 Algorithm Overview
   - 4.2 Optimisation Strategies Applied
   - 4.3 Baseline vs Optimised Model Comparison
   - 4.4 Analysis and Discussion

5. [Individual Chapter: K-Nearest Neighbors](#chapter-5-individual-chapter---k-nearest-neighbors-knn)
   - 5.1 Algorithm Overview
   - 5.2 Optimisation Strategies Applied
   - 5.3 Baseline vs Optimised Model Comparison
   - 5.4 Analysis and Discussion

6. [Conclusions and Recommendations](#chapter-6-conclusions-and-recommendations)

7. [References](#references)

8. [Appendices](#appendices)
   - Appendix A: Jupyter Notebooks
   - Appendix B: Figures and Visualisations
   - Appendix C: Dataset Information

---

<div style="page-break-after: always;"></div>

# CHAPTER 1: COMBINED REVIEW OF SELECTED ALGORITHMS

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

1. **Linear Discriminant Analysis (LDA)** - A linear classification method
2. **K-Nearest Neighbors (KNN)** - A non-linear, instance-based learning method
3. **Random Forest** - An ensemble method using bagging

Each algorithm was implemented with both baseline (default parameters) and optimised configurations to evaluate the impact of various optimisation strategies on classification performance.

## 1.2 Algorithm Classification Taxonomy

Machine learning classification algorithms can be organised into distinct categories based on their underlying mathematical principles and learning mechanisms. A visual taxonomy of classification algorithms highlighting the three classifiers selected for this study is presented in *Appendix B, Figure B1*.

> **[VISUAL: Figure B1 - Algorithm Classification Taxonomy]**
> *Place in Appendix B: A hierarchical diagram showing classification algorithm categories (Linear Methods, Non-Linear Methods, Ensemble Methods) with LDA, KNN, and Random Forest highlighted as the selected algorithms.*

The selection of algorithms from different categories ensures diversity in the approaches evaluated:
- **Linear Methods:** Linear Discriminant Analysis (LDA)
- **Non-Linear Methods:** K-Nearest Neighbors (KNN)
- **Ensemble Methods (Bagging):** Random Forest

This diversity is essential because different algorithm types may exhibit varying strengths and weaknesses when applied to network intrusion detection, where the data often contains complex, non-linear patterns and class imbalances.

## 1.3 Linear Classifier: Linear Discriminant Analysis

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

## 1.4 Non-Linear Classifier: K-Nearest Neighbors

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

Table 1 summarises the key characteristics of the three selected algorithms, providing a comparative overview of their properties relevant to network intrusion detection.

**Table 1: Comparison of Selected Classification Algorithms**

| Characteristic | LDA | KNN | Random Forest |
|---------------|-----|-----|---------------|
| **Category** | Linear | Non-Linear | Ensemble (Bagging) |
| **Training Complexity** | Low | None (Lazy) | Medium |
| **Prediction Speed** | Fast | Slow | Medium |
| **Interpretability** | High | Medium | Low |
| **Handles Non-linearity** | No | Yes | Yes |
| **Feature Importance** | No | No | Yes |
| **Sensitivity to Outliers** | High | Medium | Low |
| **Hyperparameter Sensitivity** | Low | High | Low |

The diversity in these characteristics justifies the selection of these three algorithms, as each brings unique strengths to the task of network intrusion detection.

---

<div style="page-break-after: always;"></div>

# CHAPTER 2: INTEGRATED PERFORMANCE DISCUSSION

## 2.1 Experimental Setup

### 2.1.1 Dataset Description

The NSL-KDD dataset used in this study consists of network connection records, each described by 41 features and labelled as either normal traffic or one of four attack categories: DoS (Denial of Service), Probe, R2L (Remote-to-Local), and U2R (User-to-Root).

**Table 2: Dataset Composition**

| Dataset | Records | Normal | Attack | Attack Ratio |
|---------|---------|--------|--------|--------------|
| Training Set | 63,280 | 28,447 | 34,833 | 55.0% |
| Test Set | 22,544 | 9,711 | 12,833 | 56.9% |

For this study, the multi-class labels were converted to binary classification (Normal vs Attack) to focus on the fundamental task of distinguishing malicious from benign traffic.

### 2.1.2 Data Preprocessing Pipeline

A consistent preprocessing pipeline was applied across all classifiers to ensure fair comparison. The complete workflow is illustrated in *Appendix B, Figure B11*.

> **[VISUAL: Figure B11 - Data Preprocessing and Experimental Workflow]**
> *Place in Appendix B: Flowchart showing: Raw Data → Categorical Encoding → Feature Scaling → Feature Selection → Model Training → Evaluation. Include branching paths for baseline vs optimised models.*

**Step 1: Categorical Encoding**
One-hot encoding was applied to three categorical features:
- **protocol_type:** tcp, udp, icmp (3 categories)
- **service:** http, ftp, smtp, etc. (70 categories)
- **flag:** SF, S0, REJ, etc. (11 categories)

This expanded the feature space from 41 to 122 features.

**Step 2: Feature Scaling**
MinMax normalisation (0-1 range) was applied to all numeric features:
$$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

This ensures equal contribution to distance-based calculations (critical for KNN) and improves convergence for gradient-based methods.

**Step 3: Train-Test Split**
The pre-defined NSL-KDD train and test splits were used to maintain consistency with benchmark studies and ensure reproducibility.

### 2.1.3 Computational Environment

All experiments were conducted using Python 3.x with the scikit-learn machine learning library (Pedregosa et al., 2011). The consistent software environment ensures reproducibility and fair comparison across all classifiers.

## 2.2 Performance Metrics Selection

For network intrusion detection systems, the choice of evaluation metrics is crucial as different metrics capture different aspects of classifier performance. The following metrics were selected based on their relevance to cybersecurity applications:

**Table 3: Selected Performance Metrics and Their Significance**

| Metric | Formula | Significance for IDS |
|--------|---------|---------------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness of predictions |
| **Precision** | TP/(TP+FP) | Proportion of detected attacks that are real attacks; high precision reduces false alarms |
| **Recall** | TP/(TP+FN) | Proportion of real attacks that are detected; critical for security as missed attacks are costly |
| **F1-Score** | 2*(Precision*Recall)/(Precision+Recall) | Harmonic mean providing balanced measure |
| **MCC** | Matthews Correlation Coefficient | Robust measure for imbalanced datasets |
| **AUC** | Area Under ROC Curve | Overall discriminative ability across all thresholds |

Where TP = True Positives (attacks correctly identified), TN = True Negatives (normal traffic correctly identified), FP = False Positives (false alarms), and FN = False Negatives (missed attacks).

For intrusion detection, **Recall is particularly important** because the cost of missing an attack (false negative) typically far exceeds the cost of a false alarm (false positive). However, excessively low precision leads to alert fatigue among security analysts, reducing the practical utility of the system.

## 2.3 Comparative Analysis of Optimised Models

### 2.3.1 Overall Performance Comparison

A multi-dimensional comparison of the three optimised classifiers is visualised using a radar chart and heatmap in *Appendix B, Figures B2-B3*. These visualisations allow simultaneous assessment of all six performance metrics across classifiers.

> **[VISUAL: Figure B2 - Multi-Metric Performance Radar Chart]**
> *Place in Appendix B: Radar chart comparing LDA, KNN, and Random Forest across Accuracy, Precision, Recall, F1-Score, MCC, and AUC metrics. KNN should show the most balanced profile.*

> **[VISUAL: Figure B3 - Performance Metrics Heatmap]**
> *Place in Appendix B: Colour-coded heatmap with classifiers as rows and metrics as columns. Darker colours indicate higher scores.*

### 2.3.2 Quantitative Results

Table 4 presents the complete numerical results for all optimised models.

**Table 4: Optimised Model Performance Metrics**

| Classifier | Accuracy | Precision | Recall | F1-Score | MCC | AUC |
|------------|----------|-----------|--------|----------|-----|-----|
| LDA | 0.786 | 0.969 | 0.646 | 0.775 | 0.631 | 0.948 |
| Random Forest | 0.874 | 0.964 | 0.809 | 0.880 | 0.762 | 0.970 |
| **KNN** | **0.894** | 0.955 | **0.855** | **0.902** | **0.794** | 0.934 |

The results reveal that **KNN achieved the highest F1-Score (0.902)**, indicating the best balance between precision and recall among all tested classifiers.

### 2.3.3 Impact of Optimisation

The effect of optimisation on model performance is illustrated in *Appendix B, Figure B4*, which compares baseline and optimised F1-Scores.

> **[VISUAL: Figure B4 - Baseline vs Optimised Model Performance]**
> *Place in Appendix B: Grouped bar chart comparing baseline (grey) and optimised (coloured) F1-Scores for each classifier. Include percentage change labels.*

**Table 5: Optimisation Impact Summary**

| Classifier | Baseline F1 | Optimised F1 | Change | Verdict |
|------------|-------------|--------------|--------|---------|
| LDA | 0.764 | 0.775 | +1.4% | Improved |
| Random Forest | 0.883 | 0.880 | -0.4% | Maintained |
| KNN | 0.859 | 0.902 | **+5.0%** | **Significantly Improved** |

### 2.3.4 Feature Reduction Analysis

An important aspect of optimisation was feature selection to reduce dimensionality and potentially improve model performance and efficiency. The feature selection outcomes are visualised in *Appendix B, Figure B5*.

> **[VISUAL: Figure B5 - Feature Selection Results]**
> *Place in Appendix B: Dual-panel chart showing (left) absolute feature counts selected vs removed, and (right) percentage reduction for each classifier.*

**Table 6: Feature Reduction Summary**

| Classifier | Original Features | Selected Features | Reduction |
|------------|-------------------|-------------------|-----------|
| LDA | 122 | 29 | 76.2% |
| Random Forest | 122 | 29 | 76.2% |
| KNN | 122 | 75 | 38.5% |

The varying feature requirements across classifiers reflect their different sensitivities to irrelevant features. KNN's distance-based nature makes it sensitive to all features, hence requiring more features for optimal performance.

## 2.4 Key Findings and Insights

### 2.4.1 Precision-Recall Trade-off Analysis

The precision-recall trade-off among the classifiers is visualised in *Appendix B, Figure B6*, with bubble size representing F1-Score.

> **[VISUAL: Figure B6 - Precision-Recall Trade-off Analysis]**
> *Place in Appendix B: Scatter/bubble plot with precision on y-axis, recall on x-axis. Bubble size represents F1-Score. Different colours for each classifier. KNN should appear in the top-right region.*

### 2.4.2 Key Insights

1. **Best Overall Performer:** KNN with optimised parameters achieved the highest F1-Score (0.902), demonstrating that a well-tuned instance-based learner can outperform more complex ensemble methods for this dataset.

2. **Linear Methods Limitation:** LDA achieved the highest precision (0.969) but suffered from significantly lower recall (0.646), missing approximately 35% of attacks. This highlights the limitation of linear decision boundaries for capturing complex attack patterns.

3. **Ensemble Methods Stability:** Random Forest showed consistent performance with minimal change after optimisation (-0.4%), suggesting its default parameters are already well-suited for intrusion detection tasks.

4. **Feature Selection Impact:** The optimal feature reduction varies by algorithm. KNN required more features (38.5% reduction) compared to LDA and Random Forest (76.2% reduction), reflecting KNN's distance-based sensitivity to all features.

5. **Distance Metric Importance:** KNN's improvement was largely attributed to using Manhattan distance (p=1) instead of Euclidean distance, which is more robust in high-dimensional spaces.

### 2.4.3 Ranking of Classifiers

The final classifier ranking based on F1-Score is presented in *Appendix B, Figure B7*.

> **[VISUAL: Figure B7 - Final Classifier Ranking]**
> *Place in Appendix B: Podium-style or horizontal bar chart showing final rankings: 1st KNN (0.902), 2nd Random Forest (0.880), 3rd LDA (0.775). Use gold/silver/bronze colour coding.*

**Table 7: Final Classifier Ranking**

| Rank | Classifier | F1-Score | Key Strength |
|------|------------|----------|--------------|
| 1st | KNN | 0.902 | Best overall balance |
| 2nd | Random Forest | 0.880 | Consistent, minimal tuning |
| 3rd | LDA | 0.775 | Fast, interpretable |

## 2.5 Recommendations for Deployment

Based on the comprehensive evaluation, the following recommendations are provided for deploying machine learning-based intrusion detection systems:

### For Maximum Detection Accuracy:
**Recommended Classifier:** K-Nearest Neighbors
- Configuration: k=7, distance weighting, Manhattan distance (p=1)
- Feature selection threshold: correlation > 0.05
- Trade-off: Higher computational cost at prediction time

### For Real-Time Detection Requirements:
**Recommended Classifier:** Linear Discriminant Analysis
- Configuration: lsqr solver with automatic shrinkage
- Feature selection: correlation-based (threshold > 0.1)
- Trade-off: Lower recall, may miss some sophisticated attacks

### For Balanced Performance and Robustness:
**Recommended Classifier:** Random Forest
- Configuration: Default or slightly tuned parameters
- Feature selection: Optional, based on computational constraints
- Trade-off: Less interpretable than LDA

---

<div style="page-break-after: always;"></div>

# CHAPTER 3: INDIVIDUAL CHAPTER - LINEAR DISCRIMINANT ANALYSIS (LDA)

**Author:** [Member 1 Name]

## 3.1 Algorithm Overview

Linear Discriminant Analysis (LDA) was selected as the representative linear classifier for this study. LDA is a classical statistical method that finds a linear combination of features to separate classes by maximising the ratio of between-class variance to within-class variance.

In the context of network intrusion detection, LDA attempts to find a hyperplane that best separates normal traffic from attack traffic based on the extracted network features. The simplicity and interpretability of LDA make it an attractive baseline for comparison with more complex methods.

## 3.2 Optimisation Strategies Applied

### 3.2.1 Strategy 1: Hyperparameter Tuning

The baseline LDA model used default parameters from the scikit-learn implementation. For optimisation, the following parameters were evaluated:

**Solver Selection:**
- **svd:** Singular Value Decomposition, default solver that does not compute covariance matrix
- **lsqr:** Least Squares solution, can be combined with shrinkage
- **eigen:** Eigenvalue decomposition, can be combined with shrinkage

**Shrinkage Parameter:**
- **None:** No regularisation applied
- **auto:** Automatic shrinkage using Ledoit-Wolf lemma
- **0.5:** Manual shrinkage coefficient

After evaluation, the optimal configuration was determined to be:
- **Solver:** lsqr
- **Shrinkage:** auto

The lsqr solver with automatic shrinkage was selected because it provides regularisation that improves numerical stability when dealing with features that may be collinear. Automatic shrinkage adapts the regularisation strength based on the data characteristics, avoiding the need for manual tuning.

### 3.2.2 Strategy 2: Feature Selection via Correlation Analysis

Correlation-based feature selection was implemented to identify features most relevant to the classification task:

1. Computed Pearson correlation coefficient between each feature and the binary target variable
2. Ranked features by absolute correlation value
3. Selected features with absolute correlation greater than 0.1

**Result:** The feature space was reduced from 122 features to 29 features, representing a 76.2% reduction.

This aggressive feature reduction serves multiple purposes:
- Removes noisy features that may degrade LDA's performance
- Reduces computational requirements for both training and prediction
- Focuses the model on the most discriminative network attributes

The selected 29 features represent the network connection attributes most strongly associated with attack behaviour, including features related to connection duration, protocol flags, and service-specific indicators.

## 3.3 Baseline vs Optimised Model Comparison

The detailed performance comparison between baseline and optimised LDA models is visualised in *Appendix B, Figure B8*.

> **[VISUAL: Figure B8 - LDA Performance Analysis]**
> *Place in Appendix B: Grouped bar chart comparing baseline (grey) and optimised (blue) LDA across all six metrics. Include confusion matrices for both models side by side. Add ROC curve comparison.*

**Table 8: LDA Baseline vs Optimised Performance**

| Metric | Baseline | Optimised | Absolute Change | Relative Change |
|--------|----------|-----------|-----------------|-----------------|
| Accuracy | 0.772 | 0.786 | +0.014 | +1.8% |
| Precision | 0.926 | 0.969 | +0.043 | +4.6% |
| Recall | 0.650 | 0.646 | -0.004 | -0.6% |
| F1-Score | 0.764 | 0.775 | +0.011 | +1.4% |
| MCC | 0.588 | 0.631 | +0.043 | +7.3% |
| AUC | 0.886 | 0.948 | +0.062 | +7.0% |

## 3.4 Analysis and Discussion

### 3.4.1 Performance Interpretation

The optimised LDA model demonstrates improvement across most metrics, with particularly notable gains in:

1. **AUC (+7.0%):** The substantial increase in AUC indicates that the optimised model has better overall discriminative ability across different classification thresholds. This improvement is attributed to both the regularisation effect of shrinkage and the removal of noisy features.

2. **MCC (+7.3%):** The Matthews Correlation Coefficient improvement suggests better correlation between predictions and actual labels, indicating more reliable classification performance.

3. **Precision (+4.6%):** Higher precision means fewer false alarms when the model predicts an attack. This is beneficial for reducing alert fatigue among security analysts.

### 3.4.2 Recall Limitation

The most significant limitation observed is the relatively low recall (0.646), meaning the model misses approximately 35% of actual attacks. This limitation arises from:

1. **Linear Decision Boundary:** LDA assumes that classes can be separated by a linear hyperplane. Network intrusion data often contains non-linear patterns where attacks may cluster in multiple regions of the feature space.

2. **Gaussian Distribution Assumption:** LDA assumes features follow a Gaussian distribution within each class. Network traffic data often violates this assumption, particularly for categorical features that were one-hot encoded.

3. **Attack Diversity:** The four attack categories (DoS, Probe, R2L, U2R) may have distinct feature distributions that cannot be captured by a single linear boundary.

### 3.4.3 Practical Implications

Despite its limitations, LDA offers practical advantages:

- **Speed:** Training and prediction are extremely fast, making LDA suitable for real-time applications
- **Interpretability:** The linear coefficients can be examined to understand which features contribute most to classification decisions
- **Baseline Performance:** The 0.775 F1-Score provides a reasonable baseline that can be complemented by other methods in an ensemble system

### 3.4.4 Recommendations

For production deployment, LDA should be considered as:
1. A fast first-stage filter in a multi-stage detection system
2. A baseline for anomaly detection where computational resources are limited
3. A complementary classifier in an ensemble with non-linear methods to improve overall coverage

---

<div style="page-break-after: always;"></div>

# CHAPTER 4: INDIVIDUAL CHAPTER - RANDOM FOREST

**Author:** [Member 2 Name]

## 4.1 Algorithm Overview

Random Forest was selected as the representative ensemble method using bagging. This algorithm constructs multiple decision trees during training and outputs the class that is the mode of the individual trees' predictions, leveraging the wisdom of crowds principle.

For network intrusion detection, Random Forest offers robustness against noise and the ability to capture non-linear relationships between network features and attack indicators. Its built-in feature importance mechanism also provides valuable insights into which network attributes are most indicative of malicious activity.

## 4.2 Optimisation Strategies Applied

### 4.2.1 Strategy 1: Hyperparameter Tuning

The following hyperparameters were optimised:

**Number of Estimators (n_estimators):**
- Controls the number of trees in the forest
- Tested values: 50, 100, 150, 200
- **Selected:** 150 trees

**Maximum Depth (max_depth):**
- Limits the depth of each decision tree
- Tested values: 10, 15, 20, None
- **Selected:** 20

**Minimum Samples Split (min_samples_split):**
- Minimum samples required to split an internal node
- Tested values: 2, 5, 10
- **Selected:** 5

**Minimum Samples Leaf (min_samples_leaf):**
- Minimum samples required in a leaf node
- Tested values: 1, 2, 4
- **Selected:** 2

### 4.2.2 Strategy 2: Feature Selection Based on Feature Importance

Random Forest provides built-in feature importance scores based on the mean decrease in impurity (Gini importance) when features are used for splitting:

1. Trained a Random Forest model on all features
2. Extracted feature importance scores
3. Ranked features by importance
4. Selected features contributing to 95% cumulative importance

**Result:** Reduced from 122 to 29 features (76.2% reduction)

### 4.2.3 Strategy 3: Class Imbalance Handling

Applied class weighting to address the imbalanced nature of the dataset:
- **class_weight='balanced'**: Automatically adjusts weights inversely proportional to class frequencies
- This gives more importance to the minority class during training

## 4.3 Baseline vs Optimised Model Comparison

The detailed performance comparison between baseline and optimised Random Forest models is visualised in *Appendix B, Figure B9*.

> **[VISUAL: Figure B9 - Random Forest Performance Analysis]**
> *Place in Appendix B: Grouped bar chart comparing baseline (grey) and optimised (green) Random Forest across all six metrics. Include confusion matrices for both models. Add feature importance bar chart showing top 10 features.*

**Table 9: Random Forest Baseline vs Optimised Performance**

| Metric | Baseline | Optimised | Absolute Change | Relative Change |
|--------|----------|-----------|-----------------|-----------------|
| Accuracy | 0.876 | 0.874 | -0.002 | -0.2% |
| Precision | 0.955 | 0.964 | +0.009 | +0.9% |
| Recall | 0.821 | 0.809 | -0.012 | -1.5% |
| F1-Score | 0.883 | 0.880 | -0.003 | -0.3% |
| MCC | 0.763 | 0.762 | -0.001 | -0.1% |
| AUC | 0.971 | 0.970 | -0.001 | -0.1% |

## 4.4 Analysis and Discussion

### 4.4.1 Performance Interpretation

The Random Forest classifier demonstrated remarkably stable performance between baseline and optimised configurations. This stability can be attributed to several factors:

1. **Well-Designed Default Parameters:** The scikit-learn implementation of Random Forest uses carefully chosen default values that perform well across a wide range of datasets. The default configuration already provides excellent performance for intrusion detection.

2. **Inherent Robustness:** Random Forest's ensemble nature provides natural regularisation through the averaging of multiple trees. This makes the algorithm less sensitive to hyperparameter choices compared to single tree models.

3. **Self-Regularisation:** The combination of bagging and random feature selection at each node split creates diverse trees that collectively provide robust predictions without extensive tuning.

### 4.4.2 Effect of Class Weighting

The class_weight='balanced' setting had a notable effect:
- **Increased Precision (+0.9%):** Fewer false positives
- **Decreased Recall (-1.5%):** Slightly more missed attacks

This trade-off occurs because balanced weighting forces the model to pay more attention to the minority class (normal traffic), which can improve precision but may cause some attack patterns to be classified as normal.

### 4.4.3 Feature Importance Insights

The feature importance analysis revealed the most discriminative network attributes for intrusion detection:

**Top 5 Most Important Features:**
1. **src_bytes:** Number of data bytes from source to destination
2. **dst_bytes:** Number of data bytes from destination to source
3. **count:** Number of connections to the same host in the past 2 seconds
4. **srv_count:** Number of connections to the same service in the past 2 seconds
5. **same_srv_rate:** Percentage of connections to the same service

These features relate to traffic volume and connection patterns, which are key indicators of DoS attacks (high volume) and Probe attacks (systematic scanning behaviour).

### 4.4.4 Practical Implications

Random Forest's consistent high performance with minimal tuning makes it an excellent choice for practical deployment:

- **Reliability:** The 0.880 F1-Score is achieved consistently without extensive optimisation
- **Feature Insights:** Built-in importance scores guide security analysts to focus on key network attributes
- **Parallelisability:** Tree construction can be parallelised for faster training on large datasets

### 4.4.5 Recommendations

Random Forest is recommended as:
1. The primary classifier when reliability and minimal tuning are priorities
2. A feature importance tool for understanding attack signatures
3. Part of an ensemble system leveraging its consistent baseline performance

---

<div style="page-break-after: always;"></div>

# CHAPTER 5: INDIVIDUAL CHAPTER - K-NEAREST NEIGHBORS (KNN)

**Author:** [Member 3 Name]

## 5.1 Algorithm Overview

K-Nearest Neighbors was selected as the representative non-linear, instance-based classifier. KNN classifies new observations based on the majority class among the k nearest training instances in the feature space, making no assumptions about the underlying data distribution.

For network intrusion detection, KNN's ability to capture complex decision boundaries makes it well-suited for identifying attack patterns that may not be linearly separable from normal traffic.

## 5.2 Optimisation Strategies Applied

### 5.2.1 Strategy 1: Hyperparameter Tuning

KNN's performance is highly sensitive to its hyperparameters. The following were optimised:

**Number of Neighbors (k):**
- Controls the number of nearest neighbors considered for voting
- Tested values: 3, 5, 7, 9, 11
- **Selected:** k=7

The choice of k=7 balances between:
- Small k (e.g., 3): More sensitive to noise but captures local patterns
- Large k (e.g., 11): More stable but may smooth out important boundaries

**Voting Weights:**
- **uniform:** All neighbors contribute equally
- **distance:** Closer neighbors have more influence
- **Selected:** distance

Distance weighting improves accuracy by giving more importance to neighbors that are most similar to the query point.

**Distance Metric (p parameter):**
- p=1: Manhattan distance (L1 norm)
- p=2: Euclidean distance (L2 norm)
- **Selected:** p=1 (Manhattan distance)

Manhattan distance was selected because:
1. More robust to outliers in high-dimensional spaces
2. Less affected by the curse of dimensionality
3. Better suited for mixed feature types after one-hot encoding

### 5.2.2 Strategy 2: Feature Selection via Correlation Analysis

Feature selection is particularly important for KNN due to its sensitivity to irrelevant features:

1. Computed correlation between each feature and the target
2. Applied a threshold of 0.05 (less aggressive than LDA's 0.1)
3. Selected features meeting the threshold

**Result:** Reduced from 122 to 75 features (38.5% reduction)

The less aggressive threshold recognises KNN's ability to handle non-linear relationships where features with lower individual correlation may still contribute to classification in combination with others.

## 5.3 Baseline vs Optimised Model Comparison

The detailed performance comparison between baseline and optimised KNN models is visualised in *Appendix B, Figure B10*.

> **[VISUAL: Figure B10 - KNN Performance Analysis]**
> *Place in Appendix B: Grouped bar chart comparing baseline (grey) and optimised (red) KNN across all six metrics. Include confusion matrices for both models. Show the significant improvement in Recall (+6.6%) and MCC (+11.8%) with annotation arrows.*

**Table 10: KNN Baseline vs Optimised Performance**

| Metric | Baseline | Optimised | Absolute Change | Relative Change |
|--------|----------|-----------|-----------------|-----------------|
| Accuracy | 0.850 | 0.894 | +0.044 | +5.2% |
| Precision | 0.925 | 0.955 | +0.030 | +3.2% |
| Recall | 0.802 | 0.855 | +0.053 | +6.6% |
| F1-Score | 0.859 | 0.902 | +0.043 | +5.0% |
| MCC | 0.710 | 0.794 | +0.084 | +11.8% |
| AUC | 0.883 | 0.934 | +0.051 | +5.8% |

## 5.4 Analysis and Discussion

### 5.4.1 Performance Interpretation

KNN achieved the **most significant improvement** among all classifiers after optimisation, with a 5.0% increase in F1-Score and 11.8% increase in MCC. This substantial improvement can be attributed to:

1. **Distance Weighting Effect:** Switching from uniform to distance weighting allows the model to be more confident when query points are very close to training instances of a particular class. This is especially effective for network intrusion data where some attacks closely resemble normal traffic.

2. **Manhattan Distance Advantage:** The L1 norm (Manhattan distance) is more robust in high-dimensional spaces because it treats each dimension equally rather than squaring differences. This is particularly beneficial for the one-hot encoded categorical features.

3. **Optimal k Selection:** k=7 provides sufficient smoothing to reduce noise sensitivity while preserving the ability to capture local patterns in the feature space.

### 5.4.2 Recall Improvement Analysis

The 6.6% improvement in recall is particularly significant for intrusion detection:

- Baseline detected 80.2% of attacks
- Optimised detected 85.5% of attacks
- This represents approximately 690 additional attacks detected in the test set

The improvement in attack detection comes from:
1. Better handling of borderline cases through distance weighting
2. Manhattan distance better capturing the structure of network traffic data
3. Feature selection removing noise that previously caused misclassifications

### 5.4.3 Computational Considerations

While KNN achieved the best performance, it comes with computational trade-offs:

**Training Time:** Minimal (lazy learning - just stores training data)

**Prediction Time:** O(n * d) where n is the number of training instances and d is the number of features. With 63,280 training instances and 75 features, each prediction requires significant computation.

**Memory Requirements:** Must store all training instances in memory

For real-time intrusion detection with high traffic volumes, these computational requirements may necessitate optimisation strategies such as:
- KD-trees or Ball-trees for faster neighbor search
- Approximate nearest neighbor algorithms
- Instance reduction techniques

### 5.4.4 Why KNN Outperformed the Ensemble Method

The superior performance of optimised KNN over Random Forest warrants explanation:

1. **Data Structure:** The NSL-KDD dataset may have cluster structures where attacks form distinct groups in the feature space that KNN can effectively identify.

2. **Proper Tuning:** While ensemble methods had well-tuned defaults, KNN's defaults (k=5, Euclidean distance, uniform weights) are less universally applicable, leaving more room for improvement.

3. **Non-Parametric Nature:** KNN makes no assumptions about data distribution, allowing it to adapt to the actual structure of network traffic data.

### 5.4.5 Recommendations

KNN is recommended as:
1. The primary classifier when detection accuracy is paramount and computational resources permit
2. A second-stage classifier for cases flagged as borderline by faster methods
3. A benchmark for evaluating other classifiers' ability to capture local patterns

The configuration of k=7, distance weighting, and Manhattan distance with correlation-based feature selection (threshold=0.05) is recommended for network intrusion detection applications.

---

<div style="page-break-after: always;"></div>

# CHAPTER 6: CONCLUSIONS AND RECOMMENDATIONS

## 6.1 Summary of Findings

This study evaluated three machine learning classifiers for network intrusion detection using the NSL-KDD dataset. The key findings are:

### 6.1.1 Classifier Performance Ranking

**Table 11: Final Classifier Ranking by F1-Score**

| Rank | Classifier | Category | F1-Score | Key Strength |
|------|------------|----------|----------|--------------|
| 1st | KNN | Non-Linear | 0.902 | Best balance of precision and recall |
| 2nd | Random Forest | Ensemble (Bagging) | 0.880 | Consistent performance, minimal tuning |
| 3rd | LDA | Linear | 0.775 | Fast, interpretable |

### 6.1.2 Key Insights

1. **Optimisation Impact Varies by Algorithm:** KNN benefited most from optimisation (+5.0% F1), while Random Forest showed minimal change (-0.4%), indicating its default parameters are already well-tuned.

2. **Feature Selection Must Be Algorithm-Specific:** LDA benefited from aggressive feature reduction (76.2%), while KNN required more features (only 38.5% reduction) due to its distance-based sensitivity.

3. **Distance Metrics Matter:** KNN's improvement was largely due to switching from Euclidean to Manhattan distance, which is more robust in high-dimensional spaces.

4. **Non-Linear Methods Excel:** Both KNN and Random Forest significantly outperformed the linear LDA classifier, demonstrating the importance of capturing non-linear patterns in network traffic data.

## 6.2 Recommendations for Production Deployment

### 6.2.1 Primary Recommendation

For network intrusion detection systems prioritising detection accuracy, **K-Nearest Neighbors with the following configuration** is recommended:

- k = 7 neighbors
- Distance-weighted voting
- Manhattan distance (p=1)
- Feature selection with correlation threshold > 0.05
- Expected F1-Score: 0.90

### 6.2.2 Alternative Configurations

**For Real-Time Systems:** LDA with shrinkage regularisation
- Trades accuracy for speed
- Expected F1-Score: 0.77

**For Balanced Requirements:** Random Forest with default parameters
- Minimal tuning required
- Expected F1-Score: 0.88

## 6.3 Limitations and Future Work

### 6.3.1 Limitations

1. **Binary Classification Only:** This study focused on Normal vs Attack classification. Multi-class classification for specific attack types would provide more actionable intelligence.

2. **Dataset Age:** While NSL-KDD is a standard benchmark, it may not fully represent modern network attacks.

3. **Static Model:** The trained models do not adapt to evolving attack patterns.

### 6.3.2 Future Work

1. **Multi-Class Extension:** Implement classifiers for specific attack type identification (DoS, Probe, R2L, U2R).

2. **Deep Learning:** Explore neural network approaches for automated feature learning.

3. **Online Learning:** Develop adaptive models that update with new attack patterns.

4. **Ensemble Systems:** Combine multiple classifiers in a voting ensemble for improved robustness.

5. **Real-World Validation:** Test models on contemporary network traffic datasets.

## 6.4 Concluding Remarks

This study demonstrates that machine learning provides effective tools for network intrusion detection. The optimised KNN classifier achieving 0.902 F1-Score represents strong performance suitable for practical deployment. The varying responses to optimisation across algorithms highlight the importance of algorithm-specific tuning strategies rather than one-size-fits-all approaches.

The selection of classifiers from different algorithmic categories (linear, non-linear, and ensemble methods) ensures comprehensive coverage of available methodologies and provides insights into their relative strengths for cybersecurity applications.

---

<div style="page-break-after: always;"></div>

# REFERENCES

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32. https://doi.org/10.1023/A:1010933404324

Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory, 13*(1), 21-27. https://doi.org/10.1109/TIT.1967.1053964

Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics, 7*(2), 179-188. https://doi.org/10.1111/j.1469-1809.1936.tb02137.x

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: Data mining, inference, and prediction* (2nd ed.). Springer.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.

Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed analysis of the KDD CUP 99 data set. In *Proceedings of the 2009 IEEE Symposium on Computational Intelligence for Security and Defense Applications* (pp. 1-6). IEEE. https://doi.org/10.1109/CISDA.2009.5356528

---

<div style="page-break-after: always;"></div>

# APPENDICES

## Appendix A: Individual Jupyter Notebooks

The following Jupyter notebooks contain the implementation code, performance statistics, and visualisations for each classifier:

**Table A1: Jupyter Notebook Files**

| Notebook | Classifier | Author | Location |
|----------|------------|--------|----------|
| 01_Linear_Classifier.ipynb | Linear Discriminant Analysis | Member 1 | notebooks/individual/ |
| 02_Ensemble_RandomForest.ipynb | Random Forest | Member 2 | notebooks/individual/ |
| 03_NonLinear_KNN.ipynb | K-Nearest Neighbors | Member 3 | notebooks/individual/ |

Each notebook contains:
- Data loading and preprocessing
- Baseline model training and evaluation
- Optimisation strategy implementation
- Optimised model training and evaluation
- Confusion matrices and ROC curves
- Performance comparison visualisations

## Appendix B: Figures and Visualisations

This appendix contains all figures referenced in the main report. All visualisations are generated from the Jupyter notebooks and Group_Comparative_Analysis.ipynb.

### Group Comparative Figures

**Figure B1: Algorithm Classification Taxonomy**
> *Hierarchical diagram showing the classification of machine learning algorithms into Linear Methods (LDA), Non-Linear Methods (KNN), and Ensemble Methods (Random Forest). The three algorithms selected for this study are highlighted.*

**Figure B2: Multi-Metric Performance Radar Chart**
> *Radar chart displaying performance of all three optimised classifiers across six metrics (Accuracy, Precision, Recall, F1-Score, MCC, AUC). Allows visual comparison of classifier profiles.*

**Figure B3: Performance Metrics Heatmap**
> *Colour-coded heatmap with classifiers as rows and metrics as columns. Darker colours (green) indicate higher scores. Facilitates quick identification of strengths and weaknesses.*

**Figure B4: Baseline vs Optimised Model Performance**
> *Grouped bar chart comparing F1-Scores before and after optimisation for each classifier. Include percentage change annotations.*

**Figure B5: Feature Selection Results**
> *Dual-panel visualisation: (Left) Stacked bar chart showing selected vs removed features; (Right) Pie/bar chart showing reduction percentages for each classifier.*

**Figure B6: Precision-Recall Trade-off Analysis**
> *Scatter/bubble plot positioning each classifier by precision (y-axis) and recall (x-axis). Bubble size represents F1-Score. Includes ideal region annotation.*

**Figure B7: Final Classifier Ranking**
> *Podium-style or horizontal bar chart showing final F1-Score rankings with gold/silver/bronze colour coding.*

### Individual Classifier Figures

**Figure B8: LDA Performance Analysis**
> *From 01_Linear_Classifier.ipynb: Grouped bar chart (baseline vs optimised), confusion matrices, ROC curve comparison.*

**Figure B9: Random Forest Performance Analysis**
> *From 02_Ensemble_RandomForest.ipynb: Grouped bar chart (baseline vs optimised), confusion matrices, feature importance bar chart (top 10 features).*

**Figure B10: KNN Performance Analysis**
> *From 03_NonLinear_KNN.ipynb: Grouped bar chart (baseline vs optimised), confusion matrices, ROC curve comparison. Highlight the significant improvement annotations.*

**Figure B11: Data Preprocessing and Experimental Workflow**
> *Flowchart illustrating the complete experimental pipeline from raw data through preprocessing, feature selection, model training (baseline and optimised paths), and evaluation.*

### Additional Figures from Individual Notebooks

Each individual notebook should also include:
- **Confusion Matrix (Baseline)**: 2x2 matrix showing TP, TN, FP, FN for baseline model
- **Confusion Matrix (Optimised)**: 2x2 matrix showing TP, TN, FP, FN for optimised model
- **ROC Curve Comparison**: Overlay of baseline and optimised ROC curves with AUC values
- **Feature Correlation Heatmap** (if applicable): Top correlated features with target variable

## Appendix C: Group Comparison Notebook

**File:** notebooks/group/Group_Comparative_Analysis.ipynb

This notebook contains:
- Consolidated preprocessing pipeline
- Cross-classifier performance comparison
- Statistical analysis of results
- Aggregate visualisations (Figures B1-B7)

## Appendix D: Dataset Information

**Table A2: NSL-KDD Dataset Features**

| Category | Feature Count | Examples |
|----------|---------------|----------|
| Basic | 9 | duration, protocol_type, service, flag |
| Content | 13 | num_failed_logins, logged_in, root_shell |
| Traffic | 9 | count, srv_count, serror_rate |
| Host | 10 | dst_host_count, dst_host_srv_count |

**Table A3: Attack Categories in NSL-KDD**

| Attack Type | Description | Examples |
|-------------|-------------|----------|
| DoS | Denial of Service | SYN flood, smurf, neptune |
| Probe | Surveillance/scanning | port scan, IP sweep |
| R2L | Remote to Local | password guessing, phishing |
| U2R | User to Root | buffer overflow, rootkit |

---

*End of Report*

---

**Word Count:** Approximately 5,500 words (excluding tables, figures, and appendices)

**Page Count:** Approximately 25-30 pages when formatted with 12pt font and 1.5 line spacing
