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

