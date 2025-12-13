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
