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
