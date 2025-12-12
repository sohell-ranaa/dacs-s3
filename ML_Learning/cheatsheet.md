# ML Quick Reference Cheatsheet

## Import Statements
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
```

---

## Load Data
```python
df = pd.read_csv('filename.csv')
X = df.drop('target_column', axis=1)
y = df['target_column']
```

---

## Explore Data
```python
df.shape               # (rows, columns)
df.head()              # First 5 rows
df.info()              # Data types
df.describe()          # Statistics
df.isnull().sum()      # Missing values
df['col'].value_counts()  # Class distribution
```

---

## Split Data
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

## Scale Data
```python
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train
X_test_scaled = scaler.transform(X_test)        # Transform only
```

---

## Train Model
```python
# The pattern (same for ALL models!)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## Evaluate Model
```python
# Quick check
accuracy_score(y_test, predictions)

# Full report
print(classification_report(y_test, predictions))

# Confusion matrix
confusion_matrix(y_test, predictions)

# F1 score
f1_score(y_test, predictions, average='weighted')
f1_score(y_test, predictions, average='macro')
```

---

## Hyperparameter Tuning
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_weighted'
)
grid.fit(X_train, y_train)

print(grid.best_params_)
best_model = grid.best_estimator_
```

---

## Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=10)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
```

---

## Handle Class Imbalance
```python
# Easiest method: class weights
model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)
```

---

## Compare Models
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = [
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('KNN', KNeighborsClassifier()),
    ('Random Forest', RandomForestClassifier(random_state=42))
]

for name, model in models:
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## Your Assignment Summary

**Dataset:** NSL-KDD (Network Intrusion Detection)

**Classes:** benign, dos, probe, r2l, u2r

**Steps:**
1. Load and explore data
2. Preprocess (encode + scale)
3. Train BASELINE model (default params)
4. Apply ONE optimization:
   - Hyperparameter tuning
   - Feature selection
   - Class weights (recommended!)
5. Compare baseline vs optimized
6. Write report

**Key insight:** The data is HIGHLY IMBALANCED (u2r has only 52 samples!)
Use `class_weight='balanced'` to handle this.

---

## Common Classifiers

| Algorithm | Code | Best For |
|-----------|------|----------|
| Decision Tree | `DecisionTreeClassifier()` | Interpretable models |
| KNN | `KNeighborsClassifier(n_neighbors=5)` | Small datasets |
| Random Forest | `RandomForestClassifier(n_estimators=100)` | Most problems |
| Logistic Regression | `LogisticRegression()` | Binary classification |
| SVM | `SVC()` | High-dimensional data |

---

## Metrics to Use

| Data Type | Metric | Why |
|-----------|--------|-----|
| Balanced | Accuracy | All classes equal |
| Imbalanced | F1-macro | Treats classes equally |
| Security tasks | Recall | Catch all attacks |
| General | F1-weighted | Considers class sizes |
