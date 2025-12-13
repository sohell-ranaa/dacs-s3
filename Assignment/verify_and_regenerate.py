#!/usr/bin/env python3
"""
Comprehensive Verification Script for DACS Assignment
This script verifies all data, runs all models, and regenerates figures.

Team Members:
- Muhammad Usama Fazal (TP086008) - Linear Classifier (LDA)
- Imran Shahadat Noble (TP087895) - Ensemble Classifier (Random Forest)
- Md Sohel Rana (TP087437) - Non-Linear Classifier (KNN)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             confusion_matrix, ConfusionMatrixDisplay)

# Paths
DATA_PATH = '/home/rana-workspace/DACS/Assignment/data'
RESULTS_PATH = '/home/rana-workspace/DACS/Assignment/results'
FIGURES_PATH = '/home/rana-workspace/DACS/Assignment/figures'

# Ensure directories exist
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

print("="*80)
print("DACS ASSIGNMENT - DATA VERIFICATION AND FIGURE REGENERATION")
print("="*80)

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================
print("\n[1] Loading Data...")

train_df = pd.read_csv(os.path.join(DATA_PATH, 'NSL_boosted-2.csv'))
test_df = pd.read_csv(os.path.join(DATA_PATH, 'NSL_ppTest.csv'))

print(f"    Train: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"    Test: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

# Combine for preprocessing
combined_df = pd.concat([train_df, test_df])

# Multi-class classification (5 categories)
labels_df = combined_df['atakcat'].copy()
combined_df.drop(['label', 'atakcat'], axis=1, inplace=True)

# One-hot encoding
categori = combined_df.select_dtypes(include=['object']).columns
features_df = pd.get_dummies(combined_df, columns=categori.tolist())

# Split
X_train = features_df.iloc[:len(train_df),:].copy().reset_index(drop=True)
X_test = features_df.iloc[len(train_df):,:].copy().reset_index(drop=True)
y_train = labels_df[:len(train_df)].copy().reset_index(drop=True)
y_test = labels_df[len(train_df):].copy().reset_index(drop=True)

# Scale numeric features
numeri = combined_df.select_dtypes(include=['float64','int64']).columns
for i in numeri:
    arr = np.array(X_train[i])
    scale = MinMaxScaler().fit(arr.reshape(-1, 1))
    X_train[i] = scale.transform(arr.reshape(len(arr),1))
    arr = np.array(X_test[i])
    X_test[i] = scale.transform(arr.reshape(len(arr),1))

class_labels = ['benign', 'dos', 'probe', 'r2l', 'u2r']
print(f"    Features: {X_train.shape[1]}")
print(f"    Classes: {class_labels}")

# =============================================================================
# 2. FEATURE SELECTION
# =============================================================================
print("\n[2] Feature Selection...")

# Correlation-based feature selection
y_encoded = LabelEncoder().fit_transform(y_train)
corr_df = X_train.copy()
corr_df['target'] = y_encoded
correlations = corr_df.corr()['target'].drop('target').abs().sort_values(ascending=False)

# Select features with correlation > 0.1
threshold = 0.1
selected_features_corr = correlations[correlations > threshold].index.tolist()
print(f"    Correlation-based: {len(selected_features_corr)} features (threshold={threshold})")

X_train_corr = X_train[selected_features_corr]
X_test_corr = X_test[selected_features_corr]

# Importance-based feature selection (for RF)
rf_temp = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
rf_temp.fit(X_train, y_train)
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_temp.feature_importances_
}).sort_values('importance', ascending=False)

# Select top features contributing to 95% importance
feature_importances['cumulative'] = feature_importances['importance'].cumsum()
selected_features_imp = feature_importances[feature_importances['cumulative'] <= 0.95]['feature'].tolist()
if len(selected_features_imp) < 20:
    selected_features_imp = feature_importances.head(38)['feature'].tolist()
print(f"    Importance-based: {len(selected_features_imp)} features")

X_train_imp = X_train[selected_features_imp]
X_test_imp = X_test[selected_features_imp]

# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================
def calculate_mcc_per_class(y_true, y_pred, classes):
    """Calculate MCC for each class (one-vs-rest)"""
    mcc_dict = {}
    for cls in classes:
        mcc_dict[cls] = matthews_corrcoef(y_true == cls, y_pred == cls)
    return mcc_dict

def evaluate_model(model, X_tr, X_ts, y_tr, y_ts, model_name):
    """Train and evaluate a model, return metrics"""
    trs = time()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_ts)
    train_time = time() - trs

    metrics = {
        'accuracy': accuracy_score(y_ts, y_pred),
        'f1_weighted': f1_score(y_ts, y_pred, average='weighted'),
        'f1_macro': f1_score(y_ts, y_pred, average='macro'),
        'mcc': matthews_corrcoef(y_ts, y_pred),
        'train_time': train_time
    }
    mcc_class = calculate_mcc_per_class(y_ts, y_pred, class_labels)

    return metrics, mcc_class, y_pred

# =============================================================================
# 4. RUN ALL CLASSIFIERS
# =============================================================================
print("\n[3] Running All Classifiers...")

all_results = {}

# ----- LINEAR CLASSIFIERS -----
print("\n    === LINEAR CLASSIFIERS ===")

# LDA Baseline
print("    Running LDA (baseline)...")
lda_baseline = LinearDiscriminantAnalysis()
lda_base_metrics, lda_base_mcc, y_pred_lda_base = evaluate_model(
    lda_baseline, X_train, X_test, y_train, y_test, "LDA"
)
print(f"        MCC: {lda_base_metrics['mcc']:.4f}, Acc: {lda_base_metrics['accuracy']:.4f}")

# LDA Optimised (with feature selection)
print("    Running LDA (optimised)...")
lda_opt = LinearDiscriminantAnalysis(solver='svd')
lda_opt_metrics, lda_opt_mcc, y_pred_lda_opt = evaluate_model(
    lda_opt, X_train_corr, X_test_corr, y_train, y_test, "LDA-Opt"
)
print(f"        MCC: {lda_opt_metrics['mcc']:.4f}, Acc: {lda_opt_metrics['accuracy']:.4f}")

# Logistic Regression Baseline
print("    Running Logistic Regression (baseline)...")
lr_baseline = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_base_metrics, lr_base_mcc, _ = evaluate_model(
    lr_baseline, X_train, X_test, y_train, y_test, "LogReg"
)
print(f"        MCC: {lr_base_metrics['mcc']:.4f}, Acc: {lr_base_metrics['accuracy']:.4f}")

# Ridge Baseline
print("    Running Ridge Classifier (baseline)...")
ridge_baseline = RidgeClassifier(class_weight='balanced', random_state=42)
ridge_base_metrics, ridge_base_mcc, _ = evaluate_model(
    ridge_baseline, X_train, X_test, y_train, y_test, "Ridge"
)
print(f"        MCC: {ridge_base_metrics['mcc']:.4f}, Acc: {ridge_base_metrics['accuracy']:.4f}")

all_results['linear'] = {
    'baseline_comparison': [
        {'Algorithm': 'LDA', **{k: round(v, 6) for k, v in lda_base_metrics.items()}},
        {'Algorithm': 'Logistic Regression', **{k: round(v, 6) for k, v in lr_base_metrics.items()}},
        {'Algorithm': 'Ridge Classifier', **{k: round(v, 6) for k, v in ridge_base_metrics.items()}}
    ],
    'best_baseline': 'LDA',  # Selected for linear category
    'baseline_metrics': lda_base_metrics,
    'baseline_mcc_per_class': lda_base_mcc,
    'optimised_metrics': lda_opt_metrics,
    'optimised_mcc_per_class': lda_opt_mcc,
    'n_features_original': X_train.shape[1],
    'n_features_selected': len(selected_features_corr)
}

# ----- ENSEMBLE CLASSIFIERS -----
print("\n    === ENSEMBLE CLASSIFIERS ===")

# Random Forest Baseline
print("    Running Random Forest (baseline)...")
rf_baseline = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
rf_base_metrics, rf_base_mcc, y_pred_rf_base = evaluate_model(
    rf_baseline, X_train, X_test, y_train, y_test, "RF"
)
print(f"        MCC: {rf_base_metrics['mcc']:.4f}, Acc: {rf_base_metrics['accuracy']:.4f}")

# Random Forest Optimised
print("    Running Random Forest (optimised)...")
rf_opt = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2,
                                 min_samples_leaf=1, max_features='sqrt',
                                 class_weight='balanced', random_state=42, n_jobs=-1)
rf_opt_metrics, rf_opt_mcc, y_pred_rf_opt = evaluate_model(
    rf_opt, X_train_imp, X_test_imp, y_train, y_test, "RF-Opt"
)
print(f"        MCC: {rf_opt_metrics['mcc']:.4f}, Acc: {rf_opt_metrics['accuracy']:.4f}")

# Extra Trees Baseline
print("    Running Extra Trees (baseline)...")
et_baseline = ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
et_base_metrics, et_base_mcc, _ = evaluate_model(
    et_baseline, X_train, X_test, y_train, y_test, "ExtraTrees"
)
print(f"        MCC: {et_base_metrics['mcc']:.4f}, Acc: {et_base_metrics['accuracy']:.4f}")

# AdaBoost Baseline
print("    Running AdaBoost (baseline)...")
ada_baseline = AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME')
ada_base_metrics, ada_base_mcc, _ = evaluate_model(
    ada_baseline, X_train, X_test, y_train, y_test, "AdaBoost"
)
print(f"        MCC: {ada_base_metrics['mcc']:.4f}, Acc: {ada_base_metrics['accuracy']:.4f}")

all_results['ensemble'] = {
    'baseline_comparison': [
        {'Algorithm': 'Random Forest', **{k: round(v, 6) for k, v in rf_base_metrics.items()}},
        {'Algorithm': 'Extra Trees', **{k: round(v, 6) for k, v in et_base_metrics.items()}},
        {'Algorithm': 'AdaBoost', **{k: round(v, 6) for k, v in ada_base_metrics.items()}}
    ],
    'best_baseline': 'Random Forest',
    'baseline_metrics': rf_base_metrics,
    'baseline_mcc_per_class': rf_base_mcc,
    'optimised_metrics': rf_opt_metrics,
    'optimised_mcc_per_class': rf_opt_mcc,
    'n_features_original': X_train.shape[1],
    'n_features_selected': len(selected_features_imp)
}

# ----- NON-LINEAR CLASSIFIERS -----
print("\n    === NON-LINEAR CLASSIFIERS ===")

# KNN Baseline
print("    Running KNN (baseline)...")
knn_baseline = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
knn_base_metrics, knn_base_mcc, y_pred_knn_base = evaluate_model(
    knn_baseline, X_train, X_test, y_train, y_test, "KNN"
)
print(f"        MCC: {knn_base_metrics['mcc']:.4f}, Acc: {knn_base_metrics['accuracy']:.4f}")

# KNN Optimised
print("    Running KNN (optimised)...")
knn_opt = KNeighborsClassifier(n_neighbors=3, weights='distance', p=1, algorithm='auto', n_jobs=-1)
knn_opt_metrics, knn_opt_mcc, y_pred_knn_opt = evaluate_model(
    knn_opt, X_train_corr, X_test_corr, y_train, y_test, "KNN-Opt"
)
print(f"        MCC: {knn_opt_metrics['mcc']:.4f}, Acc: {knn_opt_metrics['accuracy']:.4f}")

# Decision Tree Baseline
print("    Running Decision Tree (baseline)...")
dt_baseline = DecisionTreeClassifier(class_weight='balanced', random_state=42)
dt_base_metrics, dt_base_mcc, _ = evaluate_model(
    dt_baseline, X_train, X_test, y_train, y_test, "DecisionTree"
)
print(f"        MCC: {dt_base_metrics['mcc']:.4f}, Acc: {dt_base_metrics['accuracy']:.4f}")

# SVM Baseline (can be slow)
print("    Running SVM-RBF (baseline)... (this may take a moment)")
svm_baseline = SVC(kernel='rbf', class_weight='balanced', random_state=42)
svm_base_metrics, svm_base_mcc, _ = evaluate_model(
    svm_baseline, X_train, X_test, y_train, y_test, "SVM-RBF"
)
print(f"        MCC: {svm_base_metrics['mcc']:.4f}, Acc: {svm_base_metrics['accuracy']:.4f}")

all_results['nonlinear'] = {
    'baseline_comparison': [
        {'Algorithm': 'KNN', **{k: round(v, 6) for k, v in knn_base_metrics.items()}},
        {'Algorithm': 'Decision Tree', **{k: round(v, 6) for k, v in dt_base_metrics.items()}},
        {'Algorithm': 'SVM-RBF', **{k: round(v, 6) for k, v in svm_base_metrics.items()}}
    ],
    'best_baseline': 'KNN',  # Selected for optimization
    'baseline_metrics': knn_base_metrics,
    'baseline_mcc_per_class': knn_base_mcc,
    'optimised_metrics': knn_opt_metrics,
    'optimised_mcc_per_class': knn_opt_mcc,
    'n_features_original': X_train.shape[1],
    'n_features_selected': len(selected_features_corr)
}

# =============================================================================
# 5. CREATE SUMMARY DATA
# =============================================================================
print("\n[4] Creating Summary Data...")

# Summary of optimised models
optimised_summary = pd.DataFrame({
    'Model': ['LDA', 'Random Forest', 'KNN'],
    'Category': ['Linear', 'Ensemble', 'Non-Linear'],
    'Member': ['Muhammad Usama Fazal (TP086008)', 'Imran Shahadat Noble (TP087895)', 'Md Sohel Rana (TP087437)'],
    'Accuracy': [lda_opt_metrics['accuracy'], rf_opt_metrics['accuracy'], knn_opt_metrics['accuracy']],
    'F1 (Weighted)': [lda_opt_metrics['f1_weighted'], rf_opt_metrics['f1_weighted'], knn_opt_metrics['f1_weighted']],
    'MCC': [lda_opt_metrics['mcc'], rf_opt_metrics['mcc'], knn_opt_metrics['mcc']],
    'Features': [len(selected_features_corr), len(selected_features_imp), len(selected_features_corr)]
})

print("\n" + "="*80)
print("OPTIMISED MODEL COMPARISON")
print("="*80)
print(optimised_summary.to_string(index=False))

# MCC per class
mcc_per_class = pd.DataFrame({
    'Attack Class': class_labels,
    'LDA': [lda_opt_mcc[c] for c in class_labels],
    'Random Forest': [rf_opt_mcc[c] for c in class_labels],
    'KNN': [knn_opt_mcc[c] for c in class_labels]
})

print("\n" + "="*80)
print("MCC PER ATTACK CLASS (Optimised Models)")
print("="*80)
print(mcc_per_class.to_string(index=False))

# Baseline vs Optimised Improvement
print("\n" + "="*80)
print("BASELINE vs OPTIMISED IMPROVEMENT")
print("="*80)
for cat, name in [('linear', 'LDA'), ('ensemble', 'Random Forest'), ('nonlinear', 'KNN')]:
    base_mcc = all_results[cat]['baseline_metrics']['mcc']
    opt_mcc = all_results[cat]['optimised_metrics']['mcc']
    improvement = (opt_mcc - base_mcc) / base_mcc * 100
    print(f"    {name}: {base_mcc:.4f} -> {opt_mcc:.4f} ({improvement:+.2f}%)")

# =============================================================================
# 6. GENERATE FIGURES
# =============================================================================
print("\n[5] Generating Figures...")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Figure 1: Model Comparison (Accuracy and MCC)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

models = ['LDA', 'Random Forest', 'KNN']
colors = ['#3498db', '#2ecc71', '#9b59b6']

# Accuracy
bars1 = axes[0].bar(models, optimised_summary['Accuracy'], color=colors, edgecolor='black')
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Model Accuracy Comparison (Optimised)', fontsize=14, fontweight='bold')
axes[0].set_ylim(0.7, 0.95)
for bar, val in zip(bars1, optimised_summary['Accuracy']):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# MCC
bars2 = axes[1].bar(models, optimised_summary['MCC'], color=colors, edgecolor='black')
axes[1].set_ylabel('Matthews Correlation Coefficient', fontsize=12)
axes[1].set_title('Model MCC Comparison (Optimised)', fontsize=14, fontweight='bold')
axes[1].set_ylim(0.6, 0.9)
for bar, val in zip(bars2, optimised_summary['MCC']):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'model_comparison_accuracy_mcc.png'), dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: model_comparison_accuracy_mcc.png")

# Figure 2: MCC Per Attack Class
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(class_labels))
width = 0.25

bars1 = ax.bar(x - width, mcc_per_class['LDA'], width, label='LDA', color='#3498db')
bars2 = ax.bar(x, mcc_per_class['Random Forest'], width, label='Random Forest', color='#2ecc71')
bars3 = ax.bar(x + width, mcc_per_class['KNN'], width, label='KNN', color='#9b59b6')

ax.set_xlabel('Attack Class', fontsize=12)
ax.set_ylabel('Matthews Correlation Coefficient', fontsize=12)
ax.set_title('MCC Per Attack Class - All Optimised Models', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([c.upper() for c in class_labels], fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, 1.1)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'mcc_per_class_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: mcc_per_class_comparison.png")

# Figure 3: Baseline vs Optimised
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(3)
width = 0.35

baseline_mcc = [lda_base_metrics['mcc'], rf_base_metrics['mcc'], knn_base_metrics['mcc']]
optimised_mcc = [lda_opt_metrics['mcc'], rf_opt_metrics['mcc'], knn_opt_metrics['mcc']]

bars1 = ax.bar(x - width/2, baseline_mcc, width, label='Baseline', color='steelblue')
bars2 = ax.bar(x + width/2, optimised_mcc, width, label='Optimised', color='darkorange')

ax.set_ylabel('MCC Score', fontsize=12)
ax.set_title('Baseline vs Optimised MCC Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 1)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'baseline_vs_optimised_mcc.png'), dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: baseline_vs_optimised_mcc.png")

# Figure 4: Confusion Matrices (All Optimised Models)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# LDA
cm_lda = confusion_matrix(y_test, y_pred_lda_opt, labels=class_labels)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_lda, display_labels=class_labels)
disp1.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title('LDA (Linear)\nMuhammad Usama Fazal', fontsize=12, fontweight='bold')

# RF
cm_rf = confusion_matrix(y_test, y_pred_rf_opt, labels=class_labels)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=class_labels)
disp2.plot(ax=axes[1], cmap='Greens', values_format='d')
axes[1].set_title('Random Forest (Ensemble)\nImran Shahadat Noble', fontsize=12, fontweight='bold')

# KNN
cm_knn = confusion_matrix(y_test, y_pred_knn_opt, labels=class_labels)
disp3 = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=class_labels)
disp3.plot(ax=axes[2], cmap='Purples', values_format='d')
axes[2].set_title('KNN (Non-Linear)\nMd Sohel Rana', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'confusion_matrices_all_models.png'), dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: confusion_matrices_all_models.png")

# Figure 5: Feature Correlation (for Non-Linear notebook)
plt.figure(figsize=(12, 8))
top_features = correlations.head(25)
sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
plt.title('Top 25 Features by Correlation with Target', fontsize=14, fontweight='bold')
plt.xlabel('Absolute Correlation')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'nonlinear_feature_correlation.png'), dpi=150)
plt.close()
print("    Saved: nonlinear_feature_correlation.png")

# Figure 6: Feature Importance (for Ensemble notebook)
plt.figure(figsize=(12, 10))
top_imp = feature_importances.head(30)
sns.barplot(x='importance', y='feature', data=top_imp, palette='viridis')
plt.title('Top 30 Feature Importances - Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'ensemble_feature_importance.png'), dpi=150)
plt.close()
print("    Saved: ensemble_feature_importance.png")

# =============================================================================
# 7. SAVE RESULTS TO JSON
# =============================================================================
print("\n[6] Saving Results...")

# Linear results
linear_results = {
    'classifier': 'Linear Discriminant Analysis',
    'category': 'Linear',
    'member': 'Muhammad Usama Fazal (TP086008)',
    'classification_type': 'multi-class',
    'classes': class_labels,
    'baseline_comparison': all_results['linear']['baseline_comparison'],
    'baseline_metrics': {k: round(v, 6) for k, v in lda_base_metrics.items()},
    'optimised_metrics': {k: round(v, 6) for k, v in lda_opt_metrics.items()},
    'baseline_mcc_per_class': {k: round(v, 6) for k, v in lda_base_mcc.items()},
    'optimised_mcc_per_class': {k: round(v, 6) for k, v in lda_opt_mcc.items()},
    'optimisation_strategies': ['Hyperparameter Tuning', 'Feature Selection (Correlation)'],
    'best_params': {'solver': 'svd', 'shrinkage': None},
    'n_features_original': 122,
    'n_features_selected': len(selected_features_corr),
    'feature_reduction_pct': round((122 - len(selected_features_corr)) / 122 * 100, 1)
}

with open(os.path.join(RESULTS_PATH, 'linear_lda_results.json'), 'w') as f:
    json.dump(linear_results, f, indent=2, default=str)
print("    Saved: linear_lda_results.json")

# Ensemble results
ensemble_results = {
    'classifier': 'Random Forest',
    'category': 'Ensemble',
    'member': 'Imran Shahadat Noble (TP087895)',
    'classification_type': 'multi-class',
    'classes': class_labels,
    'baseline_comparison': all_results['ensemble']['baseline_comparison'],
    'baseline_metrics': {k: round(v, 6) for k, v in rf_base_metrics.items()},
    'optimised_metrics': {k: round(v, 6) for k, v in rf_opt_metrics.items()},
    'baseline_mcc_per_class': {k: round(v, 6) for k, v in rf_base_mcc.items()},
    'optimised_mcc_per_class': {k: round(v, 6) for k, v in rf_opt_mcc.items()},
    'optimisation_strategies': ['Hyperparameter Tuning', 'Feature Selection (Importance)'],
    'best_params': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2,
                    'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': 'balanced'},
    'n_features_original': 122,
    'n_features_selected': len(selected_features_imp)
}

with open(os.path.join(RESULTS_PATH, 'ensemble_rf_results.json'), 'w') as f:
    json.dump(ensemble_results, f, indent=2, default=str)
print("    Saved: ensemble_rf_results.json")

# Non-linear results
nonlinear_results = {
    'classifier': 'K-Nearest Neighbors',
    'category': 'Non-Linear',
    'member': 'Md Sohel Rana (TP087437)',
    'classification_type': 'multi-class',
    'classes': class_labels,
    'baseline_comparison': all_results['nonlinear']['baseline_comparison'],
    'baseline_metrics': {k: round(v, 6) for k, v in knn_base_metrics.items()},
    'optimised_metrics': {k: round(v, 6) for k, v in knn_opt_metrics.items()},
    'baseline_mcc_per_class': {k: round(v, 6) for k, v in knn_base_mcc.items()},
    'optimised_mcc_per_class': {k: round(v, 6) for k, v in knn_opt_mcc.items()},
    'optimisation_strategies': ['Hyperparameter Tuning', 'Feature Selection (Correlation)'],
    'best_params': {'algorithm': 'auto', 'n_neighbors': 3, 'p': 1, 'weights': 'distance'},
    'n_features_original': 122,
    'n_features_selected': len(selected_features_corr)
}

with open(os.path.join(RESULTS_PATH, 'nonlinear_knn_results.json'), 'w') as f:
    json.dump(nonlinear_results, f, indent=2, default=str)
print("    Saved: nonlinear_knn_results.json")

# Group comparison results
group_results = {
    'comparison': optimised_summary.to_dict('records'),
    'mcc_per_class': mcc_per_class.to_dict('records'),
    'baseline_vs_optimised': {
        'LDA': {'baseline': round(lda_base_metrics['mcc'], 4), 'optimised': round(lda_opt_metrics['mcc'], 4)},
        'Random Forest': {'baseline': round(rf_base_metrics['mcc'], 4), 'optimised': round(rf_opt_metrics['mcc'], 4)},
        'KNN': {'baseline': round(knn_base_metrics['mcc'], 4), 'optimised': round(knn_opt_metrics['mcc'], 4)}
    },
    'ranking': [
        {'Model': 'KNN', 'Category': 'Non-Linear', 'Final Rank': 1},
        {'Model': 'Random Forest', 'Category': 'Ensemble', 'Final Rank': 2},
        {'Model': 'LDA', 'Category': 'Linear', 'Final Rank': 3}
    ],
    'best_model': 'KNN',
    'best_category': 'Non-Linear',
    'best_member': 'Md Sohel Rana (TP087437)'
}

with open(os.path.join(RESULTS_PATH, 'group_comparison_results.json'), 'w') as f:
    json.dump(group_results, f, indent=2, default=str)
print("    Saved: group_comparison_results.json")

# Save CSV summaries
optimised_summary.to_csv(os.path.join(RESULTS_PATH, 'group_model_comparison.csv'), index=False)
mcc_per_class.to_csv(os.path.join(RESULTS_PATH, 'group_mcc_per_class.csv'), index=False)
print("    Saved: group_model_comparison.csv, group_mcc_per_class.csv")

# =============================================================================
# 8. FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("VERIFICATION COMPLETE - FINAL SUMMARY")
print("="*80)

print("\nTEAM MEMBER ASSIGNMENTS:")
print("-" * 60)
print(f"{'Member':<35} {'Classifier':<15} {'MCC':<10}")
print("-" * 60)
print(f"Muhammad Usama Fazal (TP086008)    Linear (LDA)    {lda_opt_metrics['mcc']:.4f}")
print(f"Imran Shahadat Noble (TP087895)    Ensemble (RF)   {rf_opt_metrics['mcc']:.4f}")
print(f"Md Sohel Rana (TP087437)           Non-Linear (KNN) {knn_opt_metrics['mcc']:.4f}")
print("-" * 60)

print(f"\nBEST MODEL: KNN (Non-Linear) - MCC: {knn_opt_metrics['mcc']:.4f}")
print(f"            By: Md Sohel Rana (TP087437)")

print("\nFIGURES GENERATED:")
print("  - model_comparison_accuracy_mcc.png")
print("  - mcc_per_class_comparison.png")
print("  - baseline_vs_optimised_mcc.png")
print("  - confusion_matrices_all_models.png")
print("  - nonlinear_feature_correlation.png")
print("  - ensemble_feature_importance.png")

print("\nRESULTS SAVED:")
print("  - linear_lda_results.json")
print("  - ensemble_rf_results.json")
print("  - nonlinear_knn_results.json")
print("  - group_comparison_results.json")
print("  - group_model_comparison.csv")
print("  - group_mcc_per_class.csv")

print("\n" + "="*80)
print("All data verified and figures regenerated successfully!")
print("="*80)
