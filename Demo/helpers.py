"""
Helper functions for DACS Assignment - Network Intrusion Detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, classification_report,
                             confusion_matrix)
from sklearn.model_selection import cross_val_predict


def show_labels_dist(X_train, X_test, y_train, y_test):
    """Display label distribution in train and test sets"""
    print("="*50)
    print("DATASET DISTRIBUTION")
    print("="*50)
    print(f"\nTraining Set: {len(X_train)} samples")
    print(y_train.value_counts())
    print(f"\nTest Set: {len(X_test)} samples")
    print(y_test.value_counts())
    print("="*50)


def show_metrics(y_true, y_pred, classes=None):
    """Display classification metrics"""
    print("\n" + "-"*50)
    print("CLASSIFICATION METRICS")
    print("-"*50)

    print(f"\nAccuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, pos_label='attack'):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, pos_label='attack'):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, pos_label='attack'):.4f}")
    print(f"MCC:       {matthews_corrcoef(y_true, y_pred):.4f}")

    print("\n" + "-"*50)
    print("CLASSIFICATION REPORT")
    print("-"*50)
    print(classification_report(y_true, y_pred))

    print("-"*50)
    print("CONFUSION MATRIX")
    print("-"*50)
    cm = confusion_matrix(y_true, y_pred)
    print(pd.DataFrame(cm,
                      index=['Actual: Normal', 'Actual: Attack'],
                      columns=['Pred: Normal', 'Pred: Attack']))


def bias_var_metrics(X_train, X_test, y_train, y_test, model, folds=10):
    """Calculate bias-variance metrics using cross-validation"""
    from sklearn.model_selection import cross_val_score

    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=folds, scoring='accuracy')

    # Train and test on full sets
    model_clone = model.__class__(**model.get_params())
    model_clone.fit(X_train, y_train)
    train_acc = model_clone.score(X_train, y_train)
    test_acc = model_clone.score(X_test, y_test)

    print(f"\nCross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Bias (1 - CV mean): {1 - cv_scores.mean():.4f}")
    print(f"Variance (CV std): {cv_scores.std():.4f}")
    print(f"Generalization Gap: {train_acc - test_acc:.4f}")
