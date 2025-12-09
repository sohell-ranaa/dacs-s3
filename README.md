# DACS - Data Analytics in Cyber Security

Network Intrusion Detection System using Machine Learning Classifiers

## Project Overview

This project implements and compares three different machine learning classifiers for network intrusion detection using the NSL-KDD dataset. Each classifier represents a different category of ML algorithms:

| Classifier | Category | Algorithm |
|------------|----------|-----------|
| LDA | Linear | Linear Discriminant Analysis |
| Random Forest | Ensemble | Random Forest Classifier |
| KNN | Non-Linear | K-Nearest Neighbors |

## Project Structure

```
DACS/
├── Assignment/              # Main assignment deliverables
│   ├── REPORT.md           # Final report (Markdown)
│   ├── notebooks/          # Jupyter notebooks for each classifier
│   │   ├── 01_LDA_Linear_Classifier.ipynb
│   │   ├── 02_RandomForest_Ensemble.ipynb
│   │   ├── 03_KNN_NonLinear.ipynb
│   │   └── 04_Group_Comparative_Analysis.ipynb
│   ├── results/            # JSON results from each classifier
│   ├── report_figures/     # Generated figures for the report
│   └── datasets -> ../datasets/NSL_KDD
│
├── Demo/                    # Standalone demo notebooks
│   ├── 01_LDA_Linear_Classifier.ipynb
│   ├── 02_RandomForest_Ensemble.ipynb
│   ├── 03_KNN_NonLinear.ipynb
│   ├── 04_Group_Comparative_Analysis.ipynb
│   ├── helpers.py          # Utility functions
│   ├── data/               # Local dataset copies
│   ├── results/            # Demo results
│   └── figures/            # Demo figures
│
├── MLbasics/                # Machine Learning tutorials
│   ├── EDA.ipynb           # Exploratory Data Analysis
│   ├── EDA2.ipynb          # EDA Part 2
│   ├── FeatSelCorrP1.ipynb # Feature Selection (Correlation)
│   ├── FeatSelCVP3.ipynb   # Feature Selection (Cross-Validation)
│   ├── hypopt-KNNP2.ipynb  # Hyperparameter Optimization
│   ├── imBalanceP4.ipynb   # Handling Imbalanced Data
│   ├── EvalViz2.ipynb      # Evaluation Visualization
│   └── plotROCAUC.ipynb    # ROC-AUC Plotting
│
├── datasets/                # Shared datasets
│   ├── NSL_KDD/            # NSL-KDD intrusion detection dataset
│   │   ├── NSL_ppTrain.csv
│   │   ├── NSL_ppTest.csv
│   │   └── NSL_boosted-2.csv
│   ├── churn_modelling.csv # Customer churn dataset
│   └── tomatjus.csv        # Tomato juice quality dataset
│
├── mylib/                   # Shared utility library
│   ├── __init__.py
│   ├── mylib.py            # Core functions
│   ├── qbinr.py            # Quantile binning
│   └── selcb.py            # Feature selection
│
└── CourseResources/         # Course materials
    ├── Lecture Slides-20251209/
    ├── Tutorial Exercises-20251209/
    └── Coursework Question Paper(s).../
```

## Dataset

**NSL-KDD Dataset** - An improved version of the KDD Cup 1999 dataset for network intrusion detection.

- **Training Set:** ~125,000 records (boosted)
- **Test Set:** ~22,000 records
- **Features:** 41 features + 1 label
- **Classes:** Normal traffic vs Attack (binary classification)

Attack categories include: DoS, Probe, R2L, U2R

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
seaborn
mlxtend
yellowbrick
jupyter
```

Install with:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn mlxtend yellowbrick jupyter
```

## Running the Notebooks

### Start Jupyter Server
```bash
cd /home/rana-workspace/DACS
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Access at: `http://<server-ip>:8888`

### For Demo/Presentation
Use notebooks in the `Demo/` folder - they are standalone and include all dependencies.

### For Full Analysis
Use notebooks in the `Assignment/notebooks/` folder - they reference the shared `mylib` library.

## Classifier Results Summary

| Metric | LDA | Random Forest | KNN |
|--------|-----|---------------|-----|
| Accuracy | ~92% | ~99% | ~99% |
| Precision | ~91% | ~99% | ~99% |
| Recall | ~94% | ~99% | ~99% |
| F1-Score | ~92% | ~99% | ~99% |
| MCC | ~0.84 | ~0.98 | ~0.98 |

## Optimization Strategies

Each classifier was optimized using two strategies:

1. **Hyperparameter Tuning** - GridSearchCV with cross-validation
2. **Feature Selection** - Correlation-based feature selection

## Utility Functions (mylib)

```python
from mylib import show_labels_dist, show_metrics, bias_var_metrics

# Show label distribution in train/test sets
show_labels_dist(X_train, X_test, y_train, y_test)

# Display classification metrics
show_metrics(y_true, y_pred, classes)

# Bias-variance decomposition
bias_var_metrics(X_train, X_test, y_train, y_test, model, folds=10)
```

## Authors

DACS Group Assignment - 3 Members

## License

Educational use only - University coursework project.
