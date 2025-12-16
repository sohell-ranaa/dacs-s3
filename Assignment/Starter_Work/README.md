# KNN for Network Intrusion Detection

**Author:** Md Sohel Rana
**Student ID:** TP087437
**Module:** CT115-3-M Data Analytics in Cyber Security
**Algorithm:** K-Nearest Neighbors (KNN)

---

## Project Summary

This is a group assignment for building a **Network Intrusion Detection System** using Machine Learning. We are using the **NSL-KDD dataset** to classify network traffic into 5 categories: benign, dos, probe, r2l, and u2r.

### Team Members & Assigned Algorithms

| Member | Student ID | Algorithm | Type |
|--------|------------|-----------|------|
| Md Sohel Rana | TP087437 | K-Nearest Neighbors (KNN) | Non-Linear, Instance-based |
| Muhammad Usama Fazal | TP086008 | Linear Discriminant Analysis (LDA) | Linear, Dimensionality Reduction |
| Imran Shahadat Noble | TP087895 | Random Forest | Ensemble, Tree-based |

### What Each Algorithm Does

**KNN (My Part):**
- Classifies based on the k closest neighbors in the feature space
- Uses distance metrics (Euclidean) to find similar samples
- Simple, interpretable, but can be slow on large datasets

**LDA (Usama):**
- Finds linear combinations of features that best separate classes
- Reduces dimensionality while maximizing class separability
- Works well when classes are linearly separable

**Random Forest (Imran):**
- Builds multiple decision trees and combines their predictions
- Handles complex patterns and feature interactions
- Robust to overfitting due to ensemble averaging

---

## My Work: KNN Implementation

### File Structure

```
Starter_Work/
├── KNN_Intrusion_Detection_Complete.ipynb  # Main notebook (all work in one file)
├── README.md                                # This file
├── data/                                    # Dataset files
│   ├── NSL_boosted-2.csv                   # Training data
│   └── NSL_ppTest.csv                      # Test data
└── mylib/                                   # Helper libraries (if any)
```

### Notebook Contents

The notebook is organized into 5 parts:

| Part | Description |
|------|-------------|
| **Part 1** | Data Exploration - Understanding dataset structure and class distribution |
| **Part 2** | Data Preprocessing - Encoding, scaling, and preparation |
| **Part 3** | Baseline Model - KNN with default k=5 as reference point |
| **Part 4** | KNN Tuning - Testing different k values to find optimal |
| **Part 5** | Summary & Conclusions - Results and future improvements |

### Approach & Decisions

| Decision | Reason |
|----------|--------|
| One-Hot Encoding | Categorical features (protocol_type, service, flag) have no ordinal relationship |
| MinMax Scaling | KNN uses distance-based calculations; features need same scale |
| Baseline first | Establishes reference point before tuning |
| Odd k values | Prevents voting ties in classification |

### References Used

1. Scikit-learn KNN Documentation
2. IBM - What is KNN?
3. GeeksforGeeks - KNN Algorithm
4. Towards Data Science - KNN Explained
5. NSL-KDD Dataset Documentation

Full references with links are included in the notebook.

---

## Next Steps

- [ ] Compare results with team members
- [ ] Try additional optimizations (feature selection, distance metrics)
- [ ] Prepare final comparison report

---

*CT115-3-M Data Analytics in Cyber Security - Group Assignment*
