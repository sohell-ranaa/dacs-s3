<div align="center">

# CT115-3-M DATA ANALYTICS IN CYBER SECURITY

## GROUP ASSIGNMENT

---

### Machine Learning-Based Network Intrusion Detection System

---

**Group Members:**

| Name | Student ID |
|:-----|:-----------|
| Muhammad Usama Fazal | TP086008 |
| Imran Shahadat Noble | TP087895 |
| Md Sohel Rana | TP086217 |

---

**Intake Code:** UC2F2408CS

**Module Title:** Data Analytics in Cyber Security

**Submission Date:** December 2024

---

*Asia Pacific University of Technology and Innovation*

</div>

<div style="page-break-after: always;"></div>

---

## TABLE OF CONTENTS

| Section | Title | Page |
|:--------|:------|-----:|
| 1 | Combined Review of Selected Algorithms | 4 |
| 1.1 | Introduction | 4 |
| 1.2 | Algorithm Classification Taxonomy | 5 |
| 1.3 | Linear Classifier: LDA (Muhammad Usama Fazal - TP086008) | 6 |
| 1.4 | Non-Linear Classifier: KNN (Md Sohel Rana - TP086217) | 7 |
| 1.5 | Ensemble Classifier: Random Forest (Imran Shahadat Noble - TP087895) | 8 |
| 1.6 | Summary of Algorithm Characteristics | 9 |
| 2 | Integrated Performance Discussion | 10 |
| 2.1 | Experimental Setup | 10 |
| 2.2 | Performance Metrics | 11 |
| 2.3 | Comparative Analysis of Optimised Models | 12 |
| 2.4 | Cross-Validation Results | 15 |
| 2.5 | Key Findings and Recommendations | 16 |
| 3 | Individual Reports | 18 |
| 3.1 | Linear Discriminant Analysis (Muhammad Usama Fazal - TP086008) | 18 |
| 3.2 | Random Forest (Imran Shahadat Noble - TP087895) | 22 |
| 3.3 | K-Nearest Neighbors (Md Sohel Rana - TP086217) | 26 |
| 4 | References | 30 |
| | **Appendices** | |
| A | JupyterLab Notebook - Linear Classifier | 31 |
| B | JupyterLab Notebook - Ensemble Classifier | 45 |
| C | JupyterLab Notebook - Non-Linear Classifier | 60 |

<div style="page-break-after: always;"></div>

---

## LIST OF FIGURES

| Figure | Title | Page |
|:-------|:------|-----:|
| Figure 1 | Algorithm Classification Taxonomy | 5 |
| Figure 2 | Overall Model Performance Comparison | 12 |
| Figure 3 | Radar Chart: Multi-Metric Performance Comparison | 13 |
| Figure 4 | Performance Heatmap Across Metrics | 13 |
| Figure 5 | MCC Per Attack Class Comparison | 14 |
| Figure 6 | Cross-Validation Box Plot | 15 |
| Figure 7 | Training Time Comparison | 16 |
| Figure 8 | Baseline vs Optimised MCC Comparison | 17 |
| Figure 9 | Confusion Matrices for All Optimised Models | 17 |
| Figure 10 | Final Classifier Ranking | 18 |
| Figure 11 | Linear Classifier Baseline Comparison | 19 |
| Figure 12 | LDA Feature Correlation Analysis | 20 |
| Figure 13 | LDA Confusion Matrices (Baseline vs Optimised) | 21 |
| Figure 14 | Ensemble Classifier Baseline Comparison | 23 |
| Figure 15 | Random Forest Feature Importance | 24 |
| Figure 16 | Random Forest Confusion Matrices | 25 |
| Figure 17 | Non-Linear Classifier Baseline Comparison | 27 |
| Figure 18 | KNN Feature Correlation Analysis | 28 |
| Figure 19 | KNN Confusion Matrices (Baseline vs Optimised) | 29 |

<div style="page-break-after: always;"></div>

---

## LIST OF TABLES

| Table | Title | Page |
|:------|:------|-----:|
| Table 1 | Class Distribution in NSL-KDD Dataset | 4 |
| Table 2 | Attack Category Descriptions | 5 |
| Table 3 | Comparison of Selected Classification Algorithms | 9 |
| Table 4 | Dataset Composition | 10 |
| Table 5 | Data Preprocessing Summary | 11 |
| Table 6 | Selected Performance Metrics | 11 |
| Table 7 | Optimised Model Performance Metrics | 12 |
| Table 8 | MCC Performance by Attack Category | 14 |
| Table 9 | Cross-Validation Results (F1-Weighted) | 15 |
| Table 10 | Feature Reduction Summary | 16 |
| Table 11 | Final Classifier Ranking | 17 |
| Table 12 | Linear Classifier Baseline Comparison | 19 |
| Table 13 | LDA Hyperparameter Tuning Configuration | 20 |
| Table 14 | LDA Top Correlated Features | 20 |
| Table 15 | LDA Baseline vs Optimised Performance | 21 |
| Table 16 | LDA MCC Per Class Comparison | 21 |
| Table 17 | Ensemble Classifier Baseline Comparison | 23 |
| Table 18 | Random Forest Hyperparameter Configuration | 24 |
| Table 19 | Random Forest Top Important Features | 24 |
| Table 20 | Random Forest Baseline vs Optimised Performance | 25 |
| Table 21 | Non-Linear Classifier Baseline Comparison | 27 |
| Table 22 | KNN Hyperparameter Tuning Configuration | 28 |
| Table 23 | KNN Baseline vs Optimised Performance | 29 |
| Table 24 | KNN MCC Per Class Comparison | 29 |

<div style="page-break-after: always;"></div>
