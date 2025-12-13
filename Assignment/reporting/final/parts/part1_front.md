# FRONT COVER

---

**CT115-3-M DATA ANALYTICS IN CYBER SECURITY**

**GROUP ASSIGNMENT**

---

**Group Members:**

| Name | Student ID |
|------|------------|
| Muhammad Usama Fazal | TP086008 |
| Imran Shahadat Noble | TP087895 |
| Md Sohel Rana | TP086217 |

**Intake Code:** UC2F2408CS

**Module Title:** Data Analytics in Cyber Security

**Assignment Title:** Machine Learning-Based Network Intrusion Detection System

**Submission Date:** December 2024

---

<div style="page-break-after: always;"></div>

## TABLE OF CONTENTS

1. [Combined Review of Selected Algorithms](#1-combined-review-of-selected-algorithms)
   - 1.1 Introduction
   - 1.2 Algorithm Classification Overview
   - 1.3 Linear Classifier: Linear Discriminant Analysis (Muhammad Usama Fazal - TP086008)
   - 1.4 Non-Linear Classifier: K-Nearest Neighbors (Md Sohel Rana - TP086217)
   - 1.5 Ensemble Classifier: Random Forest (Imran Shahadat Noble - TP087895)
   - 1.6 Summary of Algorithm Characteristics

2. [Integrated Performance Discussion](#2-integrated-performance-discussion)
   - 2.1 Experimental Setup
   - 2.2 Performance Metrics
   - 2.3 Comparative Analysis
   - 2.4 Cross-Validation Results
   - 2.5 Key Findings and Recommendations

3. [Individual Reports](#3-individual-reports)
   - 3.1 Linear Discriminant Analysis (Muhammad Usama Fazal - TP086008)
   - 3.2 Random Forest (Imran Shahadat Noble - TP087895)
   - 3.3 K-Nearest Neighbors (Md Sohel Rana - TP086217)

4. [References](#4-references)

5. [Appendices](#5-appendices)
   - Appendix A: JupyterLab Notebook - Linear Classifier (Muhammad Usama Fazal)
   - Appendix B: JupyterLab Notebook - Ensemble Classifier (Imran Shahadat Noble)
   - Appendix C: JupyterLab Notebook - Non-Linear Classifier (Md Sohel Rana)

---

<div style="page-break-after: always;"></div>

## LIST OF TABLES

| Table No. | Title | Page |
|-----------|-------|------|
| Table 1 | Class Distribution in NSL-KDD Dataset | 4 |
| Table 2 | Comparison of Selected Classification Algorithms | 6 |
| Table 3 | Dataset Composition | 7 |
| Table 4 | Optimised Model Performance Metrics | 8 |
| Table 5 | MCC Performance by Attack Category | 8 |
| Table 6 | Cross-Validation Results (F1-Weighted) | 9 |
| Table 7 | Feature Reduction Summary | 9 |
| Table 8 | Final Classifier Ranking | 10 |
| Table 9 | Linear Classifier Baseline Comparison | 11 |
| Table 10 | LDA Hyperparameter Tuning Results | 12 |
| Table 11 | LDA Baseline vs Optimised Performance | 13 |
| Table 12 | Ensemble Classifier Baseline Comparison | 14 |
| Table 13 | Random Forest Hyperparameter Configuration | 15 |
| Table 14 | Random Forest Baseline vs Optimised Performance | 16 |
| Table 15 | Non-Linear Classifier Baseline Comparison | 17 |
| Table 16 | KNN Hyperparameter Tuning Results | 18 |
| Table 17 | KNN Baseline vs Optimised Performance | 19 |

---

<div style="page-break-after: always;"></div>

## LIST OF FIGURES

| Figure No. | Title | Page |
|------------|-------|------|
| Figure 1 | Algorithm Classification Overview | 5 |
| Figure 2 | Optimised Models MCC Comparison | 8 |
| Figure 3 | MCC Per Attack Class Comparison | 9 |
| Figure 4 | Cross-Validation Performance | 10 |
| Figure 5 | LDA Baseline Comparison | 12 |
| Figure 6 | LDA Baseline vs Optimised MCC | 13 |
| Figure 7 | Random Forest Baseline Comparison | 15 |
| Figure 8 | Random Forest Baseline vs Optimised MCC | 16 |
| Figure 9 | KNN Baseline Comparison | 18 |
| Figure 10 | KNN Baseline vs Optimised MCC | 19 |

---

<div style="page-break-after: always;"></div>
