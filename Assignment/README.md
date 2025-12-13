# DACS Assignment: ML-Based Network Intrusion Detection

**Course:** CT115-3-M Data Analytics in Cyber Security
**Institution:** Asia Pacific University of Technology and Innovation
**Intake:** UC2F2408CS

---

## Team Members

| No. | Name | Student ID | Classifier | Role |
|-----|------|------------|------------|------|
| 1 | Muhammad Usama Fazal | TP086008 | Linear (LDA) | Individual Chapter 3 |
| 2 | Imran Shahadat Noble | TP087895 | Ensemble (Random Forest) | Individual Chapter 4 |
| 3 | Md Sohel Rana | TP087437 | Non-Linear (KNN) | Individual Chapter 5 |

---

## Project Overview

This project implements machine learning-based network intrusion detection using the NSL-KDD dataset. Three classifier categories are compared:

- **Linear:** Linear Discriminant Analysis (LDA)
- **Ensemble:** Random Forest
- **Non-Linear:** K-Nearest Neighbors (KNN)

### Final Results (Verified)

| Model | MCC (Baseline) | MCC (Optimised) | Improvement | Accuracy |
|-------|----------------|-----------------|-------------|----------|
| **KNN** | 0.7602 | **0.8161** | +7.35% | 87.52% |
| Random Forest | 0.8135 | 0.8146 | +0.14% | 87.09% |
| LDA | 0.6644 | 0.6712 | +1.02% | 77.48% |

**Best Model:** KNN (Non-Linear) - Md Sohel Rana (TP087437)

---

## Folder Structure

```
Assignment/
├── README.md                    # This file
├── view_report.html             # Browser-based MD viewer
├── data/                        # Datasets
│   ├── NSL_ppTrain.csv          # Training data (preprocessed)
│   └── NSL_ppTest.csv           # Test data (preprocessed)
├── notebooks/                   # Jupyter Notebooks
│   ├── 00_Assignment_Walkthrough.ipynb
│   ├── 01_Linear_Classifier.ipynb      # Muhammad Usama Fazal
│   ├── 02_Ensemble_Classifier.ipynb    # Imran Shahadat Noble
│   ├── 03_NonLinear_Classifier.ipynb   # Md Sohel Rana
│   ├── 04_Group_Comparison.ipynb       # Group comparison
│   ├── All_Classifiers_vs_skML_Comparison.ipynb
│   └── skML-complete.ipynb             # Professor's template
├── results/                     # Model metrics
│   ├── group_comparison_results.json   # Final verified results
│   ├── linear_lda_results.json
│   ├── ensemble_rf_results.json
│   └── nonlinear_knn_results.json
├── figures/                     # Generated visualizations
│   ├── model_comparison_accuracy_mcc.png
│   ├── mcc_per_class_comparison.png
│   ├── baseline_vs_optimised_mcc.png
│   └── ... (26 PNG files)
├── reporting/                   # Report documents
│   ├── final/
│   │   └── REPORT.md            # Complete report (1263 lines)
│   └── drafts/
├── specs/                       # Assignment specifications
│   ├── GROUP ASSIGNMENT - QUESTION PAPER.pdf
│   └── About_KDD-NSL.pdf
├── Recording/                   # Lecture transcripts
│   └── transcripts/
└── mylib/                       # Helper functions
    └── mlutils.py
```

---

## Quick Start Guide

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter imbalanced-learn
```

### 2. Run Jupyter Notebooks

```bash
cd /home/rana-workspace/DACS/Assignment/notebooks
jupyter notebook
```

### 3. Execute Notebooks (In Order)

| Order | Notebook | Purpose |
|-------|----------|---------|
| 1 | `00_Assignment_Walkthrough.ipynb` | Overview and setup |
| 2 | `01_Linear_Classifier.ipynb` | LDA baseline + optimization |
| 3 | `02_Ensemble_Classifier.ipynb` | Random Forest baseline + optimization |
| 4 | `03_NonLinear_Classifier.ipynb` | KNN baseline + optimization |
| 5 | `04_Group_Comparison.ipynb` | Compare all optimized models |

---

## How to View the Report

### Option 1: View in VS Code (Recommended)
1. Open `reporting/final/REPORT.md` in VS Code
2. Press `Ctrl+Shift+V` to preview markdown
3. Or click the preview icon in the top-right corner

### Option 2: View in Browser (Auto-Render)

**Method A: Use the HTML Viewer**
```bash
# Open the HTML viewer in your browser
xdg-open /home/rana-workspace/DACS/Assignment/view_report.html
# Or on Mac: open view_report.html
# Or on Windows: start view_report.html
```

**Method B: Start a Local Server**
```bash
cd /home/rana-workspace/DACS/Assignment
python3 -m http.server 8080
# Then open http://localhost:8080/view_report.html in browser
```

**Method C: Use grip (GitHub-style rendering)**
```bash
pip install grip
grip reporting/final/REPORT.md
# Opens at http://localhost:6419
```

### Option 3: Convert to PDF
```bash
# Using pandoc
pandoc reporting/final/REPORT.md -o REPORT.pdf

# Using VS Code extension "Markdown PDF"
# Right-click on REPORT.md → "Markdown PDF: Export (pdf)"
```

---

## Assignment Deliverables Checklist

### Report (PDF, max 30 pages)
- [x] Front cover with member names and intake
- [x] Chapter 1: Combined Review of Algorithms
- [x] Chapter 2: Integrated Performance Discussion
- [x] Chapter 3: LDA Individual Chapter (Muhammad Usama Fazal)
- [x] Chapter 4: Random Forest Individual Chapter (Imran Shahadat Noble)
- [x] Chapter 5: KNN Individual Chapter (Md Sohel Rana)
- [x] References (APA format)
- [x] No JupyterLab graphs in report
- [x] No code blocks in report

### Jupyter Notebooks (Appendix)
- [x] Pre/post optimization performance statistics
- [x] Optimization strategy implementation
- [x] Visualizations (confusion matrices, feature importance, etc.)

---

## Key Files

| File | Description |
|------|-------------|
| `reporting/final/REPORT.md` | Complete report (ready for PDF conversion) |
| `results/group_comparison_results.json` | Verified final metrics |
| `notebooks/04_Group_Comparison.ipynb` | Group comparison analysis |
| `specs/GROUP ASSIGNMENT - QUESTION PAPER.pdf` | Assignment requirements |

---

## Verified Metrics (Dec 13, 2024)

### MCC Per Attack Class (Optimized Models)

| Attack Class | LDA | Random Forest | KNN |
|--------------|-----|---------------|-----|
| Benign | 0.673 | 0.763 | 0.786 |
| DoS | 0.786 | **0.984** | 0.946 |
| Probe | 0.575 | **0.926** | 0.850 |
| R2L | 0.513 | 0.330 | **0.567** |
| U2R | 0.579 | **0.820** | 0.572 |

---

## Contact

For questions about specific chapters, contact the respective team member.

---

*Last Updated: December 13, 2024*
