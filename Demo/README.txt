================================================================================
DACS ASSIGNMENT - NETWORK INTRUSION DETECTION DEMO
================================================================================

This folder contains everything needed to run the Jupyter notebooks for
demonstration to the professor.

FOLDER STRUCTURE:
-----------------
DACS_Demo/
├── 01_LDA_Linear_Classifier.ipynb      <- Member 1's notebook
├── 02_RandomForest_Ensemble.ipynb      <- Member 2's notebook
├── 03_KNN_NonLinear.ipynb              <- Member 3's notebook
├── 04_Group_Comparative_Analysis.ipynb <- Group comparison notebook
├── helpers.py                          <- Helper functions
├── data/                               <- Dataset files
│   ├── NSL_boosted-2.csv              <- Training data
│   └── NSL_ppTest.csv                 <- Test data
├── results/                            <- Results JSON files
│   ├── linear_lda_results.json
│   ├── ensemble_rf_results.json
│   └── nonlinear_knn_results.json
├── figures/                            <- Report figures (for reference)
└── README.txt                          <- This file

REQUIREMENTS:
-------------
- Python 3.8+
- Jupyter Notebook or JupyterLab
- Required packages:
  pip install numpy pandas matplotlib seaborn scikit-learn

HOW TO RUN:
-----------
1. Open terminal/command prompt in this folder
2. Run: jupyter notebook
3. Open notebooks in order (01, 02, 03, then 04)
4. Run all cells in each notebook

NOTEBOOKS ORDER:
----------------
1. 01_LDA_Linear_Classifier.ipynb    - Linear method (LDA)
2. 02_RandomForest_Ensemble.ipynb    - Ensemble method (Random Forest)
3. 03_KNN_NonLinear.ipynb            - Non-linear method (KNN)
4. 04_Group_Comparative_Analysis.ipynb - Compare all three classifiers

================================================================================
