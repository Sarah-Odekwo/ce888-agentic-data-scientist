
# CE888 Deliverable 1 - Data Exploration & Planning (Agentic EDA)

## Purpose
This submission provides an automated EDA notebook plus this README summarizing key EDA findings and the agentic plan
for the final Offline Agentic Data Scientist.

## Contents
- `eda.ipynb`: Automated EDA notebook
- `README.md`: Key EDA findings, dataset risks and agentic planning proposal (this file).

## How to run
1. Place the dataset CSV in the same folder as `eda.ipynb` and update the filename variable in the notebook
2. Run the notebook from top to bottom (it prints scans, generates plots, extracts signals, and prints out the final EDA 
report).
3. Requirements: Python with `pandas`, `numpy`, `matplotlib`

## What the notebook does
- Loads an arbitrary CSV file and performs basic inspection
- Performs a "data health scan" for missing values, including replacing hidden missing characters (like "?", "NA", "NULL")
with "NaN"
- Automatically detects a likely target variable using heuristics (common target names, last column, cardinality)
- Infers the problem type (whether classification/ regression/ unsupervised) from the target
- Applies basic cleaning on the dataset: numeric coercion, dropping likely ID columns using high cardinality rules
- Detects the feature types: continuous numeric/ binary numeric and categorical (binary vs multi-class)
- Produces key EDA outputs: missingness plot, taget distribution, imbalance ratio, continuous numeric distributions, 
skewness plots, and categorical against target stacked bar charts
- Extracts and prints structured EDA signals, recommends evaluation metrics based on extracted signals and prints out 
preprocessing recommendations
- Prints a final "EDA Report" summary block

## Key findings from this dataset run
- Dataset used: `WaFn-UseC-Telco-Customer-Churn.csv`
- After basic cleaning, the dataset size is 7043 rows x 20 columns (the likely column ID was dropped)
- Target and problem type: The detected target is `Churn` , and the problem type is classification
  - The notebook computes an imbalance ratio of **2.77**, indicating there is moderate imbalance that should affect the
  evaluation metric choice.
- Missingness: The missing value were found in `TotalCharges` (with 11 missing cells, which is about 0.16% of that feature)
  - The notebook recommends **imputation** for low/moderate missingness
- Feature types and distributions: Continuous numeric features and categorical features detected
  - Skewness values calculated are small, so no transformation is suggested
- Dataset risks:
  - Class imbalance: (imbalance ratio of 2.77), this can hide the poor minority class performance is accuracy is the
  only evaluation metric being used
  - Mixed feature types: (many categoricals + some numeric) requiring careful encoding
  


## Proposed Agentic Plan 
(signals -> decisions -> reflection)

### EDA Signals extracted from the dataset
The notebook outputs a structured signal dictionary that includes:
- Dataset shape, detected target, problem type, dataset size category
- Feature type counts
- Missingness by feature, imbalance ratio (for classification)
- Recommended evaluation metrics and preprocessing recommendations that were derived from these extracted signals

### Decision rules driven by the signals
- Task routing: classification/regression/unsupervised determines the evaluation path
- Metric choice: if imbalance ratio is high (the notebook uses a threshold of > 1.5), prefer F1/ ROC-AUC/ Precision-Recall
- Preprocessing: 
  - low/moderate missingness -> imputation
  - high missingness -> consider dropping the feature entirely
  - skewness transformations only for continuous numerical features with high skew
  - Feature handling: binary numeric features are handled as categoricals

### Reflection and Re-Planning triggers:
The agent will trigger reflection and re-planning when:
- The inferred problem type conflicts with the target structure (e.g. classification problem but the target looks 
continuous), or the target detection confidence is low
- Validation performance suggests instability, prompting a change of metrics, thresholding or imbalance handling strategy
- The dataset's missingness, cardinality or feature type mix differs a significant amount from prior runs, requiring 
different feature selection

