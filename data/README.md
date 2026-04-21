## Data folder

# data/README.md

This folder holds the four datasets used to test the agent across different conditions. Each one was chosen because it naturally triggers a different set of decision branches in the pipeline. Together they cover most of the planner, reflector, and modelling logic without needing to engineer artificial scenarios.

---

## 1. Heart Disease

**File:** `heart.csv`  
**Source:** Kaggle - [johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)  
**Original source:** UCI Machine Learning Repository, Cleveland subset  
**Shape:** 1,025 rows x 14 columns  
**Target:** `target` (0 = no disease, 1 = disease)  
**Missing values:** A small number in `ca` and `thal`  
**Class balance:** ~54/46, roughly balanced  

**Why this dataset:**

This is the simplest of the four and works as a clean baseline. At just over 1,000 rows it sits right at the boundary of the small dataset guard, so the agent switches to simpler models. The class balance is close to even, which means no imbalance correction fires. This is useful because it confirms the agent only applies that logic when the data actually needs it, not by default. There is a mix of numeric and categorical features so both the scaling and encoding paths get tested.

**Command:**

```bash
python run_agent.py --data data/heart.csv --target target --output_root outputs/heart --max_replans 1
```

---

## 2. Adult Income

**File:** `adult.csv`  
**Source:** Kaggle - [wenruliu/adult-income-dataset](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)  
**Original source:** UCI Machine Learning Repository, Census Income Dataset  
**Shape:** 48,842 rows x 15 columns  
**Target:** `income` (<=50K / >50K)  
**Missing values:** `workclass`, `occupation`, and `native-country` use `?` as a placeholder  
**Class balance:** ~76/24, mild imbalance  

**Why this dataset:**

This is the large, messy real-world test. The missing values are written as `?` rather than being left blank, which is a common pattern in older UCI datasets. The agent catches these automatically through `_EXTRA_NA_VALUES` in `load_data()` so no manual cleaning is needed before running. Columns like `occupation` and `native-country` have a high number of unique values, which triggers the OrdinalEncoder path. The dataset is also large enough to bring in GradientBoosting and run the full model pool. Auto target detection is used here to show the agent can figure out the right column without being told explicitly.

**Command:**

```bash
python run_agent.py --data data/adult.csv --target auto --output_root outputs/adult --max_replans 1
```

---

## 3. Bank Marketing

**File:** `bank-full.csv`  
**Source:** UCI Machine Learning Repository - [Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)  
**Shape:** 45,211 rows x 17 columns  
**Target:** `y` (yes / no - did the client subscribe to a term deposit)  
**Missing values:** None  
**Class balance:** ~88/12, severe imbalance  

**Why this dataset:**

This dataset was chosen specifically to test the imbalance handling logic. The 88/12 split is well above the 3.0 ratio threshold, so the planner applies `class_weight='balanced'` to all compatible models and switches the primary metric to `balanced_accuracy`. The original file is semicolon-delimited rather than comma-delimited, which the delimiter sniffer in `load_data()` handles automatically without any pre-processing. High-cardinality columns like `job`, `month`, and `contact` also trigger the OrdinalEncoder path. If the reflector flags issues after the first pass, the replan escalates the imbalance handling from `class_weight` to SMOTE.

**Command:**

```bash
python run_agent.py --data data/bank-full.csv --target y --output_root outputs/bank --max_replans 1
```

---

## 4. Stroke Prediction

**File:** `healthcare-dataset-stroke-data.csv`  
**Source:** Kaggle - [fedesoriano/stroke-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)  
**Shape:** 5,110 rows x 12 columns  
**Target:** `stroke` (0 = no stroke, 1 = stroke)  
**Missing values:** `bmi` has around 200 missing values (~3.9%)  
**Class balance:** ~95/5, very severe imbalance  

**Why this dataset:**

This is the hardest test for the agent and the one most likely to trigger the full replan loop. The 95/5 class split is the most extreme across all four datasets. On the first pass the agent applies `class_weight='balanced'`, but the minority class is so small that the reflector will almost certainly flag performance issues and recommend a replan. On the second pass the strategy escalates to SMOTE and extra models are added to the pool. This gives the most direct evidence that the reflection and replanning logic works end to end. The dataset also has numeric columns with real outliers (`avg_glucose_level`, `bmi`) which trigger RobustScaler, alongside categorical columns that go through OneHotEncoder.

**Command:**

```bash
python run_agent.py --data data/healthcare-dataset-stroke-data.csv --target stroke --output_root outputs/stroke --max_replans 1
```

---

