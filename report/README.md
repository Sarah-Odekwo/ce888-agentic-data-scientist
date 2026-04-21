# Agentic Data Scientist - CE888 Final Project Report

> **Module:** CE888 Data Science and Decision Making
> **Deliverable:** Final Project Code
> **Submitted by:** 2500779

---

## Table of Contents

1. Introduction
2. System Architecture
3. Dataset Understanding
4. Planning Logic
5. Tool Use: Modelling and Evaluation
6. Reflection and Re-planning
7. Memory and Learning
8. Ethics and Limitations
9. Conclusion and Future Work
10. How to Run
11. Repository Structure

---

## 1. Introduction

### Goal

This project builds an offline Agentic Data Scientist: an autonomous system that runs a full, adaptive machine learning pipeline on any unseen tabular CSV dataset without needing a human in the loop. It was built for CE888 assignment submission. The design prioritises reasoning about data rather than only optimising for higher accuracy scores. The agent is expected to diagnose a dataset, form a plan, execute it, and revise that plan if the results are not good enough.

### Why Agentic?

A conventional ML pipeline is fixed. It applies the same preprocessing steps and the same set of models regardless of what the data actually looks like. This system works differently. It treats every dataset as a problem to understand before attempting to solve it.

This distinction shows up at every stage:

- The **data profiler** reads the raw dataset and pulls out signals: how much data is missing, whether there are outliers, whether classes are imbalanced, whether categorical features are high-cardinality, and whether numeric features are strongly correlated with each other.
- The **planner** reads those signals and produces a justified, step-by-step execution plan. Each preprocessing and modelling decision is tied back to a concrete observation about the data.
- The **reflector** checks the results after training and flags issues by severity. If the issues are serious enough, it recommends a revised strategy and hands it back to the orchestrator.
- The **memory module** keeps a record of what worked on previous runs of the same dataset so the agent can make smarter decisions the second time around.

---

## 2. System Architecture

### Data Flow

```
CSV Dataset
     |
     v
+----------------------+
|  data_profiler.py    |  Reads the dataset. Flags missing values, outliers,
|  (tools/)            |  class imbalance, high-cardinality columns, and
+----------+-----------+  correlated feature pairs.
           |
           |  dataset_profile dict + _plan_config
           v
+----------------------+
|  planner.py          |  Reads each signal and produces a step-by-step plan
|  (agents/)           |  with a written justification for each decision.
+----------+-----------+
           |
           |  plan (List[str]) + updated _plan_config
           v
+----------------------+
|  modelling.py        |  Builds the preprocessing pipeline and model
|  (tools/)            |  candidate list from _plan_config. Trains on a
+----------+-----------+  stratified split.
           |
           |  training payload (best model, all metrics, pipelines)
           v
+----------------------+
|  evaluation.py       |  Computes the confusion matrix, classification
|  (tools/)            |  report, and a paired t-test across candidates.
+----------+-----------+
           |
           |  eval_payload
           v
+----------------------+
|  reflector.py        |  Diagnoses the results. Tags issues as HIGH,
|  (agents/)           |  MEDIUM, or LOW. Assigns a replan tier between(0-3).
+----------+-----------+
           |
           v
+----------------------+       +---------------------------+
|  memory.py           | <-->  |  agentic_data_scientist.py|
|  (agents/)           |       |  (orchestrator)           |
+----------------------+       +---------------------------+
           |
           v
    outputs/{run_id}/
    |- report.md
    |- metrics.json
    |- eda_summary.json
    |- plan.json
    |- reflection.json
    +- confusion_matrix.png
```

### Module Breakdown


| Module                      | What it does                                                           |
| --------------------------- | ---------------------------------------------------------------------- |
| `tools/data_profiler.py`    | Reads the raw dataset, extracts all signals, seeds `_plan_config`      |
| `agents/planner.py`         | Converts signals into a justified plan, mutates `_plan_config`         |
| `tools/modelling.py`        | Builds the preprocessor from `_plan_config`, selects and trains models |
| `tools/evaluation.py`       | Evaluates the best model, runs a paired t-test, writes the report      |
| `agents/reflector.py`       | Diagnoses performance, assigns severity and replan tier                |
| `agents/memory.py`          | Saves run history to JSON, retrieves the best-ever strategy            |
| `agentic_data_scientist.py` | Orchestrator: connects all components and manages the replan loop      |
| `run_agent.py`              | CLI entry point                                                        |


### The Agentic Loop

The orchestrator in `agentic_data_scientist.py` runs a while-loop with a replan gate:

```
profile -> plan -> preprocess -> train -> evaluate -> reflect
               ^                                         |
               |         replan_recommended = True       |
               +-----------------------------------------+
                        (up to max_replans times)
```

After each evaluation cycle, `should_replan()` checks the reflection output. If it returns `True` and the replan budget is not exhausted, `apply_replan_strategy()` produces a fresh `(plan, profile)` pair with an escalated preprocessing setup and a broader model pool. The loop then runs again on the updated configuration. This is the core of the agentic behaviour: the system adjusts its own strategy based on what it observed, without any manual input.

---

## 3. Dataset Understanding

### What the Profiler Extracts

`tools/data_profiler.py` runs before any modelling. It pulls eight signals from the dataset, and each one feeds directly into a downstream planning decision:


| Signal                        | How it is detected                                            | What the planner does with it                                                               |
| ----------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Missing values                | `df.isna().mean()` per column                                 | Column missing >50% gets dropped; overall >30% triggers median imputation; otherwise mean   |
| Outliers                      | IQR x 3.0 on numeric columns; flagged if >1% of rows affected | RobustScaler instead of StandardScaler                                                      |
| Class imbalance               | majority count / minority count                               | Ratio >= 3.0 triggers `class_weight='balanced'`; primary metric becomes `balanced_accuracy` |
| High-cardinality categoricals | More than 20 unique values in a categorical column            | OrdinalEncoder rather than OneHotEncoder                                                    |
| Multicollinearity             | Pearson r above 0.9 between numeric pairs                     | Second column of each flagged pair is dropped                                               |
| High dimensionality           | More than 100 effective columns after drops                   | SelectKBest(k=50) using the ANOVA F-statistic                                               |
| Small dataset                 | Fewer than 1000 rows                                          | Simpler models preferred; GradientBoosting excluded                                         |
| Duplicate rows                | `df.duplicated().sum()`                                       | Note added to the profile; rows removed before training                                     |


### Why Each Signal Matters

Each of these is a data quality risk that can silently undermine model performance if nothing is done about it.

Outliers, for example, inflate the variance estimate used by StandardScaler. The result is that linear models under-weight the inlier range. RobustScaler uses the median and interquartile range instead, so extreme values do not affect how the bulk of the data gets scaled.

Class imbalance creates a different kind of problem. Standard accuracy becomes misleading when one class is rare: a model that always predicts the majority class can score high accuracy while being completely useless. `balanced_accuracy` computes the mean per-class recall, so it gives equal weight to each class regardless of how often it appears.

High-cardinality one-hot encoding produces hundreds of sparse binary columns. Tree models waste splits on them, and linear models see inflated noise. OrdinalEncoder avoids the explosion entirely.

The profiler also seeds `_plan_config`, a shared dictionary that `build_preprocessor()` and `select_models()` consume directly. This means every downstream decision traces back to a specific observation made on the data.

---

## 4. Planning Logic

### Conditional Decisions

`agents/planner.py::create_plan()` reads each signal from the profile and works through ten sequential conditional decisions, appending each one to the plan as a human-readable justification string:

```
IF any column missing > 50%        -> [drop_columns]
IF total_missing > 30%             -> impute = median      ELSE mean
IF outlier_cols not empty          -> scaler = RobustScaler ELSE StandardScaler
IF high_card_cols exist            -> OrdinalEncoder for those columns
IF low_card_cols exist             -> OneHotEncoder for those columns
IF corr_pairs detected             -> drop second column of each pair
IF effective_cols > 100            -> SelectKBest(k=50)
IF imbalance_ratio >= 3.0          -> class_weight='balanced'; metric = balanced_accuracy
IF rows < 1000                     -> prefer_simple_models = True
IF memory_hint.best_model exists   -> place that model first in candidate queue
```

### Memory-Guided Planning

When `JSONMemory.get_dataset_record(fingerprint)` returns a record from a previous run, the planner gets a `memory_hint` containing the best model name and its balanced accuracy score. It then:

1. Sets `_plan_config["priority_model"]` to that model name.
2. Adds a plan step that records the reasoning: *"prior run found for this fingerprint — placing that model first in the candidate queue."*
3. `select_models()` in `modelling.py` moves the priority model to position 1, so it is always trained even when the replan adds more candidates to the pool.

This creates a feedback loop between runs. If a model worked well before, it is guaranteed a place in the next run's evaluation rather than potentially being displaced.

### _plan_config as Shared State

`create_plan()` does more than return a list of strings. It mutates `dataset_profile["_plan_config"]` with concrete key-value decisions: which scaler to use, which imputer, which columns to drop, what k to pass to SelectKBest, and so on. This dictionary is the contract between the planner and the modelling tools. Every decision in the preprocessing pipeline can be traced back to a specific signal in the profile.

---

## 5. Tool Use: Modelling and Evaluation

### Adaptive Preprocessing

`tools/modelling.py::build_preprocessor()` reads `_plan_config` and assembles a `ColumnTransformer` with three independent sub-pipelines:

```
ColumnTransformer
|- num:       SimpleImputer(strategy) -> [Robust|Standard]Scaler
|- cat_low:   SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')
+- cat_high:  SimpleImputer(most_frequent) -> OrdinalEncoder(handle_unknown='use_encoded_value')
```

If `feature_selection == "select_k_best"`, the transformer is wrapped in a sklearn `Pipeline` that adds `SelectKBest(f_classif, k=k_best)` as a second step. This makes sure the feature selection runs on the transformed matrix rather than the raw input.

### Model Candidates

`select_models()` builds a candidate list from `_plan_config`:


| Model                            | When it is included                               |
| -------------------------------- | ------------------------------------------------- |
| `DummyClassifier(most_frequent)` | Always — establishes the performance baseline     |
| `LogisticRegression`             | Always                                            |
| `RandomForestClassifier(n=300)`  | Always                                            |
| `GradientBoostingClassifier`     | Excluded when `prefer_simple_models=True`         |
| `SVC(RBF)`                       | Only when rows <= 20,000 and cols <= 200          |
| `ExtraTreesClassifier`           | Added by the reflector on replan Tier 1 or higher |
| `VotingEnsemble(RF + GB)`        | Added by the reflector on replan Tier 2 or higher |
| `CalibratedClassifierCV`         | Added by the reflector on replan Tier 3           |


`class_weight='balanced'` is applied to all compatible models when the imbalance strategy is `class_weight`. When the reflector escalates to `smote` on a replan, `imblearn.pipeline.Pipeline` wraps a SMOTE resampler between the preprocessor and the model so oversampling only affects the training fold, not the test set.

Models are ranked by `balanced_accuracy` first and `f1_macro` as a tiebreak. The top-ranked model goes to full evaluation.

### Evaluation

`tools/evaluation.py::evaluate_best()` produces three things:

- A **confusion matrix PNG** saved to the output directory.
- A **sklearn classification report** with per-class precision, recall, and F1.
- A **paired t-test** using `scipy.stats.ttest_rel` over 5-fold stratified cross-validation, comparing the best model against every other non-dummy candidate. This tells the reflector whether the winning model's advantage is statistically meaningful or could be down to sampling variance.

The significance test result appears both in the Markdown report and in the reflector's diagnostic output, where it can contribute to a suggestion that a simpler model might be sufficient.

---

## 6. Reflection and Re-planning

### Diagnostic Checks

`agents/reflector.py::reflect()` runs five independent checks after each evaluation:


| Check                                   | Threshold                              | Severity |
| --------------------------------------- | -------------------------------------- | -------- |
| Improvement over dummy baseline         | balanced_accuracy delta < 0.05         | HIGH     |
| Absolute performance floor              | balanced_accuracy < 0.55               | HIGH     |
| Score spread across candidates          | >0.20 range                            | HIGH     |
| Score spread across candidates          | >0.10 range                            | MEDIUM   |
| Macro-F1                                | < 0.55                                 | MEDIUM   |
| Overfitting gap (train vs test)         | balanced_accuracy gap > 0.15           | MEDIUM   |
| Imbalance without a correction strategy | ratio >= 3.0 but no correction applied | MEDIUM   |


Each finding is tagged `[HIGH]`, `[MEDIUM]`, or `[LOW]` and appended to `issues`. A matching `suggestions` entry is also generated for each one — for example, a near-dummy result triggers *"Verify target column consistency; check for leakage in feature set."*

### When a Replan Fires

```python
replan_recommended = (severity_counts["high"] >= 1 OR severity_counts["medium"] >= 2)
```

The replan tier then determines how aggressively the agent escalates:


| Tier | Condition                  | What changes                                                                                         |
| ---- | -------------------------- | ---------------------------------------------------------------------------------------------------- |
| 0    | No issues                  | Nothing — run is accepted                                                                            |
| 1    | Score spread only          | Imputer escalates from mean to median; ExtraTrees added to model pool                                |
| 2    | 1 HIGH or 2+ MEDIUM issues | Everything in Tier 1, plus imbalance strategy moves from class_weight to SMOTE; VotingEnsemble added |
| 3    | Multiple HIGH issues       | Everything in Tier 2, plus KNNImputer; SelectKBest expanded; CalibratedClassifierCV added            |


### What Actually Changes

`apply_replan_strategy()` returns a new `(plan, profile)` pair without touching the originals, so the original run state is preserved for debugging. The updated `_plan_config` is picked up by `build_preprocessor()` and `select_models()` on the next loop iteration, producing a materially different pipeline with no manual input needed.

The orchestrator caps replanning at `max_replans` (default: 1) so the loop cannot run indefinitely. The replan count is written to memory so the agent can account for previous replan behaviour when it encounters the same dataset again.

---

## 7. Memory and Learning

### How Data is Stored

`agents/memory.py::JSONMemory` writes state to a local `agent_memory.json` file using an atomic write-then-rename operation (`os.replace`). This prevents a partial write from corrupting the file if the process is interrupted. The JSON file has three top-level keys:

```json
{
  "datasets": {
    "<fingerprint>": {
      "best_model": "RandomForest",
      "best_bal_acc": 0.847,
      "best_metrics": { ... },
      "best_plan_config": { ... },
      "run_count": 3,
      "first_seen": "2026-04-20T22:34:42Z",
      "last_seen": "2026-04-21T00:15:36Z"
    }
  },
  "run_history": [ { "fingerprint": "...", "bal_acc": 0.82, ... } ],
  "notes": [ { "ts": "...", "msg": "..." } ]
}
```

### Dataset Fingerprinting

Each dataset gets a deterministic fingerprint (`fp_{hash}`) computed from its shape, target column name, and column list. This lets the agent recognise the same dataset across separate runs without storing any of the actual data. Different datasets produce different fingerprints so records never collide.

### What Gets Stored and How It Gets Used


| Field              | How it is used                                                                          |
| ------------------ | --------------------------------------------------------------------------------------- |
| `best_model`       | Planner places it first in the candidate queue on the next run                          |
| `best_plan_config` | Available via `get_best_strategy()` for reuse by the planner                            |
| `run_count`        | Used in summary statistics and replan rate calculation                                  |
| `best_bal_acc`     | Guards best-ever updates — a new run only overwrites this if it actually improves on it |
| `score_trend`      | Computed by `_compute_trend()`: improving, stable, or degrading                         |
| `replan_rate`      | Fraction of runs that triggered a replan; included in the report                        |
| `avg_n_issues`     | Average number of diagnostic issues raised per run                                      |


### Recovery from Corruption

If the memory file is malformed or empty at load time, the agent copies it to `agent_memory.json.bak` and starts with a fresh in-memory store. The agent never crashes on startup because of a bad memory file.

---

## 8. Ethics and Limitations

### Ethical Considerations

**Bias:** The agent applies `class_weight='balanced'` when class imbalance is detected, which reduces the risk of a model that ignores the minority class entirely. But this correction addresses the symptom rather than the cause. If the minority class is genuinely under-represented in the data relative to the real world, no weighting strategy can compensate for that gap, only better data collection can.

**Transparency:** Every preprocessing and modelling decision is written to `plan.json` and `report.md` with a plain-English justification. A user can open either file and see exactly why a column was dropped, why RobustScaler was chosen, or why the primary metric was changed. Nothing is hidden in configuration.

**Privacy:** The system runs fully offline. No data is sent anywhere. The memory file stores only aggregate statistics: model names, scalar metrics, config keys; not any rows from the original dataset.

**Reproducibility:** Random state is seeded through the `--seed` argument (default 42). Train/test splits use `stratify=y` where possible. Given the same input file and seed, the agent produces the same outputs every time.

### Limitations

**Classification only:** The system is built around classification. The `is_classification_target()` heuristic can misfire on regression targets that have few unique float values, such as rounded prices. Regression would need a separate model pool, different metrics (R2, RMSE), and adjusted reflection thresholds.

**Single train/test split:** Models are evaluated on one held-out test set (default 20%). Metrics from a single split carry sampling variance, especially on small datasets. Full cross-validation during training would give more stable estimates but at a significant cost in runtime.

**No feature engineering:** The agent selects and scales existing features but does not create new ones. Polynomial interactions, date decomposition, and target encoding are left untouched. Real datasets often contain substantial signal that only appears in feature combinations.

**SMOTE requires imbalanced-learn:** The Tier 2 replan escalates to SMOTE oversampling, but this depends on `imblearn` being installed. If it is not available, the agent falls back silently to `class_weight` with no warning to the user that the intended strategy could not be applied.

**Memory is a flat local file:** The `JSONMemory` store is a single JSON file. It does not support concurrent writes from parallel runs, and it has no indexing, so it would slow down if very large numbers of datasets were registered; though this is unlikely in a local deployment.

---

## 9. Conclusion and Future Work

### What Was Built

This project delivers an autonomous, offline data scientist that makes data-driven decisions at every stage of the pipeline. 

The test suite covers 110 unit and integration tests across all four agent modules. All tests pass cleanly on the submitted codebase. The smoke test (test_smoke_run.py) runs a full end-to-end agent pass on demo.csv and asserts it exits with code 0, a confusion matrix exists, and a report is written.

The core contributions are:

1. A **signal-driven profiler** that reads eight dataset characteristics and seeds a shared `_plan_config` state consumed by every downstream component.
2. A **conditional planner** that converts those signals into a ten-step justified plan, with memory-guided model prioritisation on repeat runs.
3. A **tiered reflector** that applies five diagnostic rules, tags severity levels, and triggers escalated replan strategies automatically.
4. A **persistent memory module** that tracks best-ever performance, score trend, and replan history per dataset fingerprint across multiple runs.

### What Could Come Next

- **Regression support:** Extend the profiler, planner, and reflector with regression-specific logic including R2 and RMSE thresholds and a proper continuous target detector.
- **Feature engineering:** Add a feature engineering step that generates polynomial interactions, ratio features, and date-part extractions based on column types.
- **Cross-validation scoring:** Replace the single train/test split with k-fold cross-validation for more reliable metrics, especially on small datasets.
- **Hyperparameter tuning:** Add `RandomizedSearchCV` as an optional post-replan step when the reflector identifies a strong model that has not been tuned yet.
- **Cross-dataset strategy transfer:** Use embedding similarity between dataset fingerprints to borrow successful strategies from structurally similar but not identical datasets in memory.
- **Explainability output:** Generate SHAP values for the best model as a standard output artefact, so the agent can report which features were most influential.

---

## 10. How to Run

### Installation

```bash
pip install -r requirements.txt

# Optional: adds SMOTE support for Tier 2 replanning
pip install imbalanced-learn
```

### Basic Usage

```bash
python run_agent.py --data data/your_dataset.csv --target your_target_column
```

### Auto Target Detection

```bash
python run_agent.py --data data/your_dataset.csv --target auto
```

The agent infers the target column using name heuristics (`target`, `label`, `class`, `y`, `outcome`) and falls back to the last low-cardinality column if none of those match.

### All Options

```bash
python run_agent.py \
  --data        data/your_dataset.csv \
  --target      auto \
  --output      outputs/ \
  --seed        42 \
  --test-size   0.2 \
  --max-replans 1 \
  --quiet
```


| Argument        | Default    | Description                           |
| --------------- | ---------- | ------------------------------------- |
| `--data`        | required   | Path to the input CSV                 |
| `--target`      | required   | Target column name or `auto`          |
| `--output`      | `outputs/` | Root directory for run artefacts      |
| `--seed`        | `42`       | Random seed                           |
| `--test-size`   | `0.2`      | Fraction of data held out for testing |
| `--max-replans` | `1`        | Maximum replan iterations per run     |
| `--quiet`       | `False`    | Suppress verbose logging              |


### Output Files

Each run creates a timestamped subdirectory under `--output`:

```
outputs/20260421_033000_a1b2c3d4/
|- report.md              Human-readable Markdown run report
|- metrics.json           All candidate metrics plus best model detail
|- eda_summary.json       Full dataset profile dictionary
|- plan.json              Execution plan steps and justifications
|- reflection.json        Issues, suggestions, and replan tier
+- confusion_matrix.png
```

### Running the Tests

```bash
pip install pytest pytest-cov

# Run with coverage summary in the terminal
python -m pytest tests/ -v --cov=agents --cov=tools --cov-report=term-missing

# Generate a browsable HTML coverage report
python -m pytest tests/ --cov=agents --cov=tools --cov-report=html:coverage_html
open coverage_html/index.html
```

---

## 11. Repository Structure

```
.
|- agents/
|   |- planner.py              Conditional planning logic and replan config changes
|   |- reflector.py            Diagnostic reflection and tiered replan strategies
|   +- memory.py               Persistent JSON memory keyed by dataset fingerprint
|
|- tools/
|   |- data_profiler.py        Dataset signal extraction and _plan_config seeding
|   |- modelling.py            Adaptive preprocessor, model selection, and training
|   +- evaluation.py           Metrics, confusion matrix, significance test, report writer
|
|- tests/
|   |- conftest.py             Shared pytest fixtures
|   |- test_profiler.py        40 unit tests for data_profiler
|   |- test_planner.py         17 unit tests for planner
|   |- test_reflector.py       25 unit tests for reflector
|   |- test_memory.py          28 unit tests for memory
|   |- sanity_check.py         End-to-end subprocess check
|   +- test_smoke_run.py       Pytest smoke test for run_agent.py
|
|- data/
|   +- demo.csv                Small demo dataset for local testing
|
|- outputs/                    Created automatically; one subdirectory per run
|
|- report/
|    +- README.md              This file
|
|- agentic_data_scientist.py   Orchestrator (AgenticDataScientist class)
|- run_agent.py                CLI entry point
|- agent_memory.json           Persistent memory store (created on first run)
+- requirements.txt
```

