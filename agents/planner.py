"""
Planner Agent - Students must extend this significantly

The planner analyzes dataset characteristics and generates an execution plan.
Your task is to implement sophisticated planning logic that adapts to different
dataset types, sizes, and characteristics.

TODO: Extend this module with:
1. Sophisticated planning logic based on dataset profiles
2. Different plan templates for different scenarios
3. Memory-guided planning (use past successful strategies)
4. Dependency management (task ordering)
5. Conditional planning (if X then Y else Z)
6. Fallback strategies for edge cases
"""

from typing import Any, Dict, List, Optional, Tuple

# thresholds
_MISSING_DROP_PCT = 50
_MISSING_HIGH_PCT = 30
_IMBALANCE_THRESHOLD = 3
_HIGH_DIM_THRESHOLD = 100
_SMALL_DATASET_ROWS = 1000
_K_BEST_DEFAULT = 50
_K_BEST_REPLAN = 30


def create_plan(
    dataset_profile: Dict[str, Any], 
    memory_hint: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Analyse the dataset profile produce by the data_profiler.py and generate a data-aware executiion plan.
    
    Args:
        dataset_profile: Dictionary containing dataset metadata including:
            - shape: {rows: int, cols: int}
            - feature_types: {numeric: List[str], categorical: List[str]}
            - missing_pct: {col: float} - per-column missing %
            - total_missing_pct: float - overall missing %
            - outlier_cols: List[str]
            - high_card_cols: List[str]
            - corr_pairs: List[tuple]
            - imbalance_ratio: float | none
            - is_classification: bool
            - n_unique_by_col: {col: int}
            - notes: List[str]
            
        memory_hint: Optional dict with info from previous runs on similar datasets
    
    Returns:
        List[str]
            Human-readable reasoning strings written into report.md
            ALSO mutates dataset_profile["_plan_config"] 
        
    Example:
        >>> profile = {"shape": {"rows": 5000}, "imbalance_ratio": 4.5}
        >>> plan = create_plan(profile)
        >>> print(plan)
        ['profile_dataset', 'consider_imbalance_strategy', 'train_models', ...]
    
    TODO for students:
    - Implement conditional logic based on dataset size
    - Add different strategies for imbalanced datasets
    - Handle high-cardinality categorical features
    - Use memory hints to prioritize successful models
    - Create plan templates for common scenarios
    - Add preprocessing steps based on data quality
    """
    
    # Basic plan structure (students should make this much more sophisticated)
    plan: List[str] = []
    config: Dict[str, Any] = {}

    # read every signal from the data profiler
    rows = dataset_profile["shape"]["rows"]
    cols = dataset_profile["shape"]["cols"]
    is_clf = bool(dataset_profile.get("is_classification", True))
    imb = float(dataset_profile.get("imbalance_ratio") or 1)
    missing_pct = dataset_profile.get("missing_pct", {})
    total_missing = float(dataset_profile.get("total_missing_pct", 0))
    outlier_cols = list(dataset_profile.get("outlier_cols", []))
    high_card_cols = list(dataset_profile.get("high_card_cols", []))
    corr_pairs = list(dataset_profile.get("corr_pairs", []))
    num_cols = dataset_profile.get("feature_types", {}).get("numeric", [])
    cat_cols = dataset_profile.get("feature_types", {}).get("categorical", [])
    task = "classification" if is_clf else "regression"

    # step 1: dataset summary
    plan.append(
        f"[profile_dataset] {rows:,} rows x {cols} cols | task={task} | "
        f"total_missing={total_missing:.1f}% | imbalance_ration={imb:.2f}"
    )

    # step 2: drop columns with >50% missingness
    # profiler key: missing_pct
    drop_high_missing = [
        c for c, pct in missing_pct.items()
        if float(pct) > _MISSING_DROP_PCT and c in num_cols + cat_cols
    ]
    config["drop_high_missing_cols"] = drop_high_missing
    if drop_high_missing:
        plan.append(
            f"[drop_columns] {drop_high_missing} exceed {_MISSING_DROP_PCT}%"
            f"missing -> dropping before preprocessing "
            f"(imputing >50% of a column would inject more noise than signal)."
        )
    else:
        plan.append(
            f"[drop_columns] No column exceeds the {_MISSING_DROP_PCT}% missing threshold -> all features retained"
        )
    
    # step 3: imputation strategy
    # profiler key: total_missing_pct
    if total_missing > _MISSING_HIGH_PCT:
        config["impute_strategy"] = "median"
        plan.append(
            f"[impute_missing] total_missing_pct={total_missing:.1f}% "
            f" exceeds {_MISSING_HIGH_PCT}% -> median imputation "
            "(median is robust to skewed distributions and outliers; mean would be pulled toward extreme values)."
        )
    elif total_missing > 0:
        config["impute_strategy"] = "mean"
        plan.append(
            f"[impute_missing] total_missing_pct={total_missing:.1f}% "
            f" is low (<={_MISSING_HIGH_PCT}%) -> mean imputation (fast and unbiased when data are approximately normal)."
        )
    else:
        config["impute_strategy"] = "mean"
        plan.append(
            "[impute_missing] No missing values detected -> mean imputation applied as a safety net for unseen data during inference time."
        )

    # step 4: scaling strategy
    # profiler key: outlier_cols
    if outlier_cols:
        config["scaler"] = "robust"
        plan.append(
            f"[scale_features] outlier_cols={outlier_cols} "
            f"(IQRx3 detection in data_profiler.py) -> RobustScaler (scales by median and IQR; extreme values do not distort the scaling of the inlier range the way that StandardScaler would)."
        )
    else:
        config["scaler"] = "standard"
        plan.append(
            f"[scale_features] No outlier columns detected -> StandardScaler (zero-mean, unit-variance; optimal for linear models and distance-based algorithms when the data is well-behaved)."
        )

    # step 5: categorical encoding
    # profile key: high_card_cols 

    # 5a - for high-cardinality columns
    high_card_active = [
        c for c in high_card_cols
        if c not in config["drop_high_missing_cols"]
    ]
    config["high_card_cols"] = high_card_active
    config["high_card_encoding"] = "ordinal" if high_card_active else None
    if high_card_active:
        plan.append(
            f"[encode_high_card] high_card_cols={high_card_active} "
            f"(>{20} unique values per data_profiler.py) -> OrdinalEncoder "
            "(avoids one-hot dimensionality explosion; unknown categories at inference mapped to -1 via handle_unknown='use_encoded_value')."
        )

    # 5b - low-cardinality columns
    low_card_active = [
        c for c in cat_cols
        if c not in high_card_active
        and c not in config["drop_high_missing_cols"]
    ]
    config["low_card_encoding"] = "onehot" if low_card_active else None
    if low_card_active:
        plan.append(
            f"[encode_low_card] {len(low_card_active)} low-cardinality column(s) {low_card_active} -> OneHotEncoder "
            "(creates orthogonal binary features; no ordinal assumption; handle_unknown='ignore' prevents errors on unseen categories)."
        )
    if not high_card_active and not low_card_active:
        plan.append(
            "[encode_categoricals] No categorical columns found."
        )

    # step 6: drop correlated features
    # profiler key: corr_pairs
    drop_corr = list({pair[1] for pair in corr_pairs})
    config["drop_corr_cols"] = drop_corr
    if corr_pairs:
        plan.append(
            f"[drop_correlated] corr_pairs={corr_pairs} (Pearson |r| > 0.9 detected by data_profiler.py) -> "
            f"dropping {drop_corr} "
            "(retaining both couluns of a highly correlated pair inflates variance without adding any information; keeping the first of each pair preserves the signal)."
        )
    else:
        plan.append(
            "[drop_correlated] No feature pairs with |r| > 0.9 -> all numeric features retained"
        )

    # step 7: dimensionality reduction
    # profiler key: shape.cols (after accounting for the planned drops)
    effective_cols = (
        cols
        - len(config["drop_corr_cols"])
        - len(config["drop_high_missing_cols"])
    )
    if effective_cols > _HIGH_DIM_THRESHOLD:
        k = min(_K_BEST_DEFAULT, effective_cols - 1)
        config["feature_selection"] = "select_k_best"
        config["k_best"] = k
        plan.append(
            f"[feature_selection] {effective_cols} effective features "
            f"exceeds threshold ({_HIGH_DIM_THRESHOLD}) -> SelectKBest(k={k}) using ANOVA F-statistc "
            "(retains the k features most correlated with the target; reduces noise and overfitting risk on high-dimensionality data)."
        )
    else:
        config["feature_selection"]= None
        config["k_best"] = None
        plan.append(
            f"[feature_selection] {effective_cols} effective features is within the threshold ({_HIGH_DIM_THRESHOLD}) -> no dimensionality reduction is needed."
        )
    
    # step 8: class imbalance strategy
    # profiler key: imbalance_ratio, is_classification
    if is_clf and imb >= _IMBALANCE_THRESHOLD:
        config["handle_imbalance"] = "class_weight"
        config["primary_metric"] = "balanced_accuracy"
        config["secondary_metric"] = "f1_macro"
        plan.append(
            f"[handle_imbalance] imbalance_ratio={imb:.2f} >= {_IMBALANCE_THRESHOLD} -> class_weight='balanced' on all models "
            "(re-weights the loss function so minority-class errors carry more penalty: preferred over SMOTE on first pass - no synthetic data generation is required, no risk of over-generalising from duplicates)."
        )
        plan.append(
            "[select metric] Imbalanced task -> primary metric = balanced_accuracy (mean per-class recall; unaffected by class frequency skew); secondary = macro-F1."
        )
    elif is_clf:
        config["handle_imbalance"] = None
        config["primary_metric"] = "balanced_accuracy"
        config["secondary_metric"] = "f1_macro"
        plan.append(
            f"[handle_imbalance] imbalance_ratio={imb:.2f} < {_IMBALANCE_THRESHOLD} -> no imbalance correction needed."
        )
        plan.append(
            "[select_metric] Balanced classification -> balanced_accuracy + macro-F1."
        )
    else:
        config["handle_imbalance"] = None
        config["primary_metric"] = "r2"
        config["secondary_metric"] = "rmse"
        plan.append("[select_metric] Regression task -> R^2 + RMSE.")

    # step 9: dataset-size guard
    # profiler key: shape.rows
    if rows < _SMALL_DATASET_ROWS:
        config["small_dataset"] = True
        config["prefer_simpler_models"] = True
        plan.append(
            f"[size_guard] {rows} rows < {_SMALL_DATASET_ROWS} -> prefer regularized, lower-variance models (LogisticRegression, small RandomForest with max_depth cap); complex models risk overfitting on limited samples."
        )
    else:
        config["small_dataset"] = False
        config["prefer_simpler_models"] = False
        plan.append(
            f"[size_guard] {rows:,} rows >= {_SMALL_DATASET_ROWS} -> full model pool is vibale"
        )
    
    # step 10: memory-guided model prioritization
    # source: memory_hint from agents/memory.py
    if memory_hint and memory_hint.get("best_model"):
        best = memory_hint["best_model"]
        prev_score = (memory_hint.get("best_metrics") or {}).get("balanced_accuracy", 0)
        config["priority_model"] = best
        plan.append(
            f"[memory_guided] prior run found for this dataset fingerprint -> '{best}' achieved balanced_accuracy={prev_score:.3f} previously; placing it first in the candidate queue so it is always benchmarked."
        )
    else:
        config["priority_model"] = None
        plan.append(
            "[memory_guided] No prior memory for this dataset fingerprint -> all candidate models evaluated from scratch."
        )

    # inject config into profile (this is a shared state consumed by modelling.py)
    config["extra_models"] = []     # populated by get_replan_config_changes()
    dataset_profile["_plan_config"] = config

    # standard pipeline tail
    plan.extend([
        "[build_preprocessor] Assembling adaptive ColumnTransformer from _plan_config (scaler, imputer, encoders, column drops).",
        "[select_models] Choosing candidate models using _plan_config (class_weight, size guard, priority_model, extra_models).",
        "[train_models] Training all candidates on stratified train split; ranking by the primary metric.",
        "[evaluate] Generatng confusion matrix + classification report for the best model.",
        "[reflect] Analyzing performance; deciding whether to re-plan.",
        "[write_report] Saving Markdown report + JSON artefacts to output_dir.",
    ])

    return plan

####################################################################
# replan helper - called by reflector.py apply_replan_strategy()
def get_replan_config_changes(
    current_config: Dict[str, Any],
    reflection: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Produce an updated _plan_config and a list of reasoning strings explaining what changed. Called by apply_replan_strategy() in reflector.py.

    Strategy escalation ladder (each step is applied if it is applicable):
        imputer : mean -> median
        scaler : standard -> robust
        imbalance : class_weight -> oversample (random oversampling)
        features: none -> SelectKBest (k=30)
        model pool: add ExtraTreesClassifier
    """
    new_config = dict(current_config)
    changes: List[str] = []

    # escalate the imputer
    if new_config.get("impute_strategy") == "mean":
        new_config["impute_strategy"] = "median"
        changes.append(
            "[replan:imputer] mean -> median (more robust if missing data are not missing completely at random)."
        )

    # escalate the scaler
    if new_config.get("scaler") == "standard":
        new_config["scaler"] = "robust"
        changes.append(
            f"[replan:features] AddingSelectKBest (k={_K_BEST_REPLAN}) (noisy or redundant features may be suppressing model performance)."
        )
    
    # expand the model pool
    new_config["extra_models"] = ["ExtraTreesClassifier"]
    changes.append(
        "[replan:models] Adding ExtraTressClassifier (higher randomization than RandomForest; often more robust on noisy or high-variance datasets)."
    )

    return new_config, changes


    