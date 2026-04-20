from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np

# setting all thresholds used
OUTLIER_IQR_FACTOR = 3.0  # IQR multiplier for detecting extreme values
OUTLIER_ROW_PCT = 1.0 # flags a colums when greater than this percentage of rows are outliers
HIGH_CARD_THRESHOLD = 20 # unique values above which a colums is considered high-cardinality
CORR_THRESHOLD = 0.9 # pearson correlation above which two columns are correlated


def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristic target inference:
      - prefer common target-like column names
      - else last column if it has relatively low cardinality
    """
    candidates = ["target", "label", "class", "y", "outcome"]
    lower_map = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in lower_map:
            return lower_map[k]

    last = df.columns[-1]
    uniq = df[last].nunique(dropna=True)
    n = len(df)
    if n > 0 and (uniq <= 50 or (uniq / max(n, 1) < 0.05)):
        return last
    return None


def is_classification_target(series: pd.Series) -> bool:
    if series.dtype == "object" or str(series.dtype).startswith("category"):
        return True
    uniq = series.nunique(dropna=True)
    return uniq <= 50


def dataset_fingerprint(df: pd.DataFrame, target: str) -> str:
    cols = ",".join(df.columns.astype(str).tolist())
    shape = f"{df.shape[0]}x{df.shape[1]}"
    base = f"{shape}|{target}|{cols}"
    h = abs(hash(base)) % (10**12)
    return f"fp_{h}"


def profile_dataset(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset columns.")

    y = df[target]
    profile: Dict[str, Any] = {}
    notes: List[str] = []

    # basic shape of the dataset
    profile["shape"] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    profile["columns"] = df.columns.astype(str).tolist()

    # missing values
    missing = (df.isna().mean() * 100).round(2).to_dict()
    profile["missing_pct"] = {str(k): float(v) for k, v in missing.items()}
    profile["total_missing_pct"] = round(float(df.isna().values.mean()*100), 3)

    if profile["total_missing_pct"] > 30:
        notes.append(f"High overall missingness ({profile['total_missing_pct']:.1f}%) - median imputation selected.")
    
    # target metadata
    profile["target"] = str(target)
    profile["target_dtype"] = str(y.dtype)
    profile["is_classification"] = bool(is_classification_target(y))

    # Feature type splitting
    X = df.drop(columns=[target])
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.astype(str).tolist()
    cat_cols = [c for c in X.columns.astype(str).tolist() if c not in numeric_cols]

    profile["feature_types"] = {"numeric": numeric_cols, "categorical": cat_cols}
    profile["n_unique_by_col"] = {str(c): int(df[c].nunique(dropna=True)) for c in df.columns.astype(str)}

    # notes for small/high dimensionality datasets
    if profile["shape"]["rows"] < 1000:
        notes.append("Small dataset (<1000 rows): prefer simpler models / guard against overfitting.")
    if profile["shape"]["cols"] > 100:
        notes.append("High dimensionality (>100 columns): watch one-hot expansion and overfitting.")
    profile["notes"] = notes


    profile["duplicate_rows"] = int(df.duplicated().sum())
    if profile["duplicate_rows"] > 0:
        notes.append(
            f"{profile['duplicate_rows']} duplicate rows detected - will be removed before training."
    )
    # outlier detection using IQR
    outlier_cols: List[str] = []
    n_rows = profile["shape"]["rows"]
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        n_extreme = int(((series < q1 - OUTLIER_IQR_FACTOR * iqr) | (series > q3 + OUTLIER_IQR_FACTOR * iqr)).sum())
        if n_rows > 0 and (n_extreme /n_rows * 100) > OUTLIER_ROW_PCT:
            outlier_cols.append(col)
    profile["outlier_cols"] = outlier_cols
    if outlier_cols:
        notes.append(f"Outliers have been detected in {outlier_cols} (IQRx{OUTLIER_IQR_FACTOR}, >{OUTLIER_ROW_PCT}% of rows) - Robust Scaler would be used.")

    # high cardinality detection in categoricals
    high_card_cols: List[str] = [
        c for c in cat_cols if profile["n_unique_by_col"].get(c, 0) > HIGH_CARD_THRESHOLD
    ]
    profile["high_card_cols"] = high_card_cols
    if high_card_cols:
        notes.append(f"High cardinality categoricals detected: {high_card_cols} (>{HIGH_CARD_THRESHOLD} unique values) - Ordinal Encoder would be used.")
    
    # correlation detection using pearson correlation
    corr_pairs: List[Tuple[str, str]] = []
    drop_corr_cols: List[str] = []
    if len(numeric_cols) > 1:
        corr_matrix = (df[numeric_cols].corr(method="pearson", numeric_only=True).abs())
        # check to avoid duplicate pairs
        upper_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper = corr_matrix.where(upper_mask)
        for col in upper.columns:
            for row in upper.index:
                val = upper.loc[row, col]
                if pd.notna(val) and val > CORR_THRESHOLD:
                    corr_pairs.append((col, row))
    profile["corr_pairs"] = corr_pairs
    if corr_pairs:
        drop_corr_cols = list({p[1] for p in corr_pairs})
        notes.append(f"High correlation pairs detected: (|r|>{CORR_THRESHOLD}) : pairs={corr_pairs} - dropping {drop_corr_cols} to avoid multicollinearity.")
    
    profile["drop_corr_cols"] = drop_corr_cols
    profile["notes"] = notes

    # Class balance (for only classification)
    if profile["is_classification"]:
        vc = y.value_counts(dropna=False)
        profile["class_counts"] = {str(k): int(v) for k, v in vc.items()}
        if len(vc) >= 2:
            ratio = float(vc.max() / max(vc.min(), 1))
        else:
            ratio = 1.0
        profile["imbalance_ratio"] = round(ratio, 3)
        if ratio >= 3.0:
            profile["notes"].append(f"Class imbalance detected (ratio={ratio:.2f} >= 3.0) - class_weight='balanced' and balance_accuracy metric would be used.")
    else:
        profile["class_counts"] = None
        profile["imbalance_ratio"] = None
        profile["notes"].append("Non-classification target detected: this template focuses on classification.")

    profile["notes"] = notes

    # challenges list (used by planner for step descriptions)
    challenges: List[str] = []
    if profile["duplicate_rows"] > 0:
        challenges.append("duplicate_rows")
    if profile["total_missing_pct"] > 0:
        challenges.append("missing_values")
    if outlier_cols:
        challenges.append("outliers")
    if high_card_cols:
        challenges.append("high_cardinality")
    if corr_pairs:
        challenges.append("multicollinearity")
    if (profile["is_classification"] and
            profile["imbalance_ratio"] and
            profile["imbalance_ratio"] >= 3.0):
        challenges.append("class_imbalance")
    profile["challenges"] = challenges

    # _plan_config: consumed by build_preprocessor(), select_models(), train_models()
    n_features = len(numeric_cols) + len(cat_cols)
    impute_strategy = "median" if profile["total_missing_pct"] > 30.0 else "mean"
    scaler = "robust" if outlier_cols else "standard"
    handle_imbalance = "none"
    if (profile["is_classification"] and
            profile["imbalance_ratio"] and
            profile["imbalance_ratio"] >= 3.0):
        handle_imbalance = "class_weight"

    feature_selection = "select_k_best" if n_features > 50 else "none"
    k_best = 30 if feature_selection == "select_k_best" else None
    primary_metric = (
        "f1_weighted"
        if (profile["is_classification"] and
            profile["imbalance_ratio"] and
            profile["imbalance_ratio"] >= 3.0)
        else ("accuracy" if profile["is_classification"] else "r2")
    )

    profile["_plan_config"] = {
        "impute_strategy": impute_strategy,
        "scaler": scaler,
        "handle_imbalance": handle_imbalance,
        "high_card_cols": high_card_cols,
        "drop_corr_cols": drop_corr_cols,
        "feature_selection": feature_selection,
        "k_best": k_best,
        "primary_metric": primary_metric,
        "prefer_simple_models": profile["shape"]["rows"] < 1000,
        "priority_model": None,
        "extra_models": [],
    }

    return profile
