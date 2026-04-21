from typing import Any, Dict, List, Tuple, Optional

import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# optional: imbalanced-learn for oversampling (replan strategy)
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


def build_preprocessor(profile: Dict[str, Any]) -> ColumnTransformer:
    """
    Build a preprocessing pipeline driven by _plan_config in the profile.

    Reads from profile["_plan_config"]:
        impute_strategy : "mean" | "median"  (default "median")
        scaler : "standard" | "robust"  (default "standard")
        high_card_cols : List[str]  -> OrdinalEncoder instead of OneHotEncoder
        drop_corr_cols : List[str]  -> removed before building transformers
        feature_selection : "select_k_best" | None
        k_best : int  (default 30)

    Returns:
        A ColumnTransformer (no feature selection) or a sklearn Pipeline
        (ColumnTransformer -> SelectKBest) when feature_selection is set.
        Both are valid as the "preprocess" step inside train_models().
    """
    cfg = profile.get("_plan_config", {})

    # column lists from the profiler
    num_cols  = list(profile["feature_types"]["numeric"])
    cat_cols  = list(profile["feature_types"]["categorical"])

    # config values with some safe defaults
    impute_strategy = cfg.get("impute_strategy", "median")
    scaler_type = cfg.get("scaler", "standard")
    high_card_cols = cfg.get("high_card_cols", [])
    drop_corr_cols = cfg.get("drop_corr_cols", [])
    feature_selection = cfg.get("feature_selection")
    k_best = int(cfg.get("k_best") or 30)

    # remove the correlated featured flagged by the planner
    num_cols  = [c for c in num_cols  if c not in drop_corr_cols]
    cat_cols  = [c for c in cat_cols  if c not in drop_corr_cols]

    # split the categoricals
    high_card_active = [c for c in high_card_cols if c in cat_cols]
    low_card_cols    = [c for c in cat_cols if c not in high_card_active]

    # choose the imputer
    # choose the scaler based on _plan_config
    # RobustScaler used when data_profiler detected outlier columns so that extreme values do not distort the scaling of the inlier range.
    if impute_strategy == "knn":
        num_imputer = KNNImputer(n_neighbors=5)
    else:
        num_imputer = SimpleImputer(strategy=impute_strategy if impute_strategy in ("mean", "median") else "median")
    scaler = (
        RobustScaler()
        if scaler_type == "robust"
        else StandardScaler(with_mean=True)
    )

    # numeric transformer
    numeric_transformer = Pipeline(steps=[
        ("imputer", num_imputer),
        ("scaler",  scaler),
    ])

    # low cardinality categorical transformer
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    low_card_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  ohe),
    ])

    # high cardinality categorical transformer
    # OrdinalEncoder avoids the dimensionality explosion that OneHotEncoder
    high_card_transformer = Pipeline(steps=[
        ("imputer",  SimpleImputer(strategy="most_frequent")),
        ("ordinal",  OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    # assembles the column transformer
    transformers = []
    if num_cols:
        transformers.append(("num", numeric_transformer, num_cols))
    if low_card_cols:
        transformers.append(("cat_low", low_card_transformer, low_card_cols))
    if high_card_active:
        transformers.append(("cat_high", high_card_transformer, high_card_active))

    if not transformers:
        transformers.append(("num", numeric_transformer, num_cols or list(
            profile["feature_types"]["numeric"]
        )))

    ct = ColumnTransformer(transformers=transformers, remainder="drop")

    # optionally append SelectKBest
    # planner activates this when >100 effective features exist, or when the reflector triggers a replan due to high-spread results.
    if feature_selection == "select_k_best":
        return Pipeline(steps=[
            ("preprocess", ct),
            ("select", SelectKBest(f_classif, k=k_best)),
        ])

    return ct

def select_models(profile: Dict[str, Any], seed: int = 42) -> List[Tuple[str, Any]]:
    """
    Build the list of model candidates driven by _plan_config.

    Reads from profile["_plan_config"]:
        handle_imbalance : "class_weight" | "oversample" | None
            - class_weight="balanced" when "class_weight"
            - class_weight=None when "oversample" or "smote"
        extra_models : List[str]  — added by reflector on replan
            supported: "ExtraTreesClassifier", "VotingEnsemble", "CalibratedClassifier"
        priority_model : str | None — from memory hint (best past model)
            - moved to front of candidate list (trained first)
        prefer_simple_models : bool — set for small datasets (<500 rows)
            - skips GradientBoosting and SVC

    Returns:
        List of (name, estimator) tuples with DummyClassifier always first.
    """
    cfg = profile.get("_plan_config", {})
    rows = profile["shape"]["rows"]
    cols = profile["shape"]["cols"]

    handle_imbalance = cfg.get("handle_imbalance")
    extra_models = cfg.get("extra_models", [])
    priority_model = cfg.get("priority_model")
    prefer_simple = cfg.get("prefer_simple_models", False)

    class_weight = "balanced" if handle_imbalance == "class_weight" else None

    # base candidates
    candidates: List[Tuple[str, Any]] = [
        ("DummyMostFrequent", DummyClassifier(strategy="most_frequent")),
        ("LogisticRegression", LogisticRegression(
            max_iter=2000, class_weight=class_weight, random_state=seed,
        )),
        ("RandomForest", RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1, class_weight=class_weight,
        )),
    ]
    
    if not prefer_simple and rows <= 50_000:
        candidates.append(("GradientBoosting", GradientBoostingClassifier(random_state=seed,
        )))

    if not prefer_simple and rows <= 20_000 and cols <= 200:
        candidates.append(("SVC_RBF", SVC(kernel="rbf", probability=True, class_weight=class_weight,
        )))

    # extra models from reflector replan 
    extra_map: Dict[str, Any] = {
        "ExtraTreesClassifier": ExtraTreesClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1, class_weight=class_weight,
        ),
        "VotingEnsemble": VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1, class_weight=class_weight)),
                ("gb", GradientBoostingClassifier(n_estimators=200, random_state=seed)),
            ],
            voting="soft",
        ),
        "CalibratedClassifier": CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1, class_weight=class_weight), cv=3, method="isotonic",
        ),
    }
    existing_names = {name for name, _ in candidates}
    for model_name in extra_models:
        if model_name in extra_map and model_name not in existing_names:
            candidates.append((model_name, extra_map[model_name]))

    # prioritise memory hint model 
    # when memory records a strong past model for this dataset fingerprint, the planner sets priority_model so it trains first. 
    # if it is not already in the candidate list it is inserted after the dummy baseline.
    if priority_model and priority_model not in existing_names:
        if priority_model in extra_map:
            candidates.insert(1, (priority_model, extra_map[priority_model]))

    # move priority_model to position 1 if already present
    if priority_model:
        idx = next(
            (i for i, (n, _) in enumerate(candidates) if n == priority_model),
            None,
        )
        if idx is not None and idx > 1:
            candidates.insert(1, candidates.pop(idx))

    return candidates

def train_models(
    df: pd.DataFrame,
    target: str,
    preprocessor,
    candidates: List[Tuple[str, Any]],
    seed: int,
    test_size: float,
    output_dir: str,
    verbose: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train all candidate models and return a payload with the best result.

    Parameters: 
        config : optional _plan_config dict from the profile.
            When config["handle_imbalance"] == "oversample" AND imbalanced-learn
            is installed, a resampler is inserted into the pipeline
            between preprocessing and the model so the training set is
            synthetically balanced. If imbalanced-learn is not installed, falls back silently.
    """
    cfg = config or {}

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # drop rows where target is missing
    mask = ~y.isna()
    X, y = X.loc[mask], y.loc[mask]

    stratify = (
        y if (y.nunique(dropna=True) > 1 and y.value_counts().min() >= 2)
        else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    # oversampling: either oversample using RandomOverSampler or SMOTE
    imbalance_strategy = cfg.get("handle_imbalance", "")
    use_resample = imbalance_strategy in ("oversample", "smote") and HAS_IMBLEARN

    if use_resample and verbose:
        sampler_name = "SMOTE" if imbalance_strategy == "smote" else "RandomOverSampler"
        print(f"[Modelling] Oversampling strategy: {sampler_name} (imblearn)")
    elif imbalance_strategy in ("oversample", "smote") and not HAS_IMBLEARN:
        print(
            f"[Modelling] WARNING: handle_imbalance='{imbalance_strategy}' requested but imbalanced-learn is not installed. "
            "Falling back to class_weight strategy."
        )

    results: List[Dict[str, Any]] = []

    for name, model in candidates:
        if verbose:
            print(f"[Modelling] Training: {name}")

        is_dummy = "Dummy" in name

        if use_resample and not is_dummy:
            resampler = (
                SMOTE(random_state=seed)
                if imbalance_strategy == "smote" else RandomOverSampler(random_state=seed)
            )
            pipe = ImbPipeline(steps=[
                ("preprocess", preprocessor),
                ("oversample", resampler),
                ("model", model),
            ])
        else:
            pipe = Pipeline(steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ])

        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
        except Exception as exc:
            if verbose:
                print(f"[Modelling] SKIP {name}: {exc}")
            continue

        metrics = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        }

        results.append({
            "name": name,
            "pipeline": pipe,
            "metrics": metrics,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
        })

    if not results:
        raise RuntimeError("All model candidates failed during training.")

    # sort: best balanced_accuracy first, then best macro F1 as tiebreak
    results.sort(
        key=lambda r: (
            r["metrics"]["balanced_accuracy"],
            r["metrics"]["f1_macro"],
        ),
        reverse=True,
    )

    return {
        "results": results,
        "best": results[0],
        "all_metrics": [r["metrics"] for r in results],
    }

