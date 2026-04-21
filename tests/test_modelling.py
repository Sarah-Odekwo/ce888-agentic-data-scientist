import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from tools.modelling import build_preprocessor, select_models, train_models


def _make_profile(num_cols=None, cat_cols=None, imb=1.0, rows=2000, cols=5):
    num_cols = num_cols or ["age", "income"]
    cat_cols = cat_cols or ["gender"]
    return {
        "shape": {"rows": rows, "cols": cols},
        "feature_types": {"numeric": num_cols, "categorical": cat_cols},
        "is_classification": True,
        "imbalance_ratio": imb,
        "_plan_config": {
            "impute_strategy": "mean",
            "scaler": "standard",
            "high_card_cols": [],
            "drop_corr_cols": [],
            "drop_high_missing_cols": [],
            "feature_selection": None,
            "k_best": None,
            "handle_imbalance": None,
            "primary_metric": "balanced_accuracy",
            "secondary_metric": "f1_macro",
            "small_dataset": False,
            "prefer_simpler_models": False,
            "priority_model": None,
            "extra_models": [],
        },
    }


def test_build_preprocessor_basic():
    """build_preprocessor(profile) should return a transformer without error."""
    profile = _make_profile()
    preprocessor = build_preprocessor(profile)
    assert preprocessor is not None


def test_build_preprocessor_returns_transformer():
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline as SKPipeline
    profile = _make_profile()
    preprocessor = build_preprocessor(profile)
    assert isinstance(preprocessor, (ColumnTransformer, SKPipeline))


def test_select_models_includes_dummy():
    """select_models must always include a DummyClassifier entry."""
    profile = _make_profile()
    models = select_models(profile)
    names = [m["name"] for m in models]
    assert any("Dummy" in n for n in names)


def test_select_models_returns_list_of_dicts():
    profile = _make_profile()
    models = select_models(profile)
    assert isinstan


