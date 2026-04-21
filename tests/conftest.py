"""
Shared pytest fixtures for the agentic data scientist test suite.
Drop this into tests/conftest.py — no changes to any other file needed.
"""
import os
import json
import tempfile
import pytest
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Balanced 100-row DataFrame (3 numeric, 1 low-card categorical, binary target)
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "age":       rng.integers(20, 60, 100).astype(float),
        "income":    rng.normal(50_000, 10_000, 100),
        "score":     rng.uniform(0, 1, 100),
        "plan_type": rng.choice(["A", "B", "C"], 100),
        "churn":     rng.choice([0, 1], 100),
    })


# ---------------------------------------------------------------------------
# Imbalanced DataFrame (80/20 → ratio >= 3.0)
# ---------------------------------------------------------------------------
@pytest.fixture
def imbalanced_df():
    rng = np.random.default_rng(1)
    n = 200
    labels = [0] * 160 + [1] * 40
    rng.shuffle(labels)
    return pd.DataFrame({
        "x1":     rng.normal(0, 1, n),
        "x2":     rng.normal(0, 1, n),
        "target": labels,
    })


# ---------------------------------------------------------------------------
# DataFrame with 15% missing values in two columns
# ---------------------------------------------------------------------------
@pytest.fixture
def missing_df():
    rng = np.random.default_rng(2)
    n = 100
    df = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "y": rng.choice([0, 1], n),
    })
    df.loc[rng.choice(n, 15, replace=False), "a"] = float("nan")
    df.loc[rng.choice(n, 15, replace=False), "b"] = float("nan")
    return df


# ---------------------------------------------------------------------------
# DataFrame with a perfectly correlated pair (|r| ~ 1.0)
# ---------------------------------------------------------------------------
@pytest.fixture
def correlated_df():
    rng = np.random.default_rng(3)
    n = 100
    x = rng.normal(0, 1, n)
    return pd.DataFrame({
        "x1":     x,
        "x2":     x * 12.0,
        "noise":  rng.normal(0, 1, n),
        "target": rng.choice([0, 1], n),
    })


# ---------------------------------------------------------------------------
# DataFrame with deliberate outliers in one column (>1% of rows)
# ---------------------------------------------------------------------------
@pytest.fixture
def outlier_df():
    rng = np.random.default_rng(4)
    n = 200
    vals = rng.normal(0, 1, n).tolist()
    for i in range(5):
        vals[i] = 999.0
    return pd.DataFrame({
        "outlier_col": vals,
        "normal_col":  rng.normal(0, 1, n),
        "target":      rng.choice([0, 1], n),
    })


# ---------------------------------------------------------------------------
# DataFrame with a high-cardinality categorical column (35 unique values)
# ---------------------------------------------------------------------------
@pytest.fixture
def high_card_df():
    rng = np.random.default_rng(5)
    n = 150
    return pd.DataFrame({
        "region":  [f"REG_{i % 35:02d}" for i in range(n)],
        "value":   rng.normal(0, 1, n),
        "target":  rng.choice([0, 1], n),
    })


# ---------------------------------------------------------------------------
# Temporary directory + memory path for JSONMemory tests
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def memory_path(tmp_dir):
    return os.path.join(tmp_dir, "test_memory.json")


# ---------------------------------------------------------------------------
# Minimal profile dict (used by planner & reflector tests)
# ---------------------------------------------------------------------------
@pytest.fixture
def base_profile():
    return {
        "shape": {"rows": 2000, "cols": 10},
        "feature_types": {"numeric": ["a", "b"], "categorical": ["c"]},
        "imbalance_ratio": 1.2,
        "missing_pct": {"a": 0.0, "b": 0.0, "c": 0.0},
        "total_missing_pct": 0.0,
        "is_classification": True,
        "notes": [],
        "outlier_cols": [],
        "high_card_cols": [],
        "corr_pairs": [],
        "_plan_config": {},
    }


# ---------------------------------------------------------------------------
# Minimal evaluation dict (used by reflector tests)
# ---------------------------------------------------------------------------
@pytest.fixture
def good_eval():
    return {
        "model": "RandomForest",
        "accuracy": 0.87,
        "balanced_accuracy": 0.82,
        "f1_macro": 0.81,
        "precision_macro": 0.82,
        "recall_macro": 0.81,
    }


@pytest.fixture
def poor_eval():
    return {
        "model": "LogisticRegression",
        "accuracy": 0.55,
        "balanced_accuracy": 0.53,
        "f1_macro": 0.42,
        "precision_macro": 0.44,
        "recall_macro": 0.42,
    }


@pytest.fixture
def dummy_metrics():
    return [
        {"model": "DummyMostFrequent", "accuracy": 0.80,
         "balanced_accuracy": 0.50, "f1_macro": 0.44},
        {"model": "RandomForest",      "accuracy": 0.87,
         "balanced_accuracy": 0.82, "f1_macro": 0.81},
        {"model": "LogisticRegression","accuracy": 0.55,
         "balanced_accuracy": 0.53, "f1_macro": 0.42},
    ]
