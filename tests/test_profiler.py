"""
Unit tests for tools/data_profiler.py
Covers: dataset_fingerprint, infer_target_column, is_classification_target,
        profile_dataset (shape, keys, missing, outliers, high-card,
        correlation, imbalance, notes).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np

from tools.data_profiler import (
    profile_dataset,
    dataset_fingerprint,
    infer_target_column,
    is_classification_target,
)


# ── dataset_fingerprint ──────────────────────────────────────────────────────

class TestDatasetFingerprint:
    def test_returns_fp_prefix(self, simple_df):
        fp = dataset_fingerprint(simple_df, "churn")
        assert fp.startswith("fp_")

    def test_deterministic(self, simple_df):
        assert dataset_fingerprint(simple_df, "churn") == dataset_fingerprint(simple_df, "churn")

    def test_different_target_different_fp(self, simple_df):
        assert dataset_fingerprint(simple_df, "churn") != dataset_fingerprint(simple_df, "age")

    def test_different_shape_different_fp(self, simple_df):
        assert dataset_fingerprint(simple_df, "churn") != dataset_fingerprint(simple_df.iloc[:50], "churn")

    def test_returns_string(self, simple_df):
        assert isinstance(dataset_fingerprint(simple_df, "churn"), str)


# ── infer_target_column ──────────────────────────────────────────────────────

class TestInferTargetColumn:
    def test_finds_column_named_target(self):
        df = pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]})
        assert infer_target_column(df) == "target"

    def test_finds_column_named_label(self):
        df = pd.DataFrame({"x": [1, 2, 3], "label": [0, 1, 0]})
        assert infer_target_column(df) == "label"

    def test_finds_column_named_y(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
        assert infer_target_column(df) == "y"

    def test_falls_back_to_last_low_cardinality_col(self):
        df = pd.DataFrame({"a": range(100), "b": [0, 1] * 50})
        result = infer_target_column(df)
        assert result == "b"

    def test_case_insensitive_match(self):
        df = pd.DataFrame({"Feature": [1, 2], "Target": [0, 1]})
        result = infer_target_column(df)
        assert result == "Target"


# ── is_classification_target ─────────────────────────────────────────────────

class TestIsClassificationTarget:
    def test_binary_int_is_classification(self):
        assert is_classification_target(pd.Series([0, 1, 0, 1])) is True

    def test_string_series_is_classification(self):
        assert is_classification_target(pd.Series(["cat", "dog", "cat"])) is True

    def test_high_cardinality_int_is_not_classification(self):
        assert is_classification_target(pd.Series(range(200))) is False

    def test_few_unique_ints_is_classification(self):
        assert is_classification_target(pd.Series([1, 2, 3, 1, 2, 3])) is True


# ── profile_dataset — basic ───────────────────────────────────────────────────

class TestProfileDatasetBasic:
    def test_raises_on_missing_target(self, simple_df):
        with pytest.raises(ValueError, match="Target column"):
            profile_dataset(simple_df, "nonexistent")

    def test_shape_is_correct(self, simple_df):
        p = profile_dataset(simple_df, "churn")
        assert p["shape"]["rows"] == 100
        assert p["shape"]["cols"] == 5

    def test_required_keys_present(self, simple_df):
        p = profile_dataset(simple_df, "churn")
        for key in ("shape", "feature_types", "missing_pct", "total_missing_pct",
                    "notes", "is_classification", "outlier_cols",
                    "high_card_cols", "corr_pairs", "target"):
            assert key in p, f"Missing key: {key}"

    def test_target_excluded_from_features(self, simple_df):
        p = profile_dataset(simple_df, "churn")
        all_feats = p["feature_types"]["numeric"] + p["feature_types"]["categorical"]
        assert "churn" not in all_feats

    def test_numeric_and_categorical_split(self, simple_df):
        p = profile_dataset(simple_df, "churn")
        assert "age" in p["feature_types"]["numeric"]
        assert "plan_type" in p["feature_types"]["categorical"]

    def test_is_classification_true_for_binary(self, simple_df):
        p = profile_dataset(simple_df, "churn")
        assert p["is_classification"] is True

    def test_notes_is_list(self, simple_df):
        p = profile_dataset(simple_df, "churn")
        assert isinstance(p["notes"], list)


# ── profile_dataset — missing values ─────────────────────────────────────────

class TestProfileDatasetMissing:
    def test_total_missing_pct_positive(self, missing_df):
        p = profile_dataset(missing_df, "y")
        assert p["total_missing_pct"] > 0

    def test_per_column_missing_pct_positive(self, missing_df):
        p = profile_dataset(missing_df, "y")
        assert p["missing_pct"]["a"] > 0
        assert p["missing_pct"]["b"] > 0

    def test_zero_missing_on_clean_data(self, simple_df):
        p = profile_dataset(simple_df, "churn")
        assert p["total_missing_pct"] == 0.0

    def test_high_missingness_note_added(self):
        rng = np.random.default_rng(99)
        n = 100
        df = pd.DataFrame({
            "a": [float("nan")] * 70 + rng.normal(0, 1, 30).tolist(),
            "b": [float("nan")] * 70 + rng.normal(0, 1, 30).tolist(),
            "y": rng.choice([0, 1], n),
        })
        p = profile_dataset(df, "y")
        notes_lower = " ".join(p["notes"]).lower()
        assert "missing" in notes_lower or "imputation" in notes_lower


# ── profile_dataset — outliers ────────────────────────────────────────────────

class TestProfileDatasetOutliers:
    def test_detects_outlier_column(self, outlier_df):
        p = profile_dataset(outlier_df, "target")
        assert "outlier_col" in p["outlier_cols"]

    def test_normal_column_not_flagged_as_outlier(self, outlier_df):
        p = profile_dataset(outlier_df, "target")
        assert "normal_col" not in p["outlier_cols"]

    def test_outlier_note_added(self, outlier_df):
        p = profile_dataset(outlier_df, "target")
        notes_lower = " ".join(p["notes"]).lower()
        assert "outlier" in notes_lower or "robust" in notes_lower

    def test_no_outlier_on_uniform_data(self, simple_df):
        p = profile_dataset(simple_df, "churn")
        # score is uniform(0,1) — should not trigger IQR outlier detection
        assert "score" not in p["outlier_cols"]


# ── profile_dataset — high cardinality ───────────────────────────────────────

class TestProfileDatasetHighCard:
    def test_detects_high_cardinality_col(self, high_card_df):
        p = profile_dataset(high_card_df, "target")
        assert "region" in p["high_card_cols"]

    def test_low_cardinality_col_not_flagged(self, simple_df):
        p = profile_dataset(simple_df, "churn")
        assert "plan_type" not in p["high_card_cols"]

    def test_high_card_note_added(self, high_card_df):
        p = profile_dataset(high_card_df, "target")
        notes_lower = " ".join(p["notes"]).lower()
        assert "cardinality" in notes_lower or "ordinal" in notes_lower


# ── profile_dataset — correlation ────────────────────────────────────────────

class TestProfileDatasetCorrelation:
    def test_detects_correlated_pair(self, correlated_df):
        p = profile_dataset(correlated_df, "target")
        assert len(p["corr_pairs"]) >= 1

    def test_correlated_pair_contains_correct_columns(self, correlated_df):
        p = profile_dataset(correlated_df, "target")
        pair_cols = {c for pair in p["corr_pairs"] for c in pair}
        assert "x1" in pair_cols or "x2" in pair_cols

    def test_no_false_positive_on_uncorrelated_data(self, simple_df):
        p = profile_dataset(simple_df, "churn")
        assert p["corr_pairs"] == []

    def test_corr_pairs_is_list(self, simple_df):
        p = profile_dataset(simple_df, "churn")
        assert isinstance(p["corr_pairs"], list)


# ── profile_dataset — class imbalance ────────────────────────────────────────

class TestProfileDatasetImbalance:
    def test_imbalance_ratio_computed(self, imbalanced_df):
        p = profile_dataset(imbalanced_df, "target")
        assert p["imbalance_ratio"] is not None
        assert p["imbalance_ratio"] >= 3.0

    def test_imbalance_note_added(self, imbalanced_df):
        p = profile_dataset(imbalanced_df, "target")
        notes_lower = " ".join(p["notes"]).lower()
        assert "imbalance" in notes_lower or "class_weight" in notes_lower

    def test_balanced_data_ratio_below_threshold(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "x": rng.normal(0, 1, 100),
            "y": ([0] * 50) + ([1] * 50),
        })
        p = profile_dataset(df, "y")
        # perfectly balanced → ratio == 1.0, should not be flagged
        assert p["imbalance_ratio"] is None or p["imbalance_ratio"] < 3.0

    def test_class_counts_keys_are_strings(self, imbalanced_df):
        p = profile_dataset(imbalanced_df, "target")
        if p["class_counts"]:
            for k in p["class_counts"]:
                assert isinstance(k, str)
