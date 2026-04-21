import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from tools.evaluation import evaluate_best


def _make_training_payload(y_test, y_pred, model_name="TestModel"):
    metrics = {
        "model": model_name,
        "balanced_accuracy": float(
            sum(a == b for a, b in zip(y_test, y_pred)) / len(y_test)
        ),
        "accuracy": float(sum(a == b for a, b in zip(y_test, y_pred)) / len(y_test)),
        "f1_macro": 0.85,
        "precision_macro": 0.85,
        "recall_macro": 0.85,
    }
    return {
        "best": {
            "name": model_name,
            "metrics": metrics,
            "y_test": pd.Series(y_test),
            "y_pred": np.array(y_pred),
            "X_test": pd.DataFrame(),
        },
        "all_metrics": [metrics],
        "results": [],
    }


def test_compute_metrics_basic(tmp_path):
    """evaluate_best returns a dict with the expected top-level keys."""
    y_test = [0, 1, 0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 1, 0, 1, 0, 1]
    payload = _make_training_payload(y_test, y_pred)

    result = evaluate_best(payload, str(tmp_path))

    assert "best_metrics" in result
    assert "all_metrics" in result
    assert "confusion_matrix_path" in result
    assert "classification_report" in result


def test_evaluate_best_confusion_matrix_written(tmp_path):
    y_test = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 0, 0, 1, 0]
    payload = _make_training_payload(y_test, y_pred)

    result = evaluate_best(payload, str(tmp_path))

    assert os.path.exists(result["confusion_matrix_path"])


def test_evaluate_best_best_metrics_structure(tmp_path):
    y_test = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    payload = _make_training_payload(y_test, y_pred)

    result = evaluate_best(payload, str(tmp_path))
    bm = result["best_metrics"]

    assert "model" in bm
    assert "balanced_accuracy" in bm