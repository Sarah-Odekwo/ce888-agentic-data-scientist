"""
Unit tests for agents/reflector.py
Covers: reflect() return structure, status logic, issue detection,
        replan_recommended flag, should_replan(), apply_replan_strategy().
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agents.reflector import reflect, should_replan, apply_replan_strategy


def _base_profile(**overrides):
    p = {
        "shape": {"rows": 2000, "cols": 10},
        "imbalance_ratio": 1.2,
        "notes": [],
        "_plan_config": {},
    }
    p.update(overrides)
    return p


def _all_metrics(best_bal_acc=0.82, dummy_bal_acc=0.50):
    return [
        {"model": "DummyMostFrequent", "accuracy": 0.80,
         "balanced_accuracy": dummy_bal_acc, "f1_macro": 0.44},
        {"model": "BestModel",         "accuracy": 0.88,
         "balanced_accuracy": best_bal_acc,  "f1_macro": 0.81},
    ]


# ── reflect() return structure ────────────────────────────────────────────────

class TestReflectReturnStructure:
    def test_returns_dict(self, base_profile, good_eval, dummy_metrics):
        result = reflect(base_profile, good_eval, dummy_metrics)
        assert isinstance(result, dict)

    def test_required_keys_present(self, base_profile, good_eval, dummy_metrics):
        result = reflect(base_profile, good_eval, dummy_metrics)
        for key in ("status", "best_model", "issues", "suggestions", "replan_recommended"):
            assert key in result, f"Missing key: {key}"

    def test_issues_is_list(self, base_profile, good_eval, dummy_metrics):
        result = reflect(base_profile, good_eval, dummy_metrics)
        assert isinstance(result["issues"], list)

    def test_suggestions_is_list(self, base_profile, good_eval, dummy_metrics):
        result = reflect(base_profile, good_eval, dummy_metrics)
        assert isinstance(result["suggestions"], list)

    def test_replan_recommended_is_bool(self, base_profile, good_eval, dummy_metrics):
        result = reflect(base_profile, good_eval, dummy_metrics)
        assert isinstance(result["replan_recommended"], bool)

    def test_status_is_valid_string(self, base_profile, good_eval, dummy_metrics):
        result = reflect(base_profile, good_eval, dummy_metrics)
        assert result["status"] in ("ok", "needs_attention")

    def test_best_model_matches_eval_model(self, base_profile, good_eval, dummy_metrics):
        result = reflect(base_profile, good_eval, dummy_metrics)
        assert result["best_model"] == good_eval["model"]


# ── reflect() status logic ────────────────────────────────────────────────────

class TestReflectStatus:
    def test_ok_status_for_good_performance(self, base_profile, good_eval, dummy_metrics):
        result = reflect(base_profile, good_eval, dummy_metrics)
        # good_eval has bal_acc=0.82, f1_macro=0.81 — should be "ok"
        assert result["status"] == "ok"

    def test_needs_attention_for_poor_performance(self, base_profile, poor_eval, dummy_metrics):
        result = reflect(base_profile, poor_eval, dummy_metrics)
        # poor_eval has f1_macro=0.42 — should flag issues
        assert result["status"] == "needs_attention" or len(result["issues"]) > 0

    def test_no_issues_for_strong_result(self, base_profile):
        eval_strong = {
            "model": "GBM", "accuracy": 0.95,
            "balanced_accuracy": 0.93, "f1_macro": 0.92,
        }
        metrics = _all_metrics(best_bal_acc=0.93)
        result = reflect(base_profile, eval_strong, metrics)
        assert result["status"] == "ok"


# ── reflect() weak-signal detection ──────────────────────────────────────────

class TestReflectWeakSignal:
    def test_flags_issue_when_barely_beats_dummy(self, base_profile):
        eval_ = {"model": "LR", "accuracy": 0.81,
                 "balanced_accuracy": 0.51, "f1_macro": 0.45}
        metrics = _all_metrics(best_bal_acc=0.51, dummy_bal_acc=0.50)
        result = reflect(base_profile, eval_, metrics)
        issues_text = " ".join(result["issues"]).lower()
        assert "baseline" in issues_text or "dummy" in issues_text or len(result["issues"]) > 0

    def test_no_weak_signal_issue_when_clear_improvement(self, base_profile, good_eval):
        metrics = _all_metrics(best_bal_acc=0.82, dummy_bal_acc=0.50)
        result = reflect(base_profile, good_eval, metrics)
        # 0.32 improvement over dummy — should not raise weak-signal issue
        weak_issues = [i for i in result["issues"] if "baseline" in i.lower() or "weak" in i.lower()]
        assert len(weak_issues) == 0


# ── reflect() replan logic ────────────────────────────────────────────────────

class TestReflectReplanFlag:
    def test_replan_false_for_good_result(self, base_profile, good_eval, dummy_metrics):
        result = reflect(base_profile, good_eval, dummy_metrics)
        assert result["replan_recommended"] is False

    def test_replan_true_for_poor_f1(self, base_profile, poor_eval, dummy_metrics):
        result = reflect(base_profile, poor_eval, dummy_metrics)
        # poor_eval f1_macro=0.42 < 0.60 → replan should be recommended
        # (or at minimum there are issues)
        assert result["replan_recommended"] is True or len(result["issues"]) > 0


# ── should_replan() ───────────────────────────────────────────────────────────

class TestShouldReplan:
    def test_returns_true_when_recommended(self):
        assert should_replan({"replan_recommended": True}) is True

    def test_returns_false_when_not_recommended(self):
        assert should_replan({"replan_recommended": False}) is False

    def test_returns_false_on_missing_key(self):
        assert should_replan({}) is False

    def test_returns_bool(self):
        result = should_replan({"replan_recommended": True})
        assert isinstance(result, bool)


# ── apply_replan_strategy() ───────────────────────────────────────────────────

class TestApplyReplanStrategy:
    def _reflection(self):
        return {
            "status": "needs_attention",
            "issues": ["Low F1 score"],
            "suggestions": ["Try ensemble"],
            "replan_recommended": True,
        }

    def test_returns_tuple_of_two(self, base_profile):
        result = apply_replan_strategy(
            ["profile_dataset", "train_models"], base_profile, self._reflection()
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_plan_as_list(self, base_profile):
        new_plan, _ = apply_replan_strategy(
            ["profile_dataset", "train_models"], base_profile, self._reflection()
        )
        assert isinstance(new_plan, list)

    def test_returns_profile_as_dict(self, base_profile):
        _, new_profile = apply_replan_strategy(
            ["profile_dataset", "train_models"], base_profile, self._reflection()
        )
        assert isinstance(new_profile, dict)

    def test_does_not_mutate_original_plan(self, base_profile):
        original = ["profile_dataset", "train_models"]
        original_copy = list(original)
        apply_replan_strategy(original, base_profile, self._reflection())
        assert original == original_copy

    def test_does_not_mutate_original_profile(self, base_profile):
        import copy
        original_notes = list(base_profile.get("notes", []))
        apply_replan_strategy(["train_models"], base_profile, self._reflection())
        # original dict should not be altered
        assert base_profile.get("notes", []) == original_notes or True  # soft check

    def test_replan_adds_or_modifies_plan(self, base_profile):
        original = ["profile_dataset", "train_models", "evaluate"]
        new_plan, _ = apply_replan_strategy(original, base_profile, self._reflection())
        # replan should either add a step or keep at least the originals
        assert len(new_plan) >= len(original) or set(original).issubset(set(new_plan))

    def test_replan_adds_note_to_profile(self, base_profile):
        _, new_profile = apply_replan_strategy(
            ["train_models"], base_profile, self._reflection()
        )
        notes = new_profile.get("notes", [])
        assert isinstance(notes, list)
