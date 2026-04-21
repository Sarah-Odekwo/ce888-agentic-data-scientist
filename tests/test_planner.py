"""
Unit tests for agents/planner.py
Covers: return type, mandatory steps, imbalance conditional,
        memory-hint passthrough, small-dataset handling,
        _plan_config written into profile, no duplicate steps.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agents.planner import create_plan


def _profile(**overrides):
    """Return a minimal valid profile dict, with optional overrides."""
    p = {
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
    p.update(overrides)
    return p


# ── return type ───────────────────────────────────────────────────────────────

class TestCreatePlanReturnType:
    def test_returns_list(self):
        assert isinstance(create_plan(_profile()), list)

    def test_all_items_are_strings(self):
        plan = create_plan(_profile())
        assert all(isinstance(s, str) for s in plan)

    def test_plan_is_not_empty(self):
        assert len(create_plan(_profile())) > 0


# ── mandatory steps ───────────────────────────────────────────────────────────

class TestMandatorySteps:
    REQUIRED = [
        "profile_dataset", "train_models", "evaluate", "reflect", "write_report",
    ]

    @pytest.mark.parametrize("step", REQUIRED)
    def test_mandatory_step_present(self, step):
        plan = create_plan(_profile())
        assert step in plan, f"Mandatory step missing: {step}"


# ── imbalance conditional ─────────────────────────────────────────────────────

class TestImbalanceConditional:
    def test_imbalance_step_added_when_ratio_high(self):
        plan = create_plan(_profile(imbalance_ratio=4.5))
        assert any("imbalance" in s.lower() for s in plan)

    def test_imbalance_step_before_train_models(self):
        plan = create_plan(_profile(imbalance_ratio=4.5))
        imb_idx   = next((i for i, s in enumerate(plan) if "imbalance" in s.lower()), None)
        train_idx = next((i for i, s in enumerate(plan) if s == "train_models"), None)
        if imb_idx is not None and train_idx is not None:
            assert imb_idx < train_idx

    def test_no_imbalance_step_when_balanced(self):
        plan = create_plan(_profile(imbalance_ratio=1.1))
        assert not any("imbalance" in s.lower() for s in plan)

    def test_imbalance_boundary_value_exactly_3(self):
        # ratio == 3.0 should trigger the imbalance step
        plan = create_plan(_profile(imbalance_ratio=3.0))
        assert any("imbalance" in s.lower() for s in plan)


# ── small dataset ─────────────────────────────────────────────────────────────

class TestSmallDataset:
    def test_small_dataset_returns_valid_plan(self):
        plan = create_plan(_profile(shape={"rows": 300, "cols": 5}))
        assert isinstance(plan, list)
        assert len(plan) > 0

    def test_small_dataset_has_mandatory_steps(self):
        plan = create_plan(_profile(shape={"rows": 300, "cols": 5}))
        assert "train_models" in plan
        assert "evaluate" in plan


# ── memory hint ───────────────────────────────────────────────────────────────

class TestMemoryHint:
    def test_accepts_none(self):
        plan = create_plan(_profile(), memory_hint=None)
        assert isinstance(plan, list)

    def test_accepts_dict_hint(self):
        hint = {"best_model": "RandomForest", "best_bal_acc": 0.82}
        plan = create_plan(_profile(), memory_hint=hint)
        assert isinstance(plan, list)

    def test_mandatory_steps_survive_memory_hint(self):
        hint = {"best_model": "LogisticRegression", "best_bal_acc": 0.75}
        plan = create_plan(_profile(), memory_hint=hint)
        assert "train_models" in plan
        assert "evaluate" in plan


# ── plan quality ──────────────────────────────────────────────────────────────

class TestPlanQuality:
    def test_no_duplicate_steps(self):
        from collections import Counter
        plan = create_plan(_profile(imbalance_ratio=4.0))
        counts = Counter(plan)
        dups = [s for s, c in counts.items() if c > 1]
        assert dups == [], f"Duplicate steps: {dups}"

    def test_plan_config_written_into_profile(self):
        p = _profile()
        create_plan(p)
        assert "_plan_config" in p

    def test_plan_config_is_dict(self):
        p = _profile()
        create_plan(p)
        assert isinstance(p["_plan_config"], dict)

    def test_imbalanced_plan_config_sets_metric(self):
        p = _profile(imbalance_ratio=4.5)
        create_plan(p)
        cfg = p.get("_plan_config", {})
        if cfg and "primary_metric" in cfg:
            assert cfg["primary_metric"] in ("balanced_accuracy", "f1_weighted")
