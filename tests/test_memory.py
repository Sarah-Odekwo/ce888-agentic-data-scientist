"""
Unit tests for agents/memory.py  (JSONMemory class)
Covers: init/load/save, upsert_dataset_record, log_run, get_summary,
        get_run_history, get_best_result, get_best_strategy,
        add_note, persistence across instances, corruption recovery.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agents.memory import JSONMemory


FP = "fp_test_123"


def _record(model="RandomForest", bal_acc=0.80):
    return {
        "target": "churn",
        "shape":  {"rows": 1000, "cols": 10},
        "best_model":   model,
        "best_metrics": {
            "model": model,
            "balanced_accuracy": bal_acc,
            "f1_macro": bal_acc - 0.05,
        },
    }


def _reflection(replan=False, n_issues=1):
    return {
        "status": "needs_attention" if n_issues else "ok",
        "issues": [f"issue_{i}" for i in range(n_issues)],
        "suggestions": [],
        "replan_recommended": replan,
        "bal_acc": 0.80,
    }


# ── init and empty state ──────────────────────────────────────────────────────

class TestMemoryInit:
    def test_creates_fresh_memory_if_no_file(self, memory_path):
        mem = JSONMemory(memory_path)
        assert mem.data["datasets"] == {}
        assert mem.data["run_history"] == []

    def test_file_does_not_exist_before_first_save(self, memory_path):
        JSONMemory(memory_path)      # no save yet
        assert not os.path.exists(memory_path)

    def test_save_creates_file(self, memory_path):
        mem = JSONMemory(memory_path)
        mem.save()
        assert os.path.exists(memory_path)

    def test_saved_file_is_valid_json(self, memory_path):
        mem = JSONMemory(memory_path)
        mem.save()
        with open(memory_path) as f:
            data = json.load(f)
        assert "datasets" in data
        assert "run_history" in data


# ── upsert_dataset_record ─────────────────────────────────────────────────────

class TestUpsertDatasetRecord:
    def test_record_stored_after_upsert(self, memory_path):
        mem = JSONMemory(memory_path)
        mem.upsert_dataset_record(FP, _record())
        assert mem.get_dataset_record(FP) is not None

    def test_run_count_increments(self, memory_path):
        mem = JSONMemory(memory_path)
        mem.upsert_dataset_record(FP, _record())
        mem.upsert_dataset_record(FP, _record())
        assert mem.get_dataset_record(FP)["run_count"] == 2

    def test_best_score_kept_when_new_run_is_worse(self, memory_path):
        mem = JSONMemory(memory_path)
        mem.upsert_dataset_record(FP, _record(bal_acc=0.85))
        mem.upsert_dataset_record(FP, _record(bal_acc=0.70))
        assert mem.get_dataset_record(FP)["best_bal_acc"] == pytest.approx(0.85)

    def test_best_score_updated_when_new_run_is_better(self, memory_path):
        mem = JSONMemory(memory_path)
        mem.upsert_dataset_record(FP, _record(bal_acc=0.70))
        mem.upsert_dataset_record(FP, _record(bal_acc=0.90))
        assert mem.get_dataset_record(FP)["best_bal_acc"] == pytest.approx(0.90)

    def test_best_model_updated_on_improvement(self, memory_path):
        mem = JSONMemory(memory_path)
        mem.upsert_dataset_record(FP, _record(model="LR",    bal_acc=0.70))
        mem.upsert_dataset_record(FP, _record(model="GBM",   bal_acc=0.90))
        assert mem.get_dataset_record(FP)["best_model"] == "GBM"

    def test_run_history_entry_added(self, memory_path):
        mem = JSONMemory(memory_path)
        mem.upsert_dataset_record(FP, _record())
        assert len(mem.data["run_history"]) == 1

    def test_run_history_has_correct_fingerprint(self, memory_path):
        mem = JSONMemory(memory_path)
        mem.upsert_dataset_record(FP, _record())
        assert mem.data["run_history"][0]["fingerprint"] == FP


# ── log_run ───────────────────────────────────────────────────────────────────

class TestLogRun:
    def test_log_run_enriches_run_history(self, memory_path):
        mem = JSONMemory(memory_path)
        mem.upsert_dataset_record(FP, _record())
        mem.log_run(FP, {
            "run_id": "run_001",
            "plan_config": {"scaler": "robust", "impute_strategy": "mean"},
            "reflection": _reflection(replan=True, n_issues=2),
            "replan_count": 1,
        })
        entry = mem.get_run_history(FP)[-1]
        assert entry.get("replan_count") == 1
        assert entry.get("n_issues") == 2
        assert entry.get("replan_triggered") is True

    def test_log_run_stores_strategies_used(self, memory_path):
        mem = JSONMemory(memory_path)
        mem.upsert_dataset_record(FP, _record())
        mem.log_run(FP, {
            "run_id": "r1",
            "plan_config": {"scaler": "robust"},
            "reflection": _reflection(),
            "replan_count": 0,
        })
        entry = mem.get_run_history(FP)[-1]
        assert "strategies_used" in entry

    def test_log_run_creates_entry_if_no_prior_upsert(self, memory_path):
        mem = JSONMemory(memory_path)
        # log_run without prior upsert — should not crash
        mem.log_run(FP, {
            "run_id": "r_orphan",
            "plan_config": {},
            "reflection": _reflection(),
            "replan_count": 0,
        })
        assert len(mem.get_run_history(FP)) >= 1


# ── get_summary ───────────────────────────────────────────────────────────────

class TestGetSummary:
    def test_returns_zero_run_count_for_unknown_fp(self, memory_path):
        mem = JSONMemory(memory_path)
        summary = mem.get_summary("fp_unknown")
        assert summary["run_count"] == 0

    def test_run_count_correct_after_two_runs(self, memory_path):
        mem = JSONMemory(memory_path)
        for _ in range(2):
            mem.upsert_dataset_record(FP, _record())
        summary = mem.get_summary(FP)
        assert summary["run_count"] == 2

    def test_best_model_in_summary(self, memory_path):
        mem = JSONMemory(memory_path)
        mem.upsert_dataset_record(FP, _record(model="GBM", bal_acc=0.88))
        summary = mem.get_summary(FP)
        assert summary["best_model"] == "GBM"

    def test_score_trend_single_run(self, memory_path):
        mem = JSONMemory(memory_path)
        mem.upsert_dataset_record(FP, _record())
        summary = mem.get_summary(FP)
        assert summary["score_trend"] == "single_run"

    def test_score_trend_improving(self, memory_path):
        mem = JSONMemory(memory_path)
        mem.upsert_dataset_record(FP, _record(bal_acc=0.60))
        mem.upsert_dataset_record(FP, _record(bal_acc=0.60))
        mem.upsert_dataset_record(FP, _record(bal_acc=0.85))
        mem.upsert_dataset_record(FP, _record(bal_acc=0.87))
        summary = mem.get_summary(FP)
        assert summary["score_trend"] in ("improving", "stable")

    def test_replan_rate_correct(self, memory_path):
        mem = JSONMemory(memory_path)
        mem.upsert_dataset_record(FP, _record())
        mem.log_run(FP, {"run_id": "r1", "plan_config": {},
                         "reflection": _reflection(replan=True), "replan_count": 1})
        mem.upsert_dataset_record(FP, _record())
        mem.log_run(FP, {"run_id": "r2", "plan_config": {},
                         "reflection": _reflection(replan=False), "replan_count": 0})
        summary = mem.get_summary(FP)
        assert 0.0 <= summary["replan_rate"] <= 1.0

    def test_summary_has_all_keys(self, memory_path):
        mem = JSONMemory(memory_path)
        mem.upsert_dataset_record(FP, _record())
        summary = mem.get_summary(FP)
        for k in ("run_count", "best_model", "best_bal_acc",
                  "latest_bal_acc", "score_trend", "replan_rate", "avg_n_issues"):
            assert k in summary, f"Missing summary key: {k}"


# ── persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_data_persists_across_instances(self, memory_path):
        mem1 = JSONMemory(memory_path)
        mem1.upsert_dataset_record(FP, _record(bal_acc=0.88))
        # create new instance pointing to same file
        mem2 = JSONMemory(memory_path)
        assert mem2.get_dataset_record(FP) is not None
        assert mem2.get_dataset_record(FP)["best_bal_acc"] == pytest.approx(0.88)

    def test_run_history_persists(self, memory_path):
        mem1 = JSONMemory(memory_path)
        mem1.upsert_dataset_record(FP, _record())
        mem2 = JSONMemory(memory_path)
        assert len(mem2.get_run_history(FP)) == 1

    def test_multiple_fingerprints_independent(self, memory_path):
        mem = JSONMemory(memory_path)
        fp2 = "fp_other_456"
        mem.upsert_dataset_record(FP,  _record(bal_acc=0.80))
        mem.upsert_dataset_record(fp2, _record(bal_acc=0.70))
        assert mem.get_dataset_record(FP)["best_bal_acc"]  == pytest.approx(0.80)
        assert mem.get_dataset_record(fp2)["best_bal_acc"] == pytest.approx(0.70)


# ── add_note ──────────────────────────────────────────────────────────────────

class TestAddNote:
    def test_note_added_to_list(self, memory_path):
        mem = JSONMemory(memory_path)
        mem.add_note("test note")
        assert any("test note" in n["msg"] for n in mem.data["notes"])

    def test_note_persists(self, memory_path):
        mem1 = JSONMemory(memory_path)
        mem1.add_note("persistent note")
        mem2 = JSONMemory(memory_path)
        assert any("persistent note" in n["msg"] for n in mem2.data["notes"])


# ── corruption recovery ───────────────────────────────────────────────────────

class TestCorruptionRecovery:
    def test_recovers_from_corrupt_json(self, memory_path):
        # Write garbage to the memory file
        with open(memory_path, "w") as f:
            f.write("{this is not valid json{{{{")
        mem = JSONMemory(memory_path)   # should not raise
        assert mem.data["datasets"] == {}
        assert os.path.exists(memory_path + ".bak")  # backup created

    def test_recovers_from_empty_file(self, memory_path):
        with open(memory_path, "w") as f:
            f.write("")
        mem = JSONMemory(memory_path)
        assert mem.data["datasets"] == {}
