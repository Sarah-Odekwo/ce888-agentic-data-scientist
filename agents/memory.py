import json
import os
import shutil
from typing import Any, Dict, List, Optional
from datetime import datetime


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class JSONMemory:
    """
    Persistent JSON memory for the agent.

    Storage layout:
    {
      "datasets": {
        "<fingerprint>": {
          "best_model": str,
          "best_bal_acc": float,
          "best_metrics": dict,
          "best_plan_config": dict | null,
          "run_count": int,
          "first_seen": ISO str,
          "last_seen": ISO str,
          "target": str,
          "shape": {rows, cols}
        }
      },
      "run_history": [
        {
          "fingerprint": str,
          "ts": ISO str,
          "run_id": str | null,
          "best_model": str,
          "bal_acc": float,
          "f1_macro": float,
          "replan_count": int,
          "n_issues": int,
          "replan_triggered": bool,
          "strategies_used": dict   (summary of _plan_config)
        },
        ...
      ],
      "notes": [ {"ts": str, "msg": str}, ... ]
    }

    """

    def __init__(self, path: str = "agent_memory.json"):
        self.path = path
        self.data: Dict[str, Any] = {"datasets": {}, "run_history": [], "notes": []}
        self._load()

    # helper functions
    def _load(self) -> None:
        """Load memory from disk; restore defaults on corruption."""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            # merge: this ensures that the new top-level keys exist in older files
            self.data["datasets"] = loaded.get("datasets", {})
            self.data["run_history"] = loaded.get("run_history", [])
            self.data["notes"] = loaded.get("notes", [])
        except Exception:
            backup = self.path + ".bak"
            shutil.copy(self.path, backup)
            self.data = {
                "datasets":    {},
                "run_history": [],
                "notes": [{
                    "ts":  now_iso(),
                    "msg": f"Memory reset after corrupt read; backup at {backup}",
                }],
            }

    def save(self) -> None:
        """Write memory to disk atomically (write-then-rename)."""
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    def _best_bal_acc(self, fingerprint: str) -> float:
        """Return the best balanced_accuracy ever seen for a fingerprint."""
        rec = self.data["datasets"].get(fingerprint, {})
        return float(rec.get("best_bal_acc", 0.0))

    def get_dataset_record(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """
        Return the best-ever record for this dataset fingerprint.
        """
        return self.data.get("datasets", {}).get(fingerprint)

    def upsert_dataset_record(self, fingerprint: str, record: Dict[str, Any]) -> None:
        """
        Update the dataset record, keeping best-ever metrics.

        Accepts the same dict that agentic_data_scientist.py already passes:
            {last_seen, target, shape, best_model, best_metrics}
        """
        datasets = self.data.setdefault("datasets", {})
        existing = datasets.get(fingerprint, {})
        new_bal = float(
            record.get("best_metrics", {}).get("balanced_accuracy", 0.0)
        )
        is_better = new_bal > self._best_bal_acc(fingerprint)

        updated = {
            # static fields
            "target": record.get("target", existing.get("target")),
            "shape": record.get("shape", existing.get("shape")),
            "first_seen": existing.get("first_seen", now_iso()),
            "last_seen": now_iso(),
            "run_count": existing.get("run_count", 0) + 1,

            # best-ever fields - only get overwritten when the new run is better
            "best_model": (
                record.get("best_model")
                if is_better
                else existing.get("best_model")
            ),
            "best_bal_acc": (
                new_bal
                if is_better
                else existing.get("best_bal_acc", 0.0)
            ),
            "best_metrics": (
                record.get("best_metrics")
                if is_better
                else existing.get("best_metrics")
            ),
            # best_plan_config populated by log_run(); preserved here
            "best_plan_config": existing.get("best_plan_config"),
        }
        datasets[fingerprint] = updated

        # lightweight run_history entry
        self.data.setdefault("run_history", []).append({
            "fingerprint": fingerprint,
            "ts": now_iso(),
            "run_id": record.get("run_id"),
            "best_model": record.get("best_model"),
            "bal_acc": new_bal,
            "f1_macro": float(
                record.get("best_metrics", {}).get("f1_macro", 0.0)
            ),
            "is_best_ever": is_better,
        })

        self.save()

    def add_note(self, msg: str) -> None:
        """Append a free-text note to the global notes list."""
        self.data.setdefault("notes", []).append(
            {"ts": now_iso(), "msg": msg}
        )
        self.save()

    def log_run(self, fingerprint: str, extended: Dict[str, Any]) -> None:
        """
        Store rich per-run detail: _plan_config, reflection, replan_count.

        This enriches the most-recent run_history entry in-place so no duplicate entry is created.
        """
        run_id = extended.get("run_id")
        plan_config = extended.get("plan_config", {})
        reflection = extended.get("reflection", {})
        replan_count = int(extended.get("replan_count", 0))

        # find the most recent run_history entry for this fingerprint
        history = self.data.get("run_history", [])
        target_entry = None
        for entry in reversed(history):
            if entry.get("fingerprint") == fingerprint:
                target_entry = entry
                break

        if target_entry is not None:
            
            target_entry["run_id"] = run_id or target_entry.get("run_id")
            target_entry["replan_count"] = replan_count
            target_entry["n_issues"] = len(reflection.get("issues", []))
            target_entry["replan_triggered"]= bool(
                reflection.get("replan_recommended", False)
            )
            target_entry["strategies_used"] = _summarise_config(plan_config)
        else:
            # no matching entry yet, so a new one gets appended 
            history.append({
                "fingerprint": fingerprint,
                "ts": now_iso(),
                "run_id": run_id,
                "replan_count": replan_count,
                "n_issues": len(reflection.get("issues", [])),
                "replan_triggered":bool(reflection.get("replan_recommended", False)),
                "strategies_used": _summarise_config(plan_config),
            })
        
        datasets = self.data.get("datasets", {})
        rec = datasets.get(fingerprint, {})
        bal_acc = float(
            reflection.get("bal_acc", 0.0) or
            (target_entry or {}).get("bal_acc", 0.0)
        )
        if bal_acc >= float(rec.get("best_bal_acc", 0.0)):
            rec["best_plan_config"] = plan_config
            datasets[fingerprint]   = rec

        self.save()

    def get_run_history(self, fingerprint: str) -> List[Dict[str, Any]]:
        """
        Return all run records for a fingerprint, oldest first.
        """
        return [
            r for r in self.data.get("run_history", [])
            if r.get("fingerprint") == fingerprint
        ]

    def get_best_result(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """
        Return the run_history entry that achieved the highest bal_acc for this fingerprint.
        """
        runs = self.get_run_history(fingerprint)
        if not runs:
            return None
        return max(runs, key=lambda r: float(r.get("bal_acc", 0.0)))
        
    def get_best_strategy(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """
        Return the _plan_config (summarised) from the best-ever run.

        This is used by the planner as a richer memory hint than just the model name.
        """
        rec = self.data.get("datasets", {}).get(fingerprint, {})
        return rec.get("best_plan_config")

    def get_summary(self, fingerprint: str) -> Dict[str, Any]:
        """
        Return summary statistics for a fingerprint.

        Keys returned:
            run_count : total runs attempted
            best_model : model name with highest bal_acc
            best_bal_acc : float
            latest_bal_acc : float (most recent run)
            score_trend : "improving" | "stable" | "degrading" | "single_run"
            replan_rate : fraction of runs that triggered a replan
            avg_n_issues : average number of issues found per run
        """
        runs = self.get_run_history(fingerprint)
        rec  = self.data.get("datasets", {}).get(fingerprint, {})

        if not runs:
            return {
                "run_count": 0,
                "best_model": rec.get("best_model"),
                "best_bal_acc": rec.get("best_bal_acc", 0.0),
                "latest_bal_acc": 0.0,
                "score_trend": "single_run",
                "replan_rate": 0.0,
                "avg_n_issues": 0.0,
            }

        scores = [float(r.get("bal_acc", 0.0)) for r in runs]
        trend = _compute_trend(scores)
        replans = sum(1 for r in runs if r.get("replan_triggered", False))
        issues = [int(r.get("n_issues", 0)) for r in runs]

        return {
            "run_count": len(runs),
            "best_model":  rec.get("best_model"),
            "best_bal_acc": rec.get("best_bal_acc", 0.0),
            "latest_bal_acc": scores[-1],
            "score_trend": trend,
            "replan_rate": round(replans / len(runs), 2),
            "avg_n_issues": round(sum(issues) / max(len(issues), 1), 1),
        }

def all_fingerprints(self) -> List[str]:
    """Return all dataset fingerprints stored in memory."""
    return list(self.data.get("datasets", {}).keys())

def total_runs(self) -> int:
    """Return total number of runs stored across all datasets."""
    return len(self.data.get("run_history", []))

def _summarise_config(plan_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the key decision fields from _plan_config for compact storage. Avoids bloating run_history.
    """
    if not plan_config:
        return {}
    return {
        k: plan_config[k]
        for k in (
            "scaler", "impute_strategy", "handle_imbalance",
            "feature_selection", "high_card_cols", "drop_corr_cols",
            "priority_model", "extra_models", "primary_metric",
        )
        if k in plan_config
    }

def _compute_trend(scores: List[float]) -> str:
    """
    Classify the score trend across runs as improving / stable / degrading.
    """
    if len(scores) < 2:
        return "single_run"
    mid = len(scores) // 2
    first = sum(scores[:mid]) / max(mid, 1)
    second= sum(scores[mid:]) / max(len(scores) - mid, 1)
    delta = second - first
    if delta >  0.02:
        return "improving"
    if delta < -0.02:
        return "degrading"
    return "stable"
