import os
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

# helper functions

def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(max(4, len(labels)), max(4, len(labels))))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=11, pad=10)

    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9)

    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(int(cm[i, j]), "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )

    ax.set_ylabel("True label",      fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def evaluate_best(training_payload: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Compute evaluation artefacts for the best model.

    Dict with keys:
        best_metrics : metrics dict for the winning model
        all_metrics : list of metric dicts for all candidates
        confusion_matrix_path : path to the saved PNG
        classification_report : sklearn text report string
    """
    best       = training_payload["best"]
    all_metrics= training_payload["all_metrics"]

    y_test = best["y_test"]
    y_pred = best["y_pred"]

    cm = confusion_matrix(y_test, y_pred)
    labels = sorted([str(x) for x in y_test.dropna().unique().tolist()])
    cm_path= os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, labels, cm_path, f"Confusion Matrix: {best['name']}")

    cls_report = classification_report(y_test, y_pred, zero_division=0)

    return {
        "best_metrics": best["metrics"],
        "all_metrics": all_metrics,
        "confusion_matrix_path": cm_path,
        "classification_report": cls_report,
    }



def write_markdown_report(
    out_path: str,
    ctx: Any,
    fingerprint: str,
    dataset_profile: Dict[str, Any],
    plan: List[str],
    eval_payload: Dict[str, Any],
    reflection: Dict[str, Any],
    memory_summary:  Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write a Markdown report to out_path.

    Parameters:
        memory_summary : optional dict from memory.get_summary(fingerprint).
            When provided, a Memory History section is added to the report
            showing run_count, score_trend, replan_rate, avg_n_issues.
            Pass as: memory_summary=self.memory.get_summary(fp)
    """
    best = eval_payload["best_metrics"]

    def short_list(xs: List[str], n: int = 12) -> str:
        return ", ".join(xs[:n]) + (" …" if len(xs) > n else "")

    numeric = dataset_profile.get("feature_types", {}).get("numeric",     [])
    categorical = dataset_profile.get("feature_types", {}).get("categorical", [])
    notes = dataset_profile.get("notes", [])
    cfg = dataset_profile.get("_plan_config", {})

    # section helper functions

    def _plan_config_section(cfg: Dict[str, Any]) -> str:
        if not cfg:
            return "_No _plan_config found - planner may not have run._\n"
        rows_md = []
        label_map = {
            "impute_strategy": "Imputer",
            "scaler": "Scaler",
            "handle_imbalance": "Imbalance handling",
            "high_card_cols": "High-card cols → OrdinalEncoder",
            "drop_corr_cols": "Correlated cols dropped",
            "feature_selection": "Feature selection",
            "k_best": "SelectKBest k",
            "primary_metric": "Primary metric",
            "priority_model": "Priority model (from memory)",
            "extra_models": "Extra models added (replan)",
            "prefer_simple_models": "Prefer simple models",
        }
        for key, label in label_map.items():
            val = cfg.get(key)
            if val is not None and val != [] and val is not False:
                rows_md.append(f"| {label} | `{val}` |")
        if not rows_md:
            return "_All defaults used._\n"
        header = "| Decision | Value |\n|---|---|\n"
        return header + "\n".join(rows_md) + "\n"

    def _reflection_issues_section(reflection: Dict[str, Any]) -> str:
        issues = reflection.get("issues", [])
        suggestions = reflection.get("suggestions", [])
        severity = reflection.get("severity_counts", {})
        spread = reflection.get("score_spread")
        baseline = reflection.get("baseline_improvement")
        replan = reflection.get("replan_recommended", False)

        lines = []

        # severity summary bar
        h = severity.get("high", 0)
        m = severity.get("medium", 0)
        l = severity.get("low", 0)
        lines.append(
            f"**Severity counts:** HIGH={h}  MEDIUM={m}  LOW={l}"
        )
        if spread is not None:
            lines.append(f"**Score spread across candidates:** {spread:.3f}")
        if baseline is not None:
            lines.append(
                f"**Improvement over dummy baseline:** {baseline:.3f}"
            )
        lines.append(
            f"**Replan triggered:** {'Yes' if replan else 'No'}"
        )
        lines.append("")

        if issues:
            lines.append("### Issues found")
            for iss in issues:
                lines.append(f"- {iss}")
            lines.append("")

        if suggestions:
            lines.append("### Suggestions")
            for sug in suggestions:
                lines.append(f"- {sug}")

        return "\n".join(lines) + "\n"

    def _memory_section(ms: Dict[str, Any]) -> str:
        if not ms or ms.get("run_count", 0) == 0:
            return "_No prior runs recorded for this dataset._\n"
        trend_emoji = {
            "improving": "📈", "stable": "➡️",
            "degrading": "📉", "single_run": "🔵",
        }.get(ms.get("score_trend", ""), "")
        return (
            f"| Metric | Value |\n|---|---|\n"
            f"| Total runs | `{ms['run_count']}` |\n"
            f"| Best model ever | `{ms.get('best_model', 'n/a')}` |\n"
            f"| Best balanced_accuracy ever | `{ms.get('best_bal_acc', 0):.3f}` |\n"
            f"| Score trend | {trend_emoji} `{ms.get('score_trend', 'n/a')}` |\n"
            f"| Replan rate | `{ms.get('replan_rate', 0):.0%}` of runs |\n"
            f"| Avg issues per run | `{ms.get('avg_n_issues', 0)}` |\n"
        )

    # assemble markdown
    md = f"""# Agentic Data Scientist - Run Report

**Run ID:** `{ctx.run_id}`
**Started (UTC):** {ctx.started_at}
**Dataset:** `{ctx.data_path}`
**Target:** `{ctx.target}`
**Fingerprint:** `{fingerprint}`

---

## Dataset Profile

| Property | Value |
|---|---|
| Rows | **{dataset_profile["shape"]["rows"]}** |
| Columns | **{dataset_profile["shape"]["cols"]}** |
| Task | **{"Classification" if dataset_profile.get("is_classification") else "Regression"}** |
| Imbalance ratio | **{dataset_profile.get("imbalance_ratio", "n/a")}** |
| Total missing % | **{dataset_profile.get("total_missing_pct", 0):.1f}%** |
| Outlier columns | `{dataset_profile.get("outlier_cols", [])}` |
| High-card columns | `{dataset_profile.get("high_card_cols", [])}` |
| Correlated pairs | `{dataset_profile.get("corr_pairs", [])}` |

**Numeric features ({len(numeric)}):** {short_list(numeric)}
**Categorical features ({len(categorical)}):** {short_list(categorical)}

**Profiler notes**
{chr(10).join([f"- {n}" for n in notes]) if notes else "- (none)"}

---

## Planning Decisions

{_plan_config_section(cfg)}

---

## Execution Plan

{chr(10).join([f"- {step}" for step in plan])}

---

## Results — Best Model

**Model:** `{best.get("model")}`

| Metric | Score |
|---|---|
| Accuracy | **{best.get("accuracy", 0):.3f}** |
| Balanced Accuracy | **{best.get("balanced_accuracy", 0):.3f}** |
| Macro F1 | **{best.get("f1_macro", 0):.3f}** |
| Macro Precision | **{best.get("precision_macro", 0):.3f}** |
| Macro Recall | **{best.get("recall_macro", 0):.3f}** |

### All Candidates

```json
{json.dumps(eval_payload.get("all_metrics", []), indent=2)}
```

### Classification Report

```
{eval_payload.get("classification_report", "(not available)")}
```

---

## Reflection & Diagnosis

{_reflection_issues_section(reflection)}

---

## Memory History

{_memory_section(memory_summary) if memory_summary is not None else "_memory_summary not passed to this report — add memory_summary=self.memory.get_summary(fp) in agentic_data_scientist.py._"}

---

## Artefacts

- Confusion matrix: `{eval_payload.get("confusion_matrix_path", "n/a")}`
- Report: `{out_path}`
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

