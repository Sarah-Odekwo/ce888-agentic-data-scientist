import os
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

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

    ax.set_ylabel("True label", fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    
# significance test
def significance_test(
    results: List[Dict[str, Any]],
    X_data:pd.DataFrame,
    y_data: pd.Series,
    seed: int = 42,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """
    Paired t-test comparing the best model vs every other candidate
    across n_splits cross-validation folds.
    """
    non_dummy = [r for r in results if "Dummy" not in r["name"]]
    if len(non_dummy) < 2:
        return {"skipped": True, "reason": "Fewer than 2 non-dummy models available."}

    best = non_dummy[0]
    best_name = best["name"]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # collect best model fold scores once
    best_scores: List[float] = []
    for tr_idx, te_idx in skf.split(X_data, y_data):
        X_tr, X_te = X_data.iloc[tr_idx], X_data.iloc[te_idx]
        y_tr, y_te = y_data.iloc[tr_idx], y_data.iloc[te_idx]
        try:
            best["pipeline"].fit(X_tr, y_tr)
            best_scores.append(balanced_accuracy_score(y_te, best["pipeline"].predict(X_te)))
        except Exception:
            best_scores.append(0)

    # compare each other model against the best
    comparisons: Dict[str, Any] = {}
    for r in non_dummy[1:]:
        other_scores: List[float] = []
        for tr_idx, te_idx in skf.split(X_data, y_data):
            X_tr, X_te = X_data.iloc[tr_idx], X_data.iloc[te_idx]
            y_tr, y_te = y_data.iloc[tr_idx], y_data.iloc[te_idx]
            try:
                r["pipeline"].fit(X_tr, y_tr)
                other_scores.append(float(balanced_accuracy_score(y_te, r["pipeline"].predict(X_te))))
            except Exception:
                other_scores.append(0)
            

        t_stat, p_value = ttest_rel(best_scores, other_scores)
        significant = bool(p_value < 0.05)

        comparisons[r["name"]] = {
            "t_stat": round(float(t_stat),  4),
            "p_value": round(float(p_value), 4),
            "significant": significant,
            "interpretation": (
                f"{best_name} is significantly better than {r['name']} (p={p_value:.3f}, t={t_stat:.3f})"
                if significant
                else
                f"No significant difference between {best_name} and {r['name']} (p={p_value:.3f}) — simpler model may be enough"
            ),
        }

    return {
        "best_model": best_name,
        "alpha": 0.05,
        "n_folds": n_splits,
        "comparisons": comparisons,
    }

def evaluate_best(training_payload: Dict[str, Any], output_dir: str, seed: int=42) -> Dict[str, Any]:
    """
    Compute evaluation artefacts for the best model.

    Dict with keys:
        best_metrics : metrics dict for the winning model
        all_metrics : list of metric dicts for all candidates
        confusion_matrix_path : path to the saved PNG
        classification_report : sklearn text report string
        significance_test: dict
    """
    best = training_payload["best"]
    all_metrics= training_payload["all_metrics"]
    results = training_payload.get("results", [])

    y_test = best["y_test"]
    y_pred = best["y_pred"]

    cm = confusion_matrix(y_test, y_pred)
    labels = sorted([str(x) for x in y_test.dropna().unique().tolist()])
    cm_path= os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, labels, cm_path, f"Confusion Matrix: {best['name']}")
    
    cls_report = classification_report(y_test, y_pred, zero_division=0)

    X_test = best.get("X_test", pd.DataFrame())
    if not X_test.empty and len(results) >= 2:
        min_class_count = int(y_test.value_counts().min())
        safe_splits = min(5, len(y_test), min_class_count)

        if safe_splits < 2 or len(results) < 2:
            sig = {"skipped": True, "reason": f"Too few samples per class for cross-validation (min class size={min_class_count})."}
        else:
            sig = significance_test(
                results=results,
                X_data=X_test,
                y_data=y_test,
                seed=seed,
                n_splits=safe_splits,
    )

    return {
        "best_metrics": best["metrics"],
        "all_metrics": all_metrics,
        "confusion_matrix_path": cm_path,
        "classification_report": cls_report,
        "significance_test": sig,
    }


def write_markdown_report(
    out_path: str,
    ctx: Any,
    fingerprint: str,
    dataset_profile: Dict[str, Any],
    plan: List[str],
    eval_payload: Dict[str, Any],
    reflection: Dict[str, Any],
    memory_summary: Optional[Dict[str, Any]] = None,
    plan_config: Optional[Dict[str, Any]] = None,
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
    all_metrics = eval_payload.get("all_metrics", [])

    def short_list(xs: List[str], n: int = 12) -> str:
        return ", ".join(xs[:n]) + (" …" if len(xs) > n else "")

    numeric = dataset_profile.get("feature_types", {}).get("numeric",     [])
    categorical = dataset_profile.get("feature_types", {}).get("categorical", [])
    notes = dataset_profile.get("notes", [])

    # planning decisions table
    plan_config_section = ""
    if plan_config:
        rows = []
        field_labels = [
            ("impute_strategy",    "Imputer"),
            ("scaler",             "Scaler"),
            ("handle_imbalance",   "Imbalance handling"),
            ("high_card_cols",     "High-card cols → OrdinalEncoder"),
            ("drop_corr_cols",     "Correlated cols dropped"),
            ("feature_selection",  "Feature selection"),
            ("select_k",           "SelectKBest k"),
            ("primary_metric",     "Primary metric"),
        ]
        for key, label in field_labels:
            val = plan_config.get(key)
            if val is not None:
                rows.append(f"| {label} | `{val}` |")
        if rows:
            plan_config_section = (
                "## Planning Decisions\n\n"
                "| Decision | Value |\n"
                "|---|---|\n"
                + "\n".join(rows)
                + "\n\n\n---\n"
            )

    # significance test 
    sig      = eval_payload.get("significance_test", {})
    sig_body = ""

    if sig.get("skipped"):
        sig_body = f"_Skipped: {sig.get('reason', 'n/a')}_"
    elif sig:
        best_model = sig.get("best_model", "—")
        n_folds    = sig.get("n_folds", 5)
        alpha      = sig.get("alpha", 0.05)
        header     = (
            f"**Best model:** `{best_model}` | "
            f"**α = {alpha}** | **{n_folds}-fold stratified CV**\n\n"
            "| Comparison Model | t-stat | p-value | Significant? | Interpretation |\n"
            "|---|---|---|---|---|\n"
        )
        table_rows = []
        for model_name, res in sig.get("comparisons", {}).items():
            marker = "Yes" if res["significant"] else "No"
            interp = res.get("interpretation", "")
            table_rows.append(
                f"| {model_name} | {res['t_stat']:+.3f} | "
                f"{res['p_value']:.3f} | {marker} | {interp} |"
            )
        sig_body = header + "\n".join(table_rows) if table_rows else "_No comparisons available._"

    # reflection 
    if reflection:
        sev = reflection.get("severity_counts", {})
        high_n = sev.get("HIGH", 0)
        med_n = sev.get("MEDIUM", 0)
        low_n = sev.get("LOW", 0)
        spread = reflection.get("score_spread", 0.0)
        improvement = reflection.get("improvement_over_dummy", 0.0)
        replan_flag = reflection.get("replan_recommended", False)
        issues = reflection.get("issues", [])
        suggestions = reflection.get("suggestions", [])

        issues_md = "\n".join([f"- {i}" for i in issues]) if issues else "- (none)"
        sugg_md = "\n".join([f"- {s}" for s in suggestions]) if suggestions else "- (none)"

        reflection_section = (
            "## Reflection & Diagnosis\n\n"
            f"**Severity counts:** HIGH={high_n}  MEDIUM={med_n}  LOW={low_n}\n"
            f"**Score spread across candidates:** {spread:.3f}\n"
            f"**Improvement over dummy baseline:** {improvement:.3f}\n"
            f"**Replan triggered:** {'Yes' if replan_flag else 'No'}\n\n"
            "### Issues found\n"
            f"{issues_md}\n\n"
            "### Suggestions\n"
            f"{sugg_md}\n"
        )
    else:
        reflection_section = "## Reflection & Diagnosis\n\n- (none)\n"

    # memory history
    if memory_summary and memory_summary.get("run_count", 0) > 0:
        mem_section = (
            "## Memory History\n\n"
            f"| Property | Value |\n"
            f"|---|---|\n"
            f"| Total runs on this dataset | **{memory_summary['run_count']}** |\n"
            f"| Best model ever | **{memory_summary.get('best_model', '—')}** |\n"
            f"| Best balanced accuracy | **{float(memory_summary.get('best_bal_acc', 0)):.3f}** |\n"
            f"| Latest balanced accuracy | **{float(memory_summary.get('latest_bal_acc', 0)):.3f}** |\n"
            f"| Score trend | **{memory_summary.get('score_trend', '—')}** |\n"
            f"| Replan rate | **{memory_summary.get('replan_rate', 0):.0%}** |\n"
            f"| Avg issues per run | **{memory_summary.get('avg_n_issues', 0):.1f}** |\n"
        )
    else:
        mem_section = "## Memory History\n\n_No prior runs recorded for this dataset._\n"


    all_cand_block = json.dumps(all_metrics, indent=2)

   
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
| Imbalance ratio | **{dataset_profile.get("imbalance_ratio", "N/A")}** |
| Total missing % | **{dataset_profile.get("total_missing_pct", 0):.1f}%** |
| Outlier columns | `{dataset_profile.get("outlier_cols", [])}` |
| High-card columns | `{dataset_profile.get("high_card_cols", [])}` |
| Correlated pairs | `{dataset_profile.get("corr_pairs", [])}` |

**Numeric features ({len(numeric)}):** {short_list(numeric)}
**Categorical features ({len(categorical)}):** {short_list(categorical)}

**Profiler notes**
{chr(10).join([f"- {n}" for n in notes]) if notes else "- (none)"}

---

{plan_config_section}

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
{all_cand_block}
```

### Classification Report

```
{eval_payload.get("classification_report", "(not available)")}
```
---

## Statistical Significance Test

{sig_body}

---

{reflection_section}

---

{mem_section}

---

## Artefacts

- Confusion matrix: `{eval_payload.get("confusion_matrix_path")}`
- Report: `{out_path}`
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
