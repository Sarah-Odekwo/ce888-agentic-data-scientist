"""
Reflector Agent - Students must extend this significantly

The reflector evaluates execution results, identifies issues, and suggests improvements.
Your task is to implement sophisticated analysis that goes beyond simple threshold checks.

TODO: Extend this module with:
1. Statistical significance testing between models
2. Per-class performance analysis
3. Root cause diagnosis (data quality, preprocessing, model issues)
4. Actionable, prioritized suggestions
5. Learning from past reflections (meta-learning)
"""

from typing import Any, Dict, List, Tuple, Optional

# performance floor and threshold values
PERF_FLOOR = 0.55   # balanced_accuracy below this is a hard failure
SPREAD_WARN = 0.10   # any score spread above this is a medium issue
SPREAD_HIGH = 0.20   # score spread above this is a high issue
BASELINE_MIN_DELTA = 0.05   # improvement over dummy below this is considered a high issue
F1_WARN = 0.55   # macro-F1 below this triggers a suggestion
OVERFIT_GAP = 0.15   # train/test gap above this flags overfitting

def reflect(
    dataset_profile: Dict[str, Any],
    evaluation: Dict[str, Any],
    all_metrics: List[Dict[str, Any]],
    train_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Analyse results and return a structured reflection.

    Args:
        dataset_profile: Output of profile_dataset().
        evaluation: Best model metrics dict.
        all_metrics: Metrics list for all trained candidates.
        train_metrics: Optional train-set metrics for the best model used for overfitting detection.
                         

    Returns dict with keys:
        status, best_model, bal_acc, issues, suggestions, severity_counts, score_spread, baseline_improvement, replan_recommended, replan_tier
    """

    best_model = evaluation.get("model", "unknown")
    bal_acc = float(evaluation.get("balanced_accuracy", 0.0))
    f1_macro = float(evaluation.get("f1_macro", 0.0))
    imb = float(dataset_profile.get("imbalance_ratio") or 1.0)
    plan_cfg = dataset_profile.get("_plan_config", {})

    issues: List[str] = []
    suggestions: List[str] = []
    severity_counts = {"high": 0, "medium": 0, "low": 0}

    def add_issue(level: str, msg: str) -> None:
        issues.append(f"[{level.upper()}] {msg}")
        severity_counts[level] += 1

    # baseline comparison
    dummy = next((m for m in all_metrics if "Dummy" in m.get("model", "")), None)
    baseline_improvement = 0.0
    if dummy is not None:
        dummy_ba = float(dummy.get("balanced_accuracy", 0.5))
        baseline_improvement = bal_acc - dummy_ba
        if baseline_improvement < BASELINE_MIN_DELTA:
            add_issue(
                "high",
                f"Best model improves over dummy baseline by only {baseline_improvement:.3f} (threshold {BASELINE_MIN_DELTA}). "
                "Possibly due to target leakage, label noise, or severe preprocessing issue."
            )
            suggestions.append(
                "[HIGH-PRIORITY] Verify the target column consistency; check for leakage in feature set; inspect class label encoding."
            )
        else:
            suggestions.append(
                f"[OK] Model improves over baseline by {baseline_improvement:.3f} - a meaningful signal does exist"
            )

    # absolute performance floor
    if bal_acc < PERF_FLOOR:
        add_issue(
            "high",
            f"balanced_accuracy={bal_acc:.3f} is below the task floor ({PERF_FLOOR}). Model is not reliably better than random chance."
        )
        suggestions.append(
            "Consider feature engineering, alternative encodings, or a different model family."
        )
    else:
        suggestions.append(
            f"[OK] balanced_accuracy={bal_acc:.3f} exceeds the task floor ({PERF_FLOOR})."
        )

    # candidate score spread
    non_dummy = [m for m in all_metrics if "Dummy" not in m.get("model", "")]
    score_spread = 0.0
    if len(non_dummy) >= 2:
        scores = [float(m.get("balanced_accuracy", 0.0)) for m in non_dummy]
        score_spread = max(scores) - min(scores)
        worst_model = non_dummy[scores.index(min(scores))]["model"]

        if score_spread > SPREAD_HIGH:
            add_issue(
                "high",
                f"Very large score spread across candidates ({score_spread:.3f}). '{best_model}' outperforms '{worst_model}' by this margin; pipeline may be highly sensitive to model choice." 
                "This suggests feature noise, instability, or a possible mismatch in preprocessing."
            )
            suggestions.append(
                "Investigate the noisy features; consider ensembling top 2 models or adding cross-validation-based feature selection."
            )
        elif score_spread > SPREAD_WARN:
            add_issue(
                "medium",
                f"Large score spread acorss candidates ({score_spread:.3f}). '{best_model}' outperforms '{worst_model}' by this margin; the pipeline is sensitive to model choice. "
                "This suggests feature noise or instability in the preprocessing."
            )
            suggestions.append(
                "Investigate whether noisy features are inflating the variance. Consider feature selection or an ensemble to stabilize the predictions."
            )

    sig = evaluation.get("significance_test", {})
    for name, result in sig.get("comparisons", {}).items():
        if not result["significant"] and "Logistic" in name:
            suggestions.append(
                f"[OK] Best model is not significantly better than {name} (p={result['p_value']:.3f}) — simpler model may be enough."
                )

    # check for macro-f1
    if f1_macro < F1_WARN:
        add_issue(
            "medium",
            f"macro-F1={f1_macro:.3f} is below {F1_WARN}. At least one class is being predicted poorly."
        )
        suggestions.append(
            "Review per-class precision/recall in the classification report; consider threshold tuning or SMOTE for minority classes."
        )

    # imbalance check
    current_strategy = plan_cfg.get("handle_imbalance", "none")
    if imb >= 3.0:
        if current_strategy == "class_weight":
            suggestions.append(
                f"Imbalance ratio={imb:.2f}: class_weight='balanced' was used on the first pass. If performance is still poor, escalate to SMOTE oversampling on replan."
            )
        elif current_strategy == "smote":
            suggestions.append(
                f"Imbalance ratio={imb:.2f}: SMOTE was already applied. "
                "Consider threshold tuning (by lowering the decision threshold for the minority class) or collecting more minority samples."
            )
        else:
            add_issue(
                "medium",
                f"Imbalance ratio={imb:.2f} but no imbalance strategy was applied."
            )
            suggestions.append(
                "Apply class_weight='balanced' or SMOTE to handle class imbalance."
            )

    # overfitting detection
    if train_metrics is not None:
        train_ba = float(train_metrics.get("balanced_accuracy", 0.0))
        overfit_gap = train_ba - bal_acc
        if overfit_gap > OVERFIT_GAP:
            add_issue(
                "medium",
                f"Possible overfitting: train balanced_accuracy={train_ba:.3f} | test={bal_acc:.3f} (gap={overfit_gap:.3f} > {OVERFIT_GAP}). "
                "Model may not generalise well."
            )
            suggestions.append(
                "Add soem form of regularisation, increase the training data, or apply feature selection to reduce noise."
            )

    # determine the overall status and replan recemmodation
    status = "needs_attention" if issues else "ok"

    # replan if any high issue or two or more medium issues have been detected
    replan_recommended = (
        severity_counts["high"] >= 1 or
        severity_counts["medium"] >= 2
    )

    # assign a replan tier used by apply_replan_strategy()
    # Tier 0 = no replan; Tier 1 = light; Tier 2 = moderate; Tier 3 = aggressive
    if not replan_recommended:
        replan_tier = 0
    elif severity_counts["high"] == 0 and severity_counts["medium"] < 2:
        replan_tier = 1
    elif severity_counts["high"] == 1 or severity_counts["medium"] >= 2:
        replan_tier = 2
    else:
        replan_tier = 3

    return {
        "status": status,
        "best_model": best_model,
        "bal_acc": bal_acc,
        "issues": issues,
        "suggestions": suggestions,
        "severity_counts": severity_counts,
        "score_spread": score_spread,
        "baseline_improvement": baseline_improvement,
        "replan_recommended": replan_recommended,
        "replan_tier": replan_tier,
    }

def should_replan(reflection: Dict[str, Any]) -> bool:
    """
    Return True when the reflector recommends a replan attempt.

    Decision is driven by replan_recommended which accounts for:
    - Any high severity issue 
    - Two or more medium severity issues 
    """
    return bool(reflection.get("replan_recommended", False))

def apply_replan_strategy(
    plan: List[str],
    dataset_profile: Dict[str, Any],
    reflection: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Modify plan and dataset_profile based on reflection tier.

    Four distinct escalation tiers, each more aggressive than the last:

    Tier 1 — score spread only:
        - Switch imputer from mean to median (more robust)
        - Add ExtraTreesClassifier to model pool

    Tier 2 — high severity or 2+ medium issues:
        - Everything in Tier 1
        - Switch imbalance strategy: class_weight to SMOTE (if not already done)
        - Add VotingEnsemble of top-2 models to pool

    Tier 3 — multiple high severity issues:
        - Everything in Tier 2
        - Switch imputer to KNNImputer 
        - Add feature selection step if not already present
        - Increase k_best from 50 to 80 (or add is absent)
        - Add CalibratedClassifierCV wrapper for probability calibration

    Args:
        plan: Current plan steps list.
        dataset_profile: Current dataset profile dict.
        reflection: Output of reflect().

    Returns:
        Tuple of (new_plan, new_profile)
    """
    import copy
    tier = int(reflection.get("replan_tier", 1))
    new_plan = list(plan)
    new_profile = copy.deepcopy(dataset_profile)
    cfg = new_profile.setdefault("_plan_config", {})
    notes = list(new_profile.get("notes", []))

    # tier 1
    if tier >= 1:
        # switch imputer to median
        old_imputer = cfg.get("impute_strategy", "mean")
        if old_imputer == "mean":
            cfg["impute_strategy"] = "median"
            notes.append(
                "[Replan T1] impute_strategy: mean -> median (more robust to residual outliers / skewed distributions)."
            )

        # add ExtraTrees into the extra_models pool
        extras = list(cfg.get("extra_models", []))
        if "ExtraTreesClassifier" not in extras:
            extras.append("ExtraTreesClassifier")
            cfg["extra_models"] = extras
            notes.append(
                "[Replan T1] Added ExtraTreesClassifier to model pool (high variance reduction via extreme random splits)."
            )

        if "replan_tier1" not in new_plan:
            new_plan.append("replan_tier1")

     # tier 2
    if tier >= 2:
        # escalate imbalance strategy
        current_imbalance = cfg.get("handle_imbalance", "none")
        if current_imbalance != "smote":
            cfg["handle_imbalance"] = "smote"
            notes.append(
                "[Replan T2] handle_imbalance: class_weight -> smote (synthetic oversampling of minority class; avoids bias from loss-function re-weighting alone)."
            )

        # aadd VotingEnsemble instruction 
        extras = list(cfg.get("extra_models", []))
        if "VotingEnsemble" not in extras:
            extras.append("VotingEnsemble")
            cfg["extra_models"] = extras
            notes.append(
                "[Replan T2] Added VotingEnsemble (voting over top 2 candidates) to model pool — this reduces variance from model sensitivity."
            )

        if "replan_tier2" not in new_plan:
            new_plan.append("replan_tier2")

    # tier 3
    if tier >= 3:
        # switch to KNN imputation
        cfg["impute_strategy"] = "knn"
        notes.append(
            "[Replan T3] impute_strategy -> knn "
            "(KNNImputer preserves local feature correlations; this is preferred when missingness is not completely at random)."
        )

        # ensure that feature selection is active, expand k_best
        if cfg.get("feature_selection") == "none" or not cfg.get("feature_selection"):
            cfg["feature_selection"] = "select_k_best"
            cfg["k_best"] = 50
            notes.append(
                "[Replan T3] Enabled SelectKBest(k=50) — reduces feature noise that may be driving the high score spread."
            )
        else:
            old_k = cfg.get("k_best", 50)
            new_k = min(old_k + 30, 100)
            cfg["k_best"] = new_k
            notes.append(
                f"[Replan T3] Increased k_best: {old_k} -> {new_k} (broader feature retention after prior selection underperformed)."
            )

        # add probability calibration
        extras = list(cfg.get("extra_models", []))
        if "CalibratedClassifier" not in extras:
            extras.append("CalibratedClassifier")
            cfg["extra_models"] = extras
            notes.append(
                "[Replan T3] Added CalibratedClassifierCV — improves predicted probability reliability for threshold tuning."
            )

        if "replan_tier3" not in new_plan:
            new_plan.append("replan_tier3")

    new_profile["notes"] = notes
    return new_plan, new_profile





