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

_BASELINE_MARGINAL_GAP = 0.05   
_BASELINE_WEAK_GAP = 0.1
_FLOOR_IMBALANCED = 0.55
_FLOOR_BALANCED = 0.65
_SPREAD_HIGH = 0.1
_PR_GAP_THRESHOLD = 0.15
_CEILING = 0.85
_MEDIUM_ISSUES_FROM_REPLAN = 2


def reflect(
    dataset_profile: Dict[str, Any],
    evaluation: Dict[str, Any],
    all_metrics: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Run 6 diagnostic checks and return a structured reflection dict
    
    Args:
        dataset_profile: Dataset characteristics
        evaluation: Best model's metrics
        all_metrics: Metrics for all trained models
    
    Returns:
        Dictionary with keys:
            - status: str ("ok" or "needs_attention")
            - best_model: str (model name)
            - bal_acc: float (needed by should-replan ceiling check)
            - issues: List[str] (identified problems, tagged [HIGH/MEDIUM/LOW])
            - suggestions: List[str] (improvement recommendations)
            - severity_counts: {"high": int, "medium": int, "low":int}
            - score_spread: float - max-min balanced_accuracy acroos candidates
            - baseline_improvement: float | none
            - replan_recommended: bool (should we replan?)
    
    TODO for students:
    - Implement statistical tests (paired t-tests, Wilcoxon tests)
    - Add per-class performance analysis
    - Detect overfitting vs underfitting
    - Analyze confusion matrix patterns
    - Check for data quality issues
    - Prioritize suggestions by expected impact
    - Learn which suggestions work from memory
    """
    
    # read the evaluation metrics
    best_model = evaluation.get("model", "unknown")
    bal_acc = float(evaluation.get("balanced_accuracy", 0.0))
    f1_macro = float(evaluation.get("f1_macro", 0.0))
    precision = float(evaluation.get("precision_macro", 0.0))
    recall = float(evaluation.get("recall_macro", 0.0))

    # read profiler signals
    imb = float(dataset_profile.get("imbalance_ratio") or 1.0)
    outlier_cols = dataset_profile.get("outlier_cols", [])
    high_card_cols = dataset_profile.get("high_card_cols", [])
    total_missing = float(dataset_profile.get("total_missing_pct", 0.0))
    config = dataset_profile.get("_plan_config", {})
    rows = dataset_profile.get("shape", {}).get("rows", 0)
    
    
    issues: List[str] = []
    suggestions: List[str] = []
    severity: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}

    baseline_improvement: Optional[float] = None
    
    # check 1: baseline comparison
    # is the best model meaningfully better than a majority class dummy?
    dummy = next((m for m in all_metrics if "Dummy" in m.get("model", "")), None)
    
    if dummy is not None:
        dummy_ba = float(dummy.get("balanced_accuracy", 0.0))
        baseline_improvement = bal_acc - dummy_ba
        
        if baseline_improvement < _BASELINE_MARGINAL_GAP:
            issues.append(
                f"[HIGH] Best model ({best_model}) is only {baseline_improvement:.3f} above the dummy baseline - this is a near-random performance. "
                "Likely causes: traget leakage removed, weak features, or a preprocessing bug."
            )
            suggesttion["high"] += 1
            suggestions.append(
                "Verify the target column has not been unknowingly included as a feature. "
                "Check that preprocessing steps run without silent errors. "
                "Consider adding domain-specific features."
            )
        elif baseline_improvement < _BASELINE_WEAK_GAP:
            issues.append(
                f"[MEDIUM] Marginal improvement over the dummy basline ({baseline_improvement:.3f}). "
                "The model is learning but the signal is weak."
            )
            severity["medium"] += 1
            suggestions.append(
                "Feature engineering or additional data may be needed. "
                "Inspect feature importances to confirm that model is using meaningful predictors."
            )
        else:
            suggestions.append(
                f"[OK] Model improves over baseline by {baseline_improvement:.3f} - a meaningful signal does exist"
            )

    # check 2: absolute performance floor
    # adaptive: lower the floor for imbalanced tasks (as this is a harder problem)
    floor = _FLOOR_IMBALANCED if imb >= 3.0 else _FLOOR_BALANCED
    if bal_acc < floor:
        severity_tag = "HIGH" if bal_acc < floor - 0.10 else "MEDIUM"
        issues.append(
            f"[{severity_tag}] balanced_accuracy={bal_acc:.3f} is below the {'imbalanced' if imb >= 3.0 else 'balanced'} - task floor ({floor:.2f}). "
            "The model is not yet ready."
        )
        severity[severity_tag.lower()] += 1
        suggestions.append(
            "Try hyperparameter tuning (GridsearchCV / RandomisedSearchCV) on the best candidate, or add more informative features."
        )
    else:
        suggestions.append(
            f"[OK] balanced_accuracy={bal_acc:.3f} exceeds the task floor ({floor:.2f})."
        )

    # check 3: score spread across the model candidates
    # high spread = signals instabaility; models disagree = pipeline must be noisy

    non_dummy = [m for m in all_metrics if "Dummy" not in m.get("model", "")]
    
    score_spread = 0.0
    if len(non_dummy) > 1:
        scores = [float(m.get("balanced_accuracy", 0)) for m in non_dummy]
        score_spread = max(scores) - min(scores)
        worst_model = non_dummy[scores.index(min(scores))]["model"]

        if score_spread > _SPREAD_HIGH:
            issues.append(
                f"[MEDIUM] LArge score spread acorss candidates ({score_spread:.3f}). "
                f"'{best_model}' outperforms '{worst_model}' by this margin; the pipeline is sensitive to model choice, suggesting feature noise or instability in the preprocessing."
            )
            severity["medium"] += 1
            suggestions.append(
                "Investigate whether noisy features are inflating the variance. "
                "Consider feature selection or an ensemble to stabilize the predictions."
            )
        else:
            suggestions.append(
                f"[OK] Models agree (spread={score_spread:.3f}) - preprocessing is stable across all the algorithms."
                )

    # check 4: precsion-recall tradeoff
    # a large gap signals that the model is biased toward one error type

    pr_gap = abs(precision - recall)
    if pr_gap > _PR_GAP_THRESHOLD:
        if recall < precision:
            issues.append(
                f"[MEDIUM] Recall ({recall:.3f}) is much lower than precision ({precision:.3f}) - the model is conservative and misses minority-class instances. "
                "This is common when class imbalance is not fully corrected."
            )
            severity["medium"] += 1
            suggestions.append(
                "Lower the decision threshold to recover minority-class recall, or escalate to oversampling on replan."
            )
        else:
            issues.append(
                f"[LOW] Precision ({precsion:.3f}) is much lower than recall ({recall:.3f}) - the model over-predicts the positive class, generating many false positives."
            )
            severity["low"] += 1
            suggestions.append(
                "Raise the decision threshold or add regularization to the best model to reduce the false positives."
            )

    # check 5: root-cause diagnosis
    # cross-references what the profiler detected vs what the planner used
    # 5a — imbalance not handled
    if imb >= 3.0 and config.get("handle_imbalance") is None:
        issues.append(
            f"[HIGH] Imbalance ratio={imb:.2f} but no imbalance correction was applied (class_weight or oversample). "
            "The model almost certainly favours the majority class."
        )
        severity["high"] += 1
        suggestions.append(
            "Set handle_imbalance='class_weight' in _plan_config, or escalate to oversampling on replan."
        )

    # 5b — outliers present but standard scaler used
    if outlier_cols and config.get("scaler") == "standard":
        issues.append(
            f"[MEDIUM] Outlier columns {outlier_cols} were detected by data_profiler.py but StandardScaler was applied."
            " Extreme values will have compressed the inlier range, degrading the distance-based and linear models."
        )
        severity["medium"] += 1
        suggestions.append(
            "Switch to RobustScaler on replan (handled automatically by get_replan_config_changes)."
        )

    # 5c - high-cardinality columns present but not encoded with ordinal
    if high_card_cols and config.get("high_card_encoding") != "ordinal":
        issues.append(
            f"[MEDIUM] High-cardinality columns {high_card_cols} exist but OrdinalEncoder was not applied."
            " One-hot encoding these columns may have caused a dimensionality explosion."
        )
        severity["medium"] += 1
        suggestions.append(
            "Ensure high_card_cols are encoded with OrdinalEncoder — this is set automatically when _plan_config is populated by the planner."
        )

    # 5d — high missing rate with mean imputation
    if total_missing > 15.0 and config.get("impute_strategy") == "mean":
        issues.append(
            f"[LOW] total_missing_pct={total_missing:.1f}% is high but mean imputation was used. "
            "Mean is sensitive to skew and outliers; median would be safer."
        )
        severity["low"] += 1
        suggestions.append(
            "Escalate to median imputation on replan (handled automatically by get_replan_config_changes)."
        )

    # check 6: small dataset overfitting risk
    if rows < 500 and not config.get("prefer_simple_models", False):
        issues.append(
            f"[MEDIUM] Only {rows} training rows but complex models were used. There is a risk of overfitting."
        )
        severity["medium"] += 1
        suggestions.append(
            "Enable prefer_simple_models in _plan_config to cap model complexity, or gather more training data."
        )

    # determine the overall status
    status = "needs_attention" if issues else "ok"

    # replanning decision
    near_ceiling = bal_acc >= _CEILING
    has_high = severity["high"] > 0
    has_many_medium = severity["medium"] >= _MEDIUM_ISSUES_FROM_REPLAN
    replan_recommended = (not near_ceiling and (has_high or has_many_medium))

    if near_ceiling:
        suggestions.append(
            f"[OK] balanced_accuracy={bal_acc:.3f} is near the ceiling "
            f"({_CEILING}) — replanning has diminishing returns."
        )

    return {
        "status": status,
        "best_model": best_model,
        "bal_acc": bal_acc,       
        "issues": issues,
        "suggestions": suggestions,
        "severity_counts": severity,
        "score_spread": score_spread,
        "baseline_improvement": baseline_improvement,
        "replan_recommended": replan_recommended,
    }

####################################################################################
def should_replan(reflection: Dict[str, Any]) -> bool:
    """
    Multi-factor replanning policy.

    Triggers replan when:
        - Any HIGH severity issue exists AND  bal_acc < ceiling
        - OR ≥2 MEDIUM severity issues exist AND  bal_acc < ceiling

    Does NOT trigger when:
        - bal_acc ≥ 0.85  (near ceiling — diminishing returns)
        - No issues found  (status == "ok")
    """
    return bool(reflection.get("replan_recommended", False))

#######################################################################################
def apply_replan_strategy(
    plan: List[str],
    dataset_profile: Dict[str, Any],
    reflection: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Apply concrete strategy changes to the plan and profile so that the next training loop uses different preprocessing and models.
    
    - Calls get_replan_config_changes() from agents/planner.py, which
      escalates every strategy in _plan_config:
        imputer       : mean  -> median
        scaler        : standard -> robust
        imbalance     : class_weight -> oversample
        features      : none  -> SelectKBest(k=30)
        model pool    : adds ExtraTreesClassifier

    - Writes the updated config back into dataset_profile["_plan_config"]
      so build_preprocessor() and select_models() in modelling.py immediately pick up the new strategies on the next loop iteration.

    Args:

        plan            : current plan list (will be extended with replan steps)
        dataset_profile : profile dict — _plan_config will be updated in-place
        reflection      : output of reflect()

    Returns:
        (new_plan, new_profile) — new_profile has updated _plan_config
    """
    
    from agents.planner import get_replan_config_changes

    new_plan    = list(plan)
    new_profile = dict(dataset_profile)

    # pull the current config that was created by the planner first pass
    current_config = dict(new_profile.get("_plan_config", {}))

    # get escalted config & reasoning strings from the planner
    new_config, change_strings = get_replan_config_changes(
        current_config, reflection
    )

    # append replan header to the plan
    triggering_issues = reflection.get("issues", [])
    issue_summary = (
        "; ".join(triggering_issues[:2]) if triggering_issues
        else "performance below threshold"
    )
    new_plan.append(
        f"[REPLAN TRIGGERED] Reasons: {issue_summary}"
    )

    # append each strategy chnage as a plan step
    for change in change_strings:
        new_plan.append(change)

    # write the updated config back into the profile
    new_profile["_plan_config"] = new_config

    # append the summary to note
    notes = list(new_profile.get("notes", []))
    notes.append(
        f"Replan applied: {len(change_strings)} strategy change(s) — "
        + ", ".join(
            f"{k}: {current_config.get(k)!s} -> {new_config.get(k)!s}"
            for k in ["impute_strategy", "scaler", "handle_imbalance",
                      "feature_selection", "extra_models"]
            if current_config.get(k) != new_config.get(k)
        )
    )
    new_profile["notes"] = notes

    return new_plan, new_profile
#     # TODO: Add more sophisticated checks
    
#     # Check F1 score
#     # TODO: Make threshold adaptive based on problem difficulty
#     if f1_macro < 0.60:
#         issues.append("Macro F1 score is modest (<0.60).")
#         suggestions.append(
#             "Try different models, tune hyperparameters, "
#             "or improve preprocessing."
#         )
    
#     # TODO: Add imbalance-specific analysis
#     if imb >= 3.0:
#         suggestions.append(
#             "Imbalance detected: consider class_weight, "
#             "threshold tuning, or SMOTE."
#         )
    
#     # TODO: Add checks for:
#     # - Model diversity (are all models performing similarly?)
#     # - Per-class performance (which classes are problematic?)
#     # - Precision-recall tradeoff
#     # - High-cardinality categorical features
#     # - Feature importance patterns
#     # - Learning curves (overfitting/underfitting)
    
#     # Determine status
#     status = "needs_attention" if issues else "ok"
    
#     # Simple replanning trigger
#     # TODO: Make this more sophisticated
#     replan_recommended = bool(issues and f1_macro < 0.60)
    
#     return {
#         "status": status,
#         "best_model": best_model,
#         "issues": issues,
#         "suggestions": suggestions,
#         "replan_recommended": replan_recommended,
#     }


# def should_replan(reflection: Dict[str, Any]) -> bool:
#     """
#     Decide whether to trigger replanning based on reflection.
    
#     This is a simple policy. Students should implement more sophisticated logic.
    
#     TODO for students:
#     - Consider multiple factors (performance, confidence, resource budget)
#     - Implement diminishing returns detection
#     - Use memory to avoid repeating failed strategies
#     - Set adaptive thresholds based on problem difficulty
#     """
#     return bool(reflection.get("replan_recommended", False))


# def apply_replan_strategy(
#     plan: List[str],
#     dataset_profile: Dict[str, Any],
#     reflection: Dict[str, Any],
# ) -> Tuple[List[str], Dict[str, Any]]:
#     """
#     Modify the plan and dataset profile based on reflection.
    
#     This is a very basic implementation. Students should make this sophisticated.
    
#     Args:
#         plan: Current execution plan
#         dataset_profile: Current dataset profile
#         reflection: Reflection results
    
#     Returns:
#         Tuple of (modified_plan, modified_profile)
    
#     TODO for students:
#     - Implement specific strategies for specific issues
#     - Add preprocessing steps based on identified problems
#     - Modify model selection based on performance patterns
#     - Adjust hyperparameters
#     - Try ensemble methods
#     - Implement different replan strategies (aggressive, conservative)
#     """
    
#     # Copy to avoid modifying originals
#     new_plan = list(plan)
#     new_profile = dict(dataset_profile)
    
#     # Basic strategy: add a note
#     # TODO: Implement actual strategy changes
#     notes = list(new_profile.get("notes", []))
#     notes.append("Replan: adjusting strategy after reflection.")
#     new_profile["notes"] = notes
    
#     new_plan.append("replan_attempt")
    
#     # TODO: Implement sophisticated replan strategies:
#     # - If low performance: try ensemble methods
#     # - If imbalance issues: add SMOTE or adjust thresholds
#     # - If overfitting: add regularization
#     # - If underfitting: increase model complexity
#     # - If feature issues: add feature engineering steps
    
#     return new_plan, new_profile


# # TODO: Add helper functions for reflection
# # def compare_models_statistically(...):
# # def analyze_per_class_performance(...):
# # def detect_overfitting(...):
# # def detect_data_quality_issues(...):
# # def prioritize_suggestions(...):
# # def generate_explanation(...):
