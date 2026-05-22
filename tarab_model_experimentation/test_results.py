from __future__ import annotations

from typing import Any

import streamlit as st

_METRIC_LABELS: dict[str, str] = {
    "test_Quadratic_Weighted_Kappa": "QWK",
    "test_accuracy": "Accuracy",
    "test_accuracy_with_margin": "Accuracy ±1",
    "test_Distance": "Avg distance",
    "test_macro_f1": "Macro F1",
    "test_macro_precision": "Macro precision",
    "test_macro_recall": "Macro recall",
    "test_loss": "Loss",
    "test_runtime": "Runtime (s)",
    "test_samples_per_second": "Samples/s",
    "test_steps_per_second": "Steps/s",
}


def _metric_display_name(key: str) -> str:
    return _METRIC_LABELS.get(key, key.replace("test_", "").replace("_", " "))


def _format_metric_value(key: str, value: float) -> str:
    if key == "test_Quadratic_Weighted_Kappa":
        return f"{value:.4f}"
    if key in ("test_accuracy", "test_accuracy_with_margin", "test_macro_f1", "test_macro_precision", "test_macro_recall"):
        return f"{value:.4f}"
    if key == "test_Distance":
        return f"{value:.4f}"
    if key == "test_loss":
        return f"{value:.4f}"
    return f"{value:.4g}"


def build_test_metrics_comparison_table(
    exp_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    *,
    experiment_name: str,
) -> "pd.DataFrame":
    import pandas as pd

    keys = sorted(set(exp_metrics) | set(baseline_metrics))
    rows: list[dict[str, Any]] = []
    for key in keys:
        base_v = baseline_metrics.get(key)
        exp_v = exp_metrics.get(key)
        delta = None
        if base_v is not None and exp_v is not None:
            delta = exp_v - base_v
        rows.append(
            {
                "metric": _metric_display_name(key),
                "baseline": _format_metric_value(key, base_v) if base_v is not None else "—",
                experiment_name: _format_metric_value(key, exp_v) if exp_v is not None else "—",
                "Δ (exp − baseline)": (
                    f"{delta:+.4f}"
                    if delta is not None
                    and key
                    not in ("test_runtime", "test_samples_per_second", "test_steps_per_second")
                    else ("—" if delta is None else f"{delta:+.4g}")
                ),
            }
        )
    return pd.DataFrame(rows)


def render_test_results_section(
    exp_data: dict[str, Any],
    *,
    baseline_data: dict[str, Any] | None = None,
    selected_display: str | None = None,
) -> None:
    import pandas as pd

    st.markdown("### Test results")
    test_metrics = exp_data.get("test_metrics") or {}
    baseline_metrics = (baseline_data or {}).get("test_metrics") or {}
    exp_name = selected_display or "experiment"

    if not test_metrics and not baseline_metrics:
        st.info("No `***** test metrics *****` block was found in these log files.")
        return

    if baseline_metrics and test_metrics:
        st.caption("Held-out test set at end of training (from log `***** test metrics *****`).")
        cmp_df = build_test_metrics_comparison_table(
            test_metrics, baseline_metrics, experiment_name=exp_name
        )
        st.dataframe(cmp_df, width="stretch", hide_index=True)
        qwk_b = baseline_metrics.get("test_Quadratic_Weighted_Kappa")
        qwk_e = test_metrics.get("test_Quadratic_Weighted_Kappa")
        if qwk_b is not None and qwk_e is not None:
            st.markdown(
                f"**Test QWK:** baseline **{qwk_b:.4f}** · {exp_name} **{qwk_e:.4f}** "
                f"(**{qwk_e - qwk_b:+.4f}**)"
            )
        return

    if test_metrics:
        st.caption(f"Test metrics for **{exp_name}** (no baseline log parsed).")
        test_df = pd.DataFrame(
            [
                {"metric": _metric_display_name(k), "value": _format_metric_value(k, v)}
                for k, v in sorted(test_metrics.items())
            ]
        )
        st.dataframe(test_df, width="stretch", hide_index=True)
        return

    st.caption("Baseline test metrics only (selected experiment log has no test block).")
    test_df = pd.DataFrame(
        [
            {"metric": _metric_display_name(k), "value": _format_metric_value(k, v)}
            for k, v in sorted(baseline_metrics.items())
        ]
    )
    st.dataframe(test_df, width="stretch", hide_index=True)


def render_primary_test_results_section(log_files: list[str]) -> None:
    """Test metrics for canonical baseline vs dist_155K (no experiment picker)."""
    from tarab_model_experimentation.log_parsing import parse_training_log
    from tarab_model_experimentation.selection import (
        log_display_name,
        preferred_log_for_chart_label,
    )

    baseline_file = preferred_log_for_chart_label(log_files, "baseline")
    exp_file = preferred_log_for_chart_label(log_files, "dist_155K")
    if baseline_file is None or exp_file is None:
        st.info("Test results need baseline and dist_155K logs in `data/logs`.")
        return

    baseline_data = parse_training_log(baseline_file)
    exp_data = parse_training_log(exp_file)
    render_test_results_section(
        exp_data,
        baseline_data=baseline_data,
        selected_display=log_display_name(exp_file),
    )
