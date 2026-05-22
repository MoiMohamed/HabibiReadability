from __future__ import annotations

from typing import Any

import streamlit as st

from tarab_model_experimentation.barec_coarse import coarse_accuracy_from_confusion_matrix
from tarab_model_experimentation.log_parsing import parse_training_log
from tarab_model_experimentation.selection import (
    experiment_chart_label,
    experiment_chart_sort_key,
    log_display_name,
)


def _avg_from_report(
    report_df, label: str
) -> tuple[float | None, float | None, float | None]:
    if report_df is None or report_df.empty or "label" not in report_df.columns:
        return None, None, None
    row = report_df.loc[report_df["label"] == label]
    if row.empty:
        return None, None, None
    r = row.iloc[0]
    return (
        float(r["precision"]) if r["precision"] is not None else None,
        float(r["recall"]) if r["recall"] is not None else None,
        float(r["f1-score"]) if r["f1-score"] is not None else None,
    )


def _epoch_dev_metrics(epoch: dict[str, Any]) -> dict[str, float | None]:
    report_df = epoch.get("report_df")
    precision, recall, f1 = _avg_from_report(report_df, "weighted avg")
    if precision is None or recall is None or f1 is None:
        mp, mr, mf = _avg_from_report(report_df, "macro avg")
        metrics = epoch.get("metrics") or {}
        precision = precision or mp or metrics.get("eval_macro_precision")
        recall = recall or mr or metrics.get("eval_macro_recall")
        f1 = f1 or mf or metrics.get("eval_macro_f1")

    return {
        "qwk": float(epoch["qwk"]),
        "precision": float(precision) if precision is not None else None,
        "recall": float(recall) if recall is not None else None,
        "f1": float(f1) if f1 is not None else None,
    }


def _epoch_accuracy_metrics(epoch: dict[str, Any]) -> dict[str, float | None]:
    metrics = epoch.get("metrics") or {}
    cm_df = epoch.get("cm_df")

    def _f(key: str) -> float | None:
        val = metrics.get(key)
        return float(val) if val is not None else None

    return {
        "acc": _f("eval_accuracy"),
        "acc_pm1": _f("eval_accuracy_with_margin"),
        "dist": _f("eval_Distance"),
        "acc7": coarse_accuracy_from_confusion_matrix(cm_df, 7),
        "acc5": coarse_accuracy_from_confusion_matrix(cm_df, 5),
        "acc3": coarse_accuracy_from_confusion_matrix(cm_df, 3),
    }


def best_dev_per_metric_from_log(log_filename: str) -> dict[str, Any] | None:
    """Best dev value for each metric, possibly from different epochs."""
    data = parse_training_log(log_filename)
    epochs = data.get("epochs") or []
    if not epochs:
        return None

    scored: list[dict[str, Any]] = []
    for epoch in epochs:
        m = _epoch_dev_metrics(epoch)
        scored.append({"epoch": float(epoch["epoch"]), **m})

    def _best(key: str) -> dict[str, Any]:
        valid = [row for row in scored if row[key] is not None]
        if not valid:
            return {"value": None, "epoch": None}
        best_row = max(valid, key=lambda row: row[key])
        return {"value": best_row[key], "epoch": best_row["epoch"]}

    best_qwk = _best("qwk")
    best_precision = _best("precision")
    best_recall = _best("recall")
    best_f1 = _best("f1")

    return {
        "log_file": log_filename,
        "experiment": experiment_chart_label(log_filename),
        "run_name": data["meta"].get("run_name", ""),
        "qwk": best_qwk["value"],
        "qwk_epoch": best_qwk["epoch"],
        "precision": best_precision["value"],
        "precision_epoch": best_precision["epoch"],
        "recall": best_recall["value"],
        "recall_epoch": best_recall["epoch"],
        "f1": best_f1["value"],
        "f1_epoch": best_f1["epoch"],
    }


def experiment_log_files(log_files: list[str]) -> list[str]:
    """All logs including baseline."""
    return list(log_files)


def collect_experiment_comparison_rows(log_files: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for log_file in experiment_log_files(log_files):
        row = best_dev_per_metric_from_log(log_file)
        if row is not None:
            rows.append(row)
    return rows


def build_experiment_comparison_df(rows: list[dict[str, Any]]):
    import pandas as pd

    if not rows:
        return pd.DataFrame(
            columns=[
                "experiment",
                "qwk",
                "qwk_epoch",
                "precision",
                "precision_epoch",
                "recall",
                "recall_epoch",
                "f1",
                "f1_epoch",
            ]
        )

    ordered = sorted(rows, key=lambda r: experiment_chart_sort_key(r["experiment"]))
    return pd.DataFrame(ordered)


def _plot_metric_line_chart(
    df,
    *,
    value_col: str,
    title: str,
    ylabel: str,
    ax,
) -> None:
    import numpy as np

    plot_df = df.dropna(subset=[value_col])
    if plot_df.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    x = np.arange(len(plot_df))
    y = plot_df[value_col].astype(float)
    ax.plot(x, y, marker="o", linestyle="-", linewidth=2, markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["experiment"], rotation=35, ha="right")
    ax.set_title(title)
    ax.set_xlabel("Experiment")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ymin = float(y.min())
    ymax = float(y.max())
    if ymin == ymax:
        pad = 0.05 if ymax > 0 else 0.01
        ax.set_ylim(ymin - pad, ymax + pad)
    else:
        pad = (ymax - ymin) * 0.08
        ax.set_ylim(max(0.0, ymin - pad), ymax + pad)


def render_performance_comparison_section(*, log_files: list[str]) -> None:
    # --- Hidden: 2×2 best-dev line chart (precision / recall / QWK / F1 by experiment)
    # st.markdown("### Performance comparison (all experiments)")
    # st.caption(
    #     "Best dev score per metric on each experiment. "
    #     "Precision, recall, and F1 use support-weighted averages from each epoch's "
    #     "classification report. QWK, precision, recall, and F1 each use their own best epoch."
    # )
    #
    # rows = collect_experiment_comparison_rows(log_files)
    # if not rows:
    #     st.info("No parseable experiment logs found for comparison.")
    #     return
    #
    # df = build_experiment_comparison_df(rows)
    # if df.empty:
    #     st.info("Could not extract dev metrics from experiment logs.")
    #     return
    #
    # # Clockwise from top-left: precision, recall, QWK, F1.
    # metric_specs = [
    #     ("precision", "Best dev weighted precision", "Weighted precision", 0, 0),
    #     ("recall", "Best dev weighted recall", "Weighted recall", 0, 1),
    #     ("qwk", "Best dev quadratic weighted kappa", "QWK", 1, 1),
    #     ("f1", "Best dev weighted F1-score", "Weighted F1", 1, 0),
    # ]
    #
    # if df[["qwk", "precision", "recall", "f1"]].isna().all().all():
    #     st.warning("Metrics are missing from parsed logs.")
    #     return
    #
    # fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    # for col, title, ylabel, row, ax_col in metric_specs:
    #     _plot_metric_line_chart(
    #         df, value_col=col, title=title, ylabel=ylabel, ax=axes[row, ax_col]
    #     )
    #
    # fig.suptitle(
    #     "Best dev performance by experiment (metric-specific best epoch)",
    #     fontsize=14,
    #     y=1.02,
    # )
    # fig.tight_layout()
    # st.pyplot(fig, clear_figure=True)
    # plt.close(fig)

    from tarab_model_experimentation.metric_optimal_profiles import (
        render_metric_optimal_profiles_section,
    )

    render_metric_optimal_profiles_section(log_files=log_files)
