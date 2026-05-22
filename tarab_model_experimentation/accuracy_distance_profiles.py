from __future__ import annotations

from typing import Any

import streamlit as st

from tarab_model_experimentation.constants import DIST_ORANGE
from tarab_model_experimentation.log_parsing import parse_training_log
from tarab_model_experimentation.metric_optimal_profiles import (
    _add_baseline_reference_lines,
    _decorate_experiment_groups,
    _mark_scale_break,
    _panel_ylim,
    fit_figure_legend_below_labels,
)
from tarab_model_experimentation.performance_comparison import (
    _epoch_accuracy_metrics,
    experiment_log_files,
)
from tarab_model_experimentation.selection import (
    experiment_chart_label,
    experiment_chart_sort_key,
)

CRITERION_KEYS = ("qwk", "acc", "acc_pm1", "dist")
METRIC_KEYS = ("qwk", "acc", "acc_pm1", "dist", "acc7", "acc5", "acc3")
TOP_METRIC = "qwk"
MID_METRIC = "dist"
BOTTOM_METRICS = ("acc", "acc_pm1")
COARSE_METRICS = ("acc7", "acc5", "acc3")
MINIMIZE_CRITERIA = frozenset({"dist"})

METRIC_LABELS = {
    "qwk": "QWK",
    "acc": "Acc19",
    "acc_pm1": "Acc19 ±1",
    "dist": "Distance",
    "acc7": "Acc7",
    "acc5": "Acc5",
    "acc3": "Acc3",
}
METRIC_STYLES = {
    "qwk": {
        "color": "#d62728",
        "marker": "D",
        "linestyle": "-",
        "linewidth": 1.8,
        "alpha": 1.0,
        "markersize": 6,
        "zorder": 4,
    },
    "dist": {
        "color": "#9467bd",
        "marker": "v",
        "linestyle": "-",
        "linewidth": 1.3,
        "alpha": 0.85,
        "markersize": 5,
        "zorder": 4,
    },
    "acc": {
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 1.0,
        "alpha": 0.6,
        "markersize": 4.5,
        "zorder": 3,
    },
    "acc_pm1": {
        "color": DIST_ORANGE,
        "marker": "s",
        "linestyle": "--",
        "linewidth": 1.0,
        "alpha": 0.6,
        "markersize": 4.5,
        "zorder": 3,
    },
    "acc7": {
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "-",
        "linewidth": 1.0,
        "alpha": 0.65,
        "markersize": 4.5,
        "zorder": 3,
    },
    "acc5": {
        "color": "#17becf",
        "marker": "D",
        "linestyle": "-",
        "linewidth": 1.0,
        "alpha": 0.65,
        "markersize": 4.0,
        "zorder": 3,
    },
    "acc3": {
        "color": "#bcbd22",
        "marker": "p",
        "linestyle": "-",
        "linewidth": 1.0,
        "alpha": 0.65,
        "markersize": 4.5,
        "zorder": 3,
    },
}
PANEL_TITLES = {
    "qwk": "Checkpoint selected by best QWK",
    "acc": "Checkpoint selected by best accuracy",
    "acc_pm1": "Checkpoint selected by best accuracy ±1",
    "dist": "Checkpoint selected by lowest distance",
}

CHECKPOINT_VIEW_ORDER = ("qwk", "acc", "acc_pm1", "dist", "all")
CHECKPOINT_VIEW_LABELS = {
    "qwk": "QWK",
    "acc": "Accuracy",
    "acc_pm1": "Acc ±1",
    "dist": "Distance",
    "all": "All (2×2 comparison)",
}


def _best_row(rows: list[dict[str, Any]], criterion: str) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get(criterion) is not None]
    if not valid:
        return None
    if criterion in MINIMIZE_CRITERIA:
        return min(valid, key=lambda row: row[criterion])
    return max(valid, key=lambda row: row[criterion])


def profiles_at_metric_optimal_epochs(log_filename: str) -> dict[str, Any] | None:
    data = parse_training_log(log_filename)
    epochs = data.get("epochs") or []
    if not epochs:
        return None

    scored: list[dict[str, Any]] = []
    for epoch in epochs:
        scored.append(
            {
                "epoch": float(epoch["epoch"]),
                "qwk": float(epoch["qwk"]),
                **_epoch_accuracy_metrics(epoch),
            }
        )

    profiles: dict[str, dict[str, Any] | None] = {}
    for criterion in CRITERION_KEYS:
        best = _best_row(scored, criterion)
        if best is None:
            profiles[criterion] = None
            continue
        profiles[criterion] = {
            "epoch": best["epoch"],
            "qwk": best["qwk"],
            "acc": best["acc"],
            "acc_pm1": best["acc_pm1"],
            "dist": best["dist"],
            "acc7": best["acc7"],
            "acc5": best["acc5"],
            "acc3": best["acc3"],
        }

    return {
        "experiment": experiment_chart_label(log_filename),
        "profiles": profiles,
    }


def collect_accuracy_distance_profile_rows(log_files: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for log_file in experiment_log_files(log_files):
        row = profiles_at_metric_optimal_epochs(log_file)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: experiment_chart_sort_key(r["experiment"]))
    return rows


def _row_label_with_epoch(experiment: str, profile: dict[str, Any] | None) -> str:
    if not profile:
        return experiment
    epoch = profile["epoch"]
    epoch_str = str(int(epoch)) if epoch == int(epoch) else f"{epoch:g}"
    return f"{experiment} (ep {epoch_str})"


def build_profile_table_df(rows: list[dict[str, Any]], criterion: str):
    import pandas as pd

    records: list[dict[str, Any]] = []
    for row in rows:
        profile = row["profiles"].get(criterion)
        if profile is None:
            continue
        rec = {"Experiment": _row_label_with_epoch(row["experiment"], profile)}
        for key in METRIC_KEYS:
            val = profile.get(key)
            rec[METRIC_LABELS[key]] = round(float(val), 4) if val is not None else None
        records.append(rec)
    return pd.DataFrame(records)


def _collect_panel_series(
    rows: list[dict[str, Any]], criterion: str
) -> tuple[list[str], dict[str, list[float]]]:
    labels_short: list[str] = []
    series: dict[str, list[float]] = {m: [] for m in METRIC_KEYS}

    for row in rows:
        profile = row["profiles"].get(criterion)
        if profile is None:
            continue
        labels_short.append(row["experiment"])
        for metric in METRIC_KEYS:
            val = profile.get(metric)
            series[metric].append(float(val) if val is not None else float("nan"))

    return labels_short, series


def _baseline_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((r for r in rows if r["experiment"] == "baseline"), None)


def _baseline_reference_series(
    rows: list[dict[str, Any]], criterion: str
) -> dict[str, list[float]] | None:
    base = _baseline_row(rows)
    if not base:
        return None
    profile = base["profiles"].get(criterion)
    if not profile:
        return None
    return {
        m: [float(profile[m])] if profile.get(m) is not None else []
        for m in METRIC_KEYS
    }


def _highlight_best(
    ax,
    *,
    x,
    series: dict[str, list[float]],
    criterion: str,
    ylim: tuple[float, float] | None,
) -> None:
    import numpy as np

    vals = np.array(series.get(criterion, []), dtype=float)
    if vals.size == 0 or np.all(np.isnan(vals)):
        return
    if criterion in MINIMIZE_CRITERIA:
        best_idx = int(np.nanargmin(vals))
    else:
        best_idx = int(np.nanargmax(vals))
    best_y = float(vals[best_idx])
    style = METRIC_STYLES[criterion]

    if ylim:
        y_span = ylim[1] - ylim[0]
    else:
        ymin, ymax = ax.get_ylim()
        y_span = ymax - ymin

    if criterion in MINIMIZE_CRITERIA:
        label_y = best_y - y_span * 0.11
        va = "top"
    else:
        label_y = best_y + y_span * 0.11
        va = "bottom"

    ax.scatter(
        [best_idx],
        [best_y],
        s=42,
        facecolors="white",
        edgecolors=style["color"],
        linewidths=1.5,
        zorder=6,
    )
    ax.annotate(
        "best",
        xy=(best_idx, best_y),
        xytext=(best_idx, label_y),
        textcoords="data",
        ha="center",
        va=va,
        fontsize=7,
        color=style["color"],
        arrowprops={
            "arrowstyle": "-",
            "color": style["color"],
            "linewidth": 0.9,
            "shrinkA": 2,
            "shrinkB": 2,
        },
        zorder=7,
    )


def _set_metric_ylabel(ax, metrics: tuple[str, ...]) -> None:
    if len(metrics) == 1:
        ax.set_ylabel(METRIC_LABELS[metrics[0]], fontsize=9, color="#333333")
    elif metrics == COARSE_METRICS:
        ax.set_ylabel("Acc7 / Acc5 / Acc3", fontsize=9, color="#333333")
    else:
        ax.set_ylabel("Acc19 / Acc19 ±1", fontsize=9, color="#333333")


def _ylim_with_annotation_pad(
    ylim: tuple[float, float] | None,
    *,
    criterion: str,
    section_metrics: tuple[str, ...] | str,
) -> tuple[float, float] | None:
    allowed = (section_metrics,) if isinstance(section_metrics, str) else section_metrics
    if ylim is None or criterion not in allowed:
        return ylim
    lo, hi = ylim
    pad = (hi - lo) * 0.14
    if criterion in MINIMIZE_CRITERIA:
        return lo - pad, hi
    return lo, hi + pad


def _plot_metric_lines(
    ax,
    *,
    x,
    series: dict[str, list[float]],
    metrics: tuple[str, ...],
    ylim: tuple[float, float] | None,
    baseline_series: dict[str, list[float]] | None,
    criterion: str,
    highlight_best: bool,
) -> None:
    _add_baseline_reference_lines(ax, baseline_series=baseline_series, metrics=metrics)

    for metric in metrics:
        style = METRIC_STYLES[metric]
        ax.plot(
            x,
            series[metric],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            color=style["color"],
            alpha=style["alpha"],
            zorder=style["zorder"],
        )

    if ylim:
        ax.set_ylim(*ylim)

    if highlight_best and criterion in metrics:
        _highlight_best(ax, x=x, series=series, criterion=criterion, ylim=ylim)

    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#bbbbbb")


def _plot_stacked_panel(
    fig,
    grid_cell,
    *,
    rows: list[dict[str, Any]],
    criterion: str,
    show_xlabel_title: bool,
    show_metric_ylabel: bool,
    baseline_series: dict[str, list[float]] | None,
) -> list[tuple[Any, Any]] | None:
    import numpy as np

    labels_short, series = _collect_panel_series(rows, criterion)
    if not labels_short:
        ax = fig.add_subplot(grid_cell)
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        return None

    inner = grid_cell.subgridspec(4, 1, height_ratios=[1, 0.85, 0.9, 1.05], hspace=0.13)
    ax_qwk = fig.add_subplot(inner[0])
    ax_dist = fig.add_subplot(inner[1])
    ax_coarse = fig.add_subplot(inner[2])
    ax_acc = fig.add_subplot(inner[3])

    x = np.arange(len(labels_short))
    ylim_qwk = _ylim_with_annotation_pad(
        _panel_ylim(series, (TOP_METRIC,), baseline_series=baseline_series),
        criterion=criterion,
        section_metrics=TOP_METRIC,
    )
    ylim_dist = _ylim_with_annotation_pad(
        _panel_ylim(series, (MID_METRIC,), baseline_series=baseline_series),
        criterion=criterion,
        section_metrics=MID_METRIC,
    )
    ylim_coarse = _ylim_with_annotation_pad(
        _panel_ylim(series, COARSE_METRICS, baseline_series=baseline_series),
        criterion=criterion,
        section_metrics=COARSE_METRICS,
    )
    ylim_acc = _ylim_with_annotation_pad(
        _panel_ylim(series, BOTTOM_METRICS, baseline_series=baseline_series),
        criterion=criterion,
        section_metrics=BOTTOM_METRICS,
    )

    _plot_metric_lines(
        ax_qwk,
        x=x,
        series=series,
        metrics=(TOP_METRIC,),
        ylim=ylim_qwk,
        baseline_series=baseline_series,
        criterion=criterion,
        highlight_best=(criterion == TOP_METRIC),
    )
    ax_qwk.set_title(PANEL_TITLES[criterion], fontsize=10, pad=6)
    if show_metric_ylabel:
        _set_metric_ylabel(ax_qwk, (TOP_METRIC,))
    else:
        ax_qwk.set_ylabel("")
    ax_qwk.set_xticks(x)
    ax_qwk.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    _decorate_experiment_groups(ax_qwk, labels_short)

    _plot_metric_lines(
        ax_dist,
        x=x,
        series=series,
        metrics=(MID_METRIC,),
        ylim=ylim_dist,
        baseline_series=baseline_series,
        criterion=criterion,
        highlight_best=(criterion == MID_METRIC),
    )
    if show_metric_ylabel:
        _set_metric_ylabel(ax_dist, (MID_METRIC,))
    else:
        ax_dist.set_ylabel("")
    ax_dist.set_xticks(x)
    ax_dist.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    _decorate_experiment_groups(ax_dist, labels_short)

    _plot_metric_lines(
        ax_coarse,
        x=x,
        series=series,
        metrics=COARSE_METRICS,
        ylim=ylim_coarse,
        baseline_series=baseline_series,
        criterion=criterion,
        highlight_best=(criterion in COARSE_METRICS),
    )
    if show_metric_ylabel:
        _set_metric_ylabel(ax_coarse, COARSE_METRICS)
    else:
        ax_coarse.set_ylabel("")
    ax_coarse.set_xticks(x)
    ax_coarse.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    _decorate_experiment_groups(ax_coarse, labels_short)

    _plot_metric_lines(
        ax_acc,
        x=x,
        series=series,
        metrics=BOTTOM_METRICS,
        ylim=ylim_acc,
        baseline_series=baseline_series,
        criterion=criterion,
        highlight_best=(criterion in BOTTOM_METRICS),
    )
    if show_metric_ylabel:
        _set_metric_ylabel(ax_acc, BOTTOM_METRICS)
    else:
        ax_acc.set_ylabel("")
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(labels_short, rotation=35, ha="right", fontsize=8)
    if show_xlabel_title:
        ax_acc.set_xlabel("Experiment", fontsize=9, labelpad=8)
    _decorate_experiment_groups(ax_acc, labels_short)

    for ax in (ax_qwk, ax_dist, ax_coarse):
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", which="both", bottom=False)
    for ax in (ax_dist, ax_coarse, ax_acc):
        ax.spines["top"].set_visible(False)

    return [
        (ax_qwk, ax_dist),
        (ax_dist, ax_coarse),
        (ax_coarse, ax_acc),
    ]


def _metric_legend_handles():
    from matplotlib.lines import Line2D

    handles = []
    for metric in (TOP_METRIC, MID_METRIC, *COARSE_METRICS, *BOTTOM_METRICS):
        style = METRIC_STYLES[metric]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                markersize=8,
                label=METRIC_LABELS[metric],
            )
        )
    return handles


def _render_profiles_figure(*, rows: list[dict[str, Any]], view: str):
    import matplotlib.pyplot as plt

    scale_break_pairs: list[tuple[Any, Any]] = []

    if view == "all":
        fig = plt.figure(figsize=(14, 16))
        outer_gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.26)
        panel_specs = [
            ("qwk", 0, 0),
            ("acc", 0, 1),
            ("acc_pm1", 1, 0),
            ("dist", 1, 1),
        ]
        for _criterion, row, col in panel_specs:
            breaks = _plot_stacked_panel(
                fig,
                outer_gs[row, col],
                rows=rows,
                criterion=_criterion,
                show_xlabel_title=(col == 0),
                show_metric_ylabel=(col == 0),
                baseline_series=_baseline_reference_series(rows, _criterion),
            )
            if breaks is not None:
                scale_break_pairs.extend(breaks)
        fig.subplots_adjust(left=0.09, right=0.99, top=0.94, bottom=0.13)
    else:
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(1, 1)
        breaks = _plot_stacked_panel(
            fig,
            gs[0, 0],
            rows=rows,
            criterion=view,
            show_xlabel_title=True,
            show_metric_ylabel=True,
            baseline_series=_baseline_reference_series(rows, view),
        )
        if breaks is not None:
            scale_break_pairs.extend(breaks)
        fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.14)

    for ax_upper, ax_lower in scale_break_pairs:
        _mark_scale_break(fig, ax_upper, ax_lower)
    fit_figure_legend_below_labels(fig, handles=_metric_legend_handles(), ncol=7)
    return fig


def render_accuracy_distance_profiles_section(*, log_files: list[str]) -> None:
    import matplotlib.pyplot as plt

    st.markdown("### Accuracy & distance profile at QWK-optimal checkpoints")

    rows = collect_accuracy_distance_profile_rows(log_files)
    if not rows:
        st.info("No parseable experiment logs found for accuracy/distance profiles.")
        return

    view = "qwk"  # default; radio hidden for now
    # view = st.radio(
    #     "Checkpoint selection criterion (accuracy / distance)",
    #     options=list(CHECKPOINT_VIEW_ORDER),
    #     format_func=lambda key: CHECKPOINT_VIEW_LABELS[key],
    #     index=0,
    #     horizontal=True,
    #     key="accuracy_distance_profile_view",
    # )
    #
    # if view == "all":
    #     st.caption(
    #         "Four panels: QWK → distance → Acc7/5/3 → Acc19/±1 at each criterion's optimal "
    #         "checkpoint. Coarse Acc from BAREC collapse (Fig. 1). Dashed lines = baseline; "
    #         "∥ = scale breaks. Distance: lower is better."
    #     )
    # else:
    st.caption(
        f"{PANEL_TITLES[view]}. Stacked: QWK, distance, BAREC Acc7/5/3, Acc19/±1 at that "
        "checkpoint; dashed lines = baseline; ∥ = scale breaks."
    )

    fig = _render_profiles_figure(rows=rows, view=view)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    table_criteria = CRITERION_KEYS if view == "all" else (view,)
    with st.expander("Numeric tables (accuracy / distance)"):
        for criterion in table_criteria:
            table_df = build_profile_table_df(rows, criterion)
            if table_df.empty:
                continue
            st.caption(PANEL_TITLES[criterion])
            st.dataframe(table_df, width="stretch", hide_index=True)

    if view == "qwk":
        from tarab_model_experimentation.presentation_insights import (
            render_qwk_profile_match_distribution_insight,
        )

        render_qwk_profile_match_distribution_insight(rows, log_files=log_files)

    from tarab_model_experimentation.pseudo_label_oov import render_pseudo_label_oov_section
    from tarab_model_experimentation.focused_analysis import (
        render_focused_baseline_vs_dist_155k_section,
    )

    render_pseudo_label_oov_section()
    render_focused_baseline_vs_dist_155k_section(log_files=log_files)
