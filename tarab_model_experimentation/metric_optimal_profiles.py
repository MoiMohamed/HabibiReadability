from __future__ import annotations

from typing import Any

import streamlit as st

from tarab_model_experimentation.constants import DIST_ORANGE
from tarab_model_experimentation.log_parsing import parse_training_log
from tarab_model_experimentation.performance_comparison import (
    _epoch_dev_metrics,
    experiment_log_files,
)
from tarab_model_experimentation.selection import (
    experiment_chart_label,
    experiment_chart_sort_key,
)

CRITERION_KEYS = ("precision", "recall", "f1", "qwk")
METRIC_KEYS = ("precision", "recall", "f1", "qwk")
PRF_METRICS = ("precision", "recall", "f1")
METRIC_LABELS = {
    "precision": "Weighted precision",
    "recall": "Weighted recall",
    "f1": "Weighted F1",
    "qwk": "QWK",
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
    "precision": {
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 1.0,
        "alpha": 0.6,
        "markersize": 4.5,
        "zorder": 3,
    },
    "recall": {
        "color": DIST_ORANGE,
        "marker": "s",
        "linestyle": "-",
        "linewidth": 1.0,
        "alpha": 0.6,
        "markersize": 4.5,
        "zorder": 3,
    },
    "f1": {
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "--",
        "linewidth": 1.0,
        "alpha": 0.6,
        "markersize": 4.5,
        "zorder": 3,
    },
}
PANEL_TITLES = {
    "precision": "Checkpoint selected by best precision",
    "recall": "Checkpoint selected by best recall",
    "f1": "Checkpoint selected by best F1",
    "qwk": "Checkpoint selected by best QWK",
}
GROUP_FILL = {
    "baseline": "#f5f5f5",
    "dist": "#eef3fa",
    "uni": "#f3f7ee",
}

CHECKPOINT_VIEW_ORDER = ("qwk", "precision", "recall", "f1", "all")
CHECKPOINT_VIEW_LABELS = {
    "qwk": "QWK",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "all": "All (2×2 comparison)",
}


def profiles_at_metric_optimal_epochs(log_filename: str) -> dict[str, Any] | None:
    data = parse_training_log(log_filename)
    epochs = data.get("epochs") or []
    if not epochs:
        return None

    scored: list[dict[str, Any]] = []
    for epoch in epochs:
        scored.append({"epoch": float(epoch["epoch"]), **_epoch_dev_metrics(epoch)})

    profiles: dict[str, dict[str, Any] | None] = {}
    for criterion in CRITERION_KEYS:
        valid = [row for row in scored if row.get(criterion) is not None]
        if not valid:
            profiles[criterion] = None
            continue
        best = max(valid, key=lambda row: row[criterion])
        profiles[criterion] = {
            "epoch": best["epoch"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "qwk": best["qwk"],
        }

    return {
        "experiment": experiment_chart_label(log_filename),
        "profiles": profiles,
    }


def collect_metric_optimal_profile_rows(log_files: list[str]) -> list[dict[str, Any]]:
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
            rec[METRIC_LABELS[key]] = round(float(val), 3) if val is not None else None
        records.append(rec)
    return pd.DataFrame(records)


def _experiment_group(label: str) -> str:
    if label == "baseline":
        return "baseline"
    if label.startswith("dist_") or label == "length_matched":
        return "dist"
    if label.startswith("uni_"):
        return "uni"
    return "other"


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


def _data_ylim(
    values: list[float],
    *,
    rel_pad: float = 0.08,
    min_pad: float = 0.003,
    top_extra_frac: float = 0.0,
) -> tuple[float, float] | None:
    import numpy as np

    clean = [float(v) for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return None
    ymin, ymax = min(clean), max(clean)
    if ymin == ymax:
        pad = max(min_pad, abs(ymax) * 0.02 if ymax else 0.01)
    else:
        pad = max((ymax - ymin) * rel_pad, min_pad)
    lo = ymin - pad
    hi = ymax + pad
    if top_extra_frac > 0:
        hi += (hi - lo) * top_extra_frac
    return lo, hi


def _panel_ylim(
    series: dict[str, list[float]],
    metrics: tuple[str, ...],
    *,
    baseline_series: dict[str, list[float]] | None,
    top_extra_frac: float = 0.0,
) -> tuple[float, float] | None:
    values: list[float] = []
    for metric in metrics:
        values.extend(series.get(metric, []))
        if baseline_series:
            values.extend(baseline_series.get(metric, []))
    return _data_ylim(values, top_extra_frac=top_extra_frac)


def _group_spans(labels: list[str]) -> list[tuple[str, int, int]]:
    """Return (group_name, start_idx, end_idx) inclusive spans."""
    if not labels:
        return []
    spans: list[tuple[str, int, int]] = []
    start = 0
    current = _experiment_group(labels[0])
    for i, label in enumerate(labels[1:], start=1):
        g = _experiment_group(label)
        if g != current:
            spans.append((current, start, i - 1))
            start = i
            current = g
    spans.append((current, start, len(labels) - 1))
    return spans


def _decorate_experiment_groups(ax, labels: list[str]) -> None:
    for group, start, end in _group_spans(labels):
        fill = GROUP_FILL.get(group)
        if fill:
            ax.axvspan(start - 0.5, end + 0.5, facecolor=fill, edgecolor="none", zorder=0)
        if group == "dist" and end + 1 < len(labels) and _experiment_group(labels[end + 1]) == "uni":
            ax.axvline(end + 0.5, color="#999999", linestyle=":", linewidth=1.0, zorder=1)


def _add_baseline_reference_lines(
    ax,
    *,
    baseline_series: dict[str, list[float]] | None,
    metrics: tuple[str, ...],
) -> None:
    if not baseline_series:
        return
    for metric in metrics:
        vals = baseline_series.get(metric, [])
        if not vals:
            continue
        y = vals[0]
        if y is None or y != y:  # NaN check without numpy
            continue
        ax.axhline(
            y,
            color="#666666",
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
            zorder=2,
        )


def _highlight_best(
    ax,
    *,
    x,
    series: dict[str, list[float]],
    criterion: str,
    metrics: tuple[str, ...],
    ylim: tuple[float, float] | None,
) -> None:
    import numpy as np

    vals = np.array(series.get(criterion, []), dtype=float)
    if vals.size == 0 or np.all(np.isnan(vals)):
        return
    best_idx = int(np.nanargmax(vals))
    best_y = float(vals[best_idx])
    style = METRIC_STYLES[criterion]

    if ylim:
        y_span = ylim[1] - ylim[0]
    else:
        ymin, ymax = ax.get_ylim()
        y_span = ymax - ymin
    label_y = best_y + y_span * 0.11

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
        va="bottom",
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
    """Neutral y-axis labels; metric colors appear in the legend only."""
    if len(metrics) == 1:
        ax.set_ylabel(METRIC_LABELS[metrics[0]], fontsize=9, color="#333333")
    else:
        ax.set_ylabel("Weighted P / R / F1", fontsize=9, color="#333333")


def _plot_metric_lines(
    ax,
    *,
    x,
    series: dict[str, list[float]],
    metrics: tuple[str, ...],
    show_legend: bool,
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
            label=METRIC_LABELS[metric] if show_legend else None,
            zorder=style["zorder"],
        )

    if ylim:
        ax.set_ylim(*ylim)

    if highlight_best and criterion in metrics:
        _highlight_best(
            ax, x=x, series=series, criterion=criterion, metrics=metrics, ylim=ylim
        )

    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#bbbbbb")


def _mark_scale_break(fig, ax_qwk, ax_prf) -> None:
    """Horizontal rule + ∥ marker between QWK and P/R/F1 subplots."""
    from matplotlib.lines import Line2D

    pos_q = ax_qwk.get_position()
    pos_p = ax_prf.get_position()
    gap_y = (pos_q.y0 + pos_p.y1) / 2
    x_left = pos_q.x0

    fig.add_artist(
        Line2D(
            [pos_q.x0, pos_q.x1],
            [gap_y, gap_y],
            transform=fig.transFigure,
            color="#777777",
            linewidth=0.9,
            clip_on=False,
            zorder=12,
        )
    )

    tick_h = 0.007
    tick_w = 0.005
    for x_off in (0.0, 0.011):
        fig.add_artist(
            Line2D(
                [x_left + x_off, x_left + x_off + tick_w],
                [gap_y - tick_h, gap_y + tick_h],
                transform=fig.transFigure,
                color="#777777",
                linewidth=0.9,
                clip_on=False,
                zorder=12,
            )
        )


def _plot_split_panel(
    fig,
    grid_cell,
    *,
    rows: list[dict[str, Any]],
    criterion: str,
    show_xlabel_title: bool,
    show_metric_ylabel: bool,
    baseline_series: dict[str, list[float]] | None,
) -> tuple[Any, Any] | None:
    import numpy as np

    labels_short, series = _collect_panel_series(rows, criterion)
    if not labels_short:
        ax = fig.add_subplot(grid_cell)
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        return None

    inner = grid_cell.subgridspec(2, 1, height_ratios=[1, 1.15], hspace=0.16)
    ax_qwk = fig.add_subplot(inner[0])
    ax_prf = fig.add_subplot(inner[1])

    x = np.arange(len(labels_short))
    ylim_qwk = _panel_ylim(
        series,
        ("qwk",),
        baseline_series=baseline_series,
        top_extra_frac=0.14 if criterion == "qwk" else 0.0,
    )
    ylim_prf = _panel_ylim(
        series,
        PRF_METRICS,
        baseline_series=baseline_series,
        top_extra_frac=0.14 if criterion in PRF_METRICS else 0.0,
    )

    _plot_metric_lines(
        ax_qwk,
        x=x,
        series=series,
        metrics=("qwk",),
        show_legend=False,
        ylim=ylim_qwk,
        baseline_series=baseline_series,
        criterion=criterion,
        highlight_best=(criterion == "qwk"),
    )
    ax_qwk.set_title(PANEL_TITLES[criterion], fontsize=10, pad=6)
    if show_metric_ylabel:
        _set_metric_ylabel(ax_qwk, ("qwk",))
    else:
        ax_qwk.set_ylabel("")
    ax_qwk.set_xticks(x)
    ax_qwk.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    _decorate_experiment_groups(ax_qwk, labels_short)

    _plot_metric_lines(
        ax_prf,
        x=x,
        series=series,
        metrics=PRF_METRICS,
        show_legend=False,
        ylim=ylim_prf,
        baseline_series=baseline_series,
        criterion=criterion,
        highlight_best=(criterion in PRF_METRICS),
    )
    if show_metric_ylabel:
        _set_metric_ylabel(ax_prf, PRF_METRICS)
    else:
        ax_prf.set_ylabel("")
    ax_prf.set_xticks(x)
    ax_prf.set_xticklabels(labels_short, rotation=35, ha="right", fontsize=8)
    if show_xlabel_title:
        ax_prf.set_xlabel("Experiment", fontsize=9, labelpad=8)
    _decorate_experiment_groups(ax_prf, labels_short)

    ax_qwk.spines["bottom"].set_visible(False)
    ax_prf.spines["top"].set_visible(False)
    ax_qwk.tick_params(axis="x", which="both", bottom=False)
    return ax_qwk, ax_prf


def _metric_legend_handles():
    from matplotlib.lines import Line2D

    handles = []
    for metric in ("qwk", *PRF_METRICS):
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


def _figure_bottom_axis(fig):
    axes = fig.get_axes()
    if not axes:
        return None
    return min(axes, key=lambda ax: ax.get_position().y0)


def _label_bottom_figure_y(fig, *, renderer) -> float:
    ax_bottom = _figure_bottom_axis(fig)
    if ax_bottom is None:
        return fig.subplotpars.bottom * 0.75
    return ax_bottom.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y0


def fit_figure_legend_below_labels(fig, *, handles, ncol: int) -> None:
    """Place legend below rotated x tick labels without overlapping them."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap = 0.02

    leg = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, _label_bottom_figure_y(fig, renderer=renderer) - gap),
        bbox_transform=fig.transFigure,
        ncol=ncol,
        frameon=False,
        fontsize=11,
        markerscale=1.2,
        columnspacing=1.1,
        handletextpad=0.45,
        borderaxespad=0.0,
    )

    fig.canvas.draw()
    leg_bbox = leg.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    pad = 0.014
    if leg_bbox.y0 < pad:
        fig.subplots_adjust(bottom=fig.subplotpars.bottom + (pad - leg_bbox.y0))
        fig.canvas.draw()
        leg.set_bbox_to_anchor(
            (0.5, _label_bottom_figure_y(fig, renderer=renderer) - gap),
            transform=fig.transFigure,
        )
        fig.canvas.draw()


def _add_figure_legend_bottom(fig) -> None:
    fit_figure_legend_below_labels(fig, handles=_metric_legend_handles(), ncol=4)


def _render_profiles_figure(
    *,
    rows: list[dict[str, Any]],
    view: str,
):
    import matplotlib.pyplot as plt

    scale_break_pairs: list[tuple[Any, Any]] = []

    if view == "all":
        fig = plt.figure(figsize=(14, 12))
        outer_gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.26)
        panel_specs = [
            ("precision", 0, 0),
            ("recall", 0, 1),
            ("f1", 1, 0),
            ("qwk", 1, 1),
        ]
        for _criterion, row, col in panel_specs:
            pair = _plot_split_panel(
                fig,
                outer_gs[row, col],
                rows=rows,
                criterion=_criterion,
                show_xlabel_title=(col == 0),
                show_metric_ylabel=(col == 0),
                baseline_series=_baseline_reference_series(rows, _criterion),
            )
            if pair is not None:
                scale_break_pairs.append(pair)
        fig.subplots_adjust(left=0.09, right=0.99, top=0.94, bottom=0.13)
    else:
        fig = plt.figure(figsize=(12, 6.5))
        gs = fig.add_gridspec(1, 1)
        pair = _plot_split_panel(
            fig,
            gs[0, 0],
            rows=rows,
            criterion=view,
            show_xlabel_title=True,
            show_metric_ylabel=True,
            baseline_series=_baseline_reference_series(rows, view),
        )
        if pair is not None:
            scale_break_pairs.append(pair)
        fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.14)

    for ax_qwk, ax_prf in scale_break_pairs:
        _mark_scale_break(fig, ax_qwk, ax_prf)
    _add_figure_legend_bottom(fig)
    return fig


def render_metric_optimal_profiles_section(*, log_files: list[str]) -> None:
    import matplotlib.pyplot as plt

    st.markdown("### Performance profile at QWK-optimal checkpoints")

    rows = collect_metric_optimal_profile_rows(log_files)
    if not rows:
        st.info("No parseable experiment logs found for metric-optimal profiles.")
        return

    view = "qwk"  # default; radio hidden for now
    # view = st.radio(
    #     "Checkpoint selection criterion",
    #     options=list(CHECKPOINT_VIEW_ORDER),
    #     format_func=lambda key: CHECKPOINT_VIEW_LABELS[key],
    #     index=0,
    #     horizontal=True,
    #     key="metric_optimal_profile_view",
    # )
    #
    # if view == "all":
    #     st.caption(
    #         "Four panels: full metric profile at each criterion's optimal checkpoint. "
    #         "Precision, recall, and F1 are support-weighted averages. "
    #         "Dashed lines = baseline; ∥ = scale break. Y-axes zoom to each panel's data."
    #     )
    # else:
    st.caption(
        f"{PANEL_TITLES[view]}. All metrics at that checkpoint "
        "(P/R/F1 are support-weighted); dashed lines = baseline; "
        "∥ = scale break between QWK and weighted P/R/F1. "
        "Y-axes zoom to the plotted values."
    )

    fig = _render_profiles_figure(rows=rows, view=view)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    table_criteria = CRITERION_KEYS if view == "all" else (view,)
    with st.expander("Numeric tables"):
        for criterion in table_criteria:
            table_df = build_profile_table_df(rows, criterion)
            if table_df.empty:
                continue
            st.caption(PANEL_TITLES[criterion])
            st.dataframe(table_df, width="stretch", hide_index=True)

    from tarab_model_experimentation.accuracy_distance_profiles import (
        render_accuracy_distance_profiles_section,
    )

    render_accuracy_distance_profiles_section(log_files=log_files)
