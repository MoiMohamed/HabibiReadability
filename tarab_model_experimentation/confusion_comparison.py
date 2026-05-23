from __future__ import annotations

from typing import Any

import streamlit as st

from tarab_model_experimentation.log_parsing import parse_training_log
from tarab_model_experimentation.constants import DIST_ORANGE
from tarab_model_experimentation.metrics import (
    compute_confusion_deviation_stats,
    per_class_accuracy_series,
    per_class_precision_series,
    predicted_class_distribution,
    signed_error_count_series,
    true_class_distribution,
)
from tarab_model_experimentation.selection import (
    experiment_chart_label,
    preferred_log_for_chart_label,
)

_BASELINE_COLOR = "#1f77b4"
_EXPERIMENT_COLORS = {
    "dist_155K": DIST_ORANGE,
    # "dist_155K_wo_19": "#2ca02c",
}

COMPARISON_SPECS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("Prediction analysis", ("baseline", "dist_155K")),
    # (
    #     "Prediction analysis: baseline vs dist_155K_wo_19",
    #     ("baseline", "dist_155K_wo_19"),
    # ),
)


def log_file_for_chart_label(log_files: list[str], label: str) -> str | None:
    return preferred_log_for_chart_label(log_files, label)


def best_qwk_epoch_from_log(log_filename: str) -> dict[str, Any] | None:
    data = parse_training_log(log_filename)
    epochs = data.get("epochs") or []
    if not epochs:
        return None
    return max(epochs, key=lambda epoch: float(epoch["qwk"]))


def _load_comparison_epochs(
    log_files: list[str], compare_labels: tuple[str, str]
) -> dict[str, dict[str, Any]] | None:
    loaded: dict[str, dict[str, Any]] = {}
    for label in compare_labels:
        log_file = log_file_for_chart_label(log_files, label)
        if log_file is None:
            return None
        epoch = best_qwk_epoch_from_log(log_file)
        if epoch is None or epoch.get("cm_df") is None:
            return None
        loaded[label] = {
            "log_file": log_file,
            "epoch": float(epoch["epoch"]),
            "qwk": float(epoch["qwk"]),
            "cm_df": epoch["cm_df"],
        }
    return loaded


def _color_for_label(label: str) -> str:
    if label == "baseline":
        return _BASELINE_COLOR
    return _EXPERIMENT_COLORS.get(label, "#9467bd")


def _plot_predicted_distribution(loaded: dict[str, dict[str, Any]], compare_labels: tuple[str, str]):
    import matplotlib.pyplot as plt
    import numpy as np

    series = {
        label: predicted_class_distribution(entry["cm_df"])
        for label, entry in loaded.items()
    }
    levels = sorted(set().union(*(s.index for s in series.values())))
    x = np.arange(len(levels))
    width = 0.38

    gold = true_class_distribution(loaded[compare_labels[0]]["cm_df"])
    gold_counts = np.array([gold.get(level, 0) for level in levels], dtype=float)
    gold_pct = 100.0 * gold_counts / gold_counts.sum() if gold_counts.sum() else gold_counts

    fig, ax = plt.subplots(figsize=(14, 4.5))
    for idx, label in enumerate(compare_labels):
        dist = series[label]
        counts = np.array([dist.get(level, 0) for level in levels], dtype=float)
        pct = 100.0 * counts / counts.sum() if counts.sum() else counts
        offset = (idx - 0.5) * width
        ax.bar(
            x + offset,
            pct,
            width=width,
            color=_color_for_label(label),
            label=f"{label} (epoch {loaded[label]['epoch']:.0f})",
        )

    ax.plot(
        x,
        gold_pct,
        color="#444444",
        linestyle="--",
        linewidth=1.2,
        marker="D",
        markersize=3,
        label="dev (true)",
        zorder=4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(level) for level in levels], rotation=0)
    ax.set_xlabel("Predicted readability level")
    ax.set_ylabel("% of predictions")
    ax.set_title("Predicted class distribution (dev set, best QWK checkpoint)")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _plot_per_class_metric_panels(
    ax_bars,
    ax_delta,
    *,
    series: dict,
    levels: list,
    compare_labels: tuple[str, str],
    loaded: dict[str, dict[str, Any]],
    ylabel: str,
    delta_ylabel: str,
):
    import numpy as np

    baseline_label, experiment_label = compare_labels
    x = np.arange(len(levels))
    width = 0.38

    for idx, label in enumerate(compare_labels):
        vals = np.array([series[label].get(level, np.nan) for level in levels], dtype=float)
        offset = (idx - 0.5) * width
        ax_bars.bar(
            x + offset,
            vals,
            width=width,
            color=_color_for_label(label),
            label=f"{label} (epoch {loaded[label]['epoch']:.0f})",
            edgecolor="white",
            linewidth=0.4,
        )

    baseline_vals = np.array([series[baseline_label].get(level, np.nan) for level in levels])
    experiment_vals = np.array([series[experiment_label].get(level, np.nan) for level in levels])
    delta = experiment_vals - baseline_vals

    ax_bars.set_ylim(0, 1.05)
    ax_bars.set_ylabel(ylabel)
    ax_bars.grid(True, axis="y", alpha=0.25)
    ax_bars.axhline(0.5, color="#999999", linewidth=0.6, linestyle=":", alpha=0.6)

    bar_colors = np.where(delta >= 0, "#2ca02c", "#d62728")
    ax_delta.bar(x, delta, width=0.72, color=bar_colors, edgecolor="white", linewidth=0.4)
    ax_delta.axhline(0, color="#444444", linewidth=0.8)
    ax_delta.set_ylabel(delta_ylabel)
    ax_delta.grid(True, axis="y", alpha=0.25)
    if np.any(~np.isnan(delta)):
        dmin, dmax = float(np.nanmin(delta)), float(np.nanmax(delta))
        pad = max(0.05, 0.1 * (dmax - dmin))
        ax_delta.set_ylim(min(-0.35, dmin - pad), max(0.35, dmax + pad))


def _plot_per_class_accuracy_comparison(
    loaded: dict[str, dict[str, Any]], compare_labels: tuple[str, str]
):
    import matplotlib.pyplot as plt

    baseline_label, experiment_label = compare_labels
    short_exp = experiment_label.replace("dist_", "d").replace("_", " ")
    recall_series = {
        label: per_class_accuracy_series(entry["cm_df"])
        for label, entry in loaded.items()
    }
    precision_series = {
        label: per_class_precision_series(entry["cm_df"])
        for label, entry in loaded.items()
    }
    levels = sorted(set().union(*(s.index for s in recall_series.values())))
    x_labels = [str(level) for level in levels]

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 9.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1, 2.2, 1], "hspace": 0.06},
    )
    ax_recall, ax_d_recall, ax_prec, ax_d_prec = axes

    _plot_per_class_metric_panels(
        ax_recall,
        ax_d_recall,
        series=recall_series,
        levels=levels,
        compare_labels=compare_labels,
        loaded=loaded,
        ylabel="Recall (per true level)",
        delta_ylabel=f"Δ recall\n({short_exp} − base)",
    )
    _plot_per_class_metric_panels(
        ax_prec,
        ax_d_prec,
        series=precision_series,
        levels=levels,
        compare_labels=compare_labels,
        loaded=loaded,
        ylabel="Precision (per predicted level)",
        delta_ylabel=f"Δ precision\n({short_exp} − base)",
    )

    ax_recall.set_title("Per-class recall & precision (dev set, best QWK checkpoint)")
    ax_recall.legend(loc="upper right", frameon=False, ncol=2)
    ax_d_prec.set_xlabel("Readability level")
    for ax in axes:
        ax.set_xticks(range(len(levels)))
    ax_d_prec.set_xticklabels(x_labels)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.08, hspace=0.14)
    return fig


def build_confusion_matrix_delta(
    loaded: dict[str, dict[str, Any]], compare_labels: tuple[str, str]
):
    """ΔC = C_dist − C_baseline with 1-based level labels."""
    import numpy as np
    import pandas as pd

    baseline_label, experiment_label = compare_labels
    base_cm = loaded[baseline_label]["cm_df"].to_numpy(dtype=np.int64)
    exp_cm = loaded[experiment_label]["cm_df"].to_numpy(dtype=np.int64)
    labels = [int(x) + 1 for x in loaded[baseline_label]["cm_df"].index]
    delta = exp_cm - base_cm
    return pd.DataFrame(delta, index=labels, columns=labels)


def _confusion_delta_magnitude(delta_df):
    import numpy as np

    arr = delta_df.to_numpy(dtype=float)
    limit = max(1.0, float(np.max(np.abs(arr))))
    return limit


def _plot_confusion_delta_heatmap(
    delta_df,
    *,
    title: str,
    figsize: tuple[float, float] = (10, 8.5),
):
    import matplotlib.pyplot as plt
    import numpy as np

    data = delta_df.to_numpy(dtype=float)
    labels = [str(x) for x in delta_df.index]
    limit = _confusion_delta_magnitude(delta_df)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        origin="lower",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted level")
    ax.set_ylabel("True level")
    ax.set_title(title)

    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            v = int(data[r, c])
            if v == 0:
                continue
            color = "white" if abs(v) > limit * 0.55 else "black"
            ax.text(c, r, f"{v:+d}", ha="center", va="center", fontsize=6, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ count (dist_155K − baseline)")
    fig.tight_layout()
    return fig


def _render_confusion_delta_heatmap(
    loaded: dict[str, dict[str, Any]], compare_labels: tuple[str, str]
) -> None:
    import matplotlib.pyplot as plt

    _, experiment_label = compare_labels
    short_exp = experiment_label.replace("dist_", "d").replace("_", " ")
    delta_df = build_confusion_matrix_delta(loaded, compare_labels)
    fig = _plot_confusion_delta_heatmap(
        delta_df,
        title=f"ΔC = {short_exp} − baseline (dev, best QWK)",
    )
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _delta_metric_by_level(
    loaded: dict[str, dict[str, Any]],
    compare_labels: tuple[str, str],
    *,
    metric: str,
) -> dict[int, float]:
    import numpy as np

    baseline_label, experiment_label = compare_labels
    if metric == "precision":
        base = per_class_precision_series(loaded[baseline_label]["cm_df"])
        exp = per_class_precision_series(loaded[experiment_label]["cm_df"])
    else:
        base = per_class_accuracy_series(loaded[baseline_label]["cm_df"])
        exp = per_class_accuracy_series(loaded[experiment_label]["cm_df"])
    levels = sorted(set(base.index) | set(exp.index))
    return {
        int(lev): float(exp.get(lev, np.nan)) - float(base.get(lev, np.nan))
        for lev in levels
    }


def build_distillation_relationship_df(
    loaded: dict[str, dict[str, Any]], compare_labels: tuple[str, str]
):
    """All levels: Δ precision, Tarab median conf, dev gold support, dist training share."""
    import numpy as np
    import pandas as pd

    from tarab_model_experimentation.class19_investigation import (
        DIST_155K_SPLIT_CSV,
        LEVELS,
        _load_split_pretrain_df,
        _tarab_confidence_stats_by_level,
    )

    baseline_label, _experiment_label = compare_labels
    dist_df = _load_split_pretrain_df(str(DIST_155K_SPLIT_CSV))
    if dist_df is None:
        return None

    conf = _tarab_confidence_stats_by_level(dist_df)
    median_pct = (conf["median"] * 100.0).reindex(LEVELS)
    support = true_class_distribution(loaded[baseline_label]["cm_df"]).reindex(LEVELS)
    delta_prec = _delta_metric_by_level(loaded, compare_labels, metric="precision")
    delta_rec = _delta_metric_by_level(loaded, compare_labels, metric="recall")
    base_prec = per_class_precision_series(loaded[baseline_label]["cm_df"])
    base_rec = per_class_accuracy_series(loaded[baseline_label]["cm_df"])
    class_share = _dist_155k_class_share_pct_by_level() or {}

    rows: list[dict[str, Any]] = []
    for lev in LEVELS:
        m = median_pct.loc[lev]
        d_prec = delta_prec.get(lev, np.nan)
        d_rec = delta_rec.get(lev, np.nan)
        bp = base_prec.get(lev, np.nan)
        br = base_rec.get(lev, np.nan)
        sup = support.loc[lev] if lev in support.index else np.nan
        if np.isnan(m) or np.isnan(d_prec) or np.isnan(d_rec) or np.isnan(bp) or np.isnan(br) or np.isnan(sup):
            continue
        rows.append(
            {
                "level": int(lev),
                "confidence_pct": float(m),
                "delta_recall": float(d_rec),
                "delta_precision": float(d_prec),
                "baseline_recall": float(br),
                "baseline_precision": float(bp),
                "support": int(sup),
                "class_share_%": float(class_share.get(lev, 0.0)),
            }
        )
    if len(rows) < 2:
        return None
    return pd.DataFrame(rows)


def _dist_155k_class_share_pct_by_level() -> dict[int, float] | None:
    from tarab_model_experimentation.class19_investigation import (
        DIST_155K_SPLIT_CSV,
        LEVELS,
        _label_distribution,
        _load_split_pretrain_df,
    )

    dist_df = _load_split_pretrain_df(str(DIST_155K_SPLIT_CSV))
    if dist_df is None:
        return None
    counts = _label_distribution(dist_df)
    total = int(counts.sum())
    if total <= 0:
        return None
    return {int(lev): 100.0 * int(counts.loc[lev]) / total for lev in LEVELS}


def _bubble_sizes(support_series, *, ref: float | None = None) -> list[float]:
    import numpy as np

    ref_max = float(ref if ref is not None else support_series.max())
    if ref_max <= 0:
        return [120.0] * len(support_series)
    scaled = np.sqrt(support_series.astype(float) / ref_max)
    return (scaled * 900.0 + 80.0).tolist()


def _spearman_rho(x, y) -> float | None:
    """Spearman ρ via ranked Pearson (no scipy — Streamlit Cloud has pandas only)."""
    import pandas as pd

    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3:
        return None
    rx = pair["x"].rank(method="average")
    ry = pair["y"].rank(method="average")
    rho = rx.corr(ry, method="pearson")
    return None if pd.isna(rho) else float(rho)


def _pad_scatter_limits(ax, df, *, x_col: str, y_col: str, pad_frac: float = 0.12) -> None:
    x = df[x_col].astype(float)
    y = df[y_col].astype(float)
    x_span = max(float(x.max() - x.min()), 1.0)
    y_span = max(float(y.max() - y.min()), 0.05)
    pad_x = pad_frac * x_span
    pad_y = pad_frac * y_span
    ax.set_xlim(float(x.min()) - pad_x, float(x.max()) + pad_x)
    ax.set_ylim(float(y.min()) - pad_y, float(y.max()) + pad_y)


def _annotate_spearman_rho(ax, rho: float, *, n: int | None = None) -> None:
    label = f"Spearman ρ = {rho:+.2f}" if n is None else f"ρ = {rho:+.2f} (n={n})"
    ax.text(
        0.98,
        0.04,
        label,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
    )


def _annotate_bubble_levels(ax, df, *, x_col: str, y_col: str) -> None:
    for _, row in df.iterrows():
        ax.annotate(
            f"L{int(row['level'])}",
            (row[x_col], row[y_col]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="#333333",
        )


def _plot_confidence_vs_delta_precision_bubble(df):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9.5, 6))
    colors = np.where(df["delta_precision"] >= 0, "#3a9e6e", "#c94040")
    sizes = _bubble_sizes(df["support"])
    ax.scatter(
        df["confidence_pct"],
        df["delta_precision"],
        s=sizes,
        c=colors,
        alpha=0.72,
        edgecolors="white",
        linewidths=0.6,
    )
    _annotate_bubble_levels(ax, df, x_col="confidence_pct", y_col="delta_precision")
    _pad_scatter_limits(ax, df, x_col="confidence_pct", y_col="delta_precision")
    ax.axhline(0, color="#888888", linewidth=0.7, linestyle="--", alpha=0.7)
    ax.set_xlabel("Median Tarab pseudo-label confidence (%)")
    ax.set_ylabel("Δ precision (dist_155K − baseline)")
    ax.set_title("Confidence × support × precision change (all 19 levels)")
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    rho = _spearman_rho(df["confidence_pct"], df["delta_precision"])
    if rho is not None:
        _annotate_spearman_rho(ax, rho)
    fig.tight_layout()
    return fig


def _dev_support_stats_line(df) -> tuple[float, str]:
    med = float(df["support"].median())
    med_label = f"{med:.0f}" if med == int(med) else f"{med:.1f}"
    sup_min, sup_max = int(df["support"].min()), int(df["support"].max())
    stats_line = f"Dev gold count: min {sup_min}, median {med_label}, max {sup_max}"
    return med, stats_line


def _plot_confidence_vs_delta_precision_faceted(df):
    import matplotlib.pyplot as plt
    import numpy as np

    med, stats_line = _dev_support_stats_line(df)
    panel_titles = ("High dev support (≥ median)", "Low dev support (< median)")
    parts = [df[df["support"] >= med], df[df["support"] < med]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
    ref_max = float(df["support"].max())

    y = df["delta_precision"].astype(float)
    y_span = max(float(y.max() - y.min()), 0.05)
    pad_y = 0.12 * y_span
    ylim = (float(y.min()) - pad_y, float(y.max()) + pad_y)

    for ax, title, part in zip(axes, panel_titles, parts):
        ax.set_title(title)
        if part.empty:
            continue
        colors = np.where(part["delta_precision"] >= 0, "#3a9e6e", "#c94040")
        ax.scatter(
            part["confidence_pct"],
            part["delta_precision"],
            s=_bubble_sizes(part["support"], ref=ref_max),
            c=colors,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.6,
        )
        _annotate_bubble_levels(ax, part, x_col="confidence_pct", y_col="delta_precision")
        x = part["confidence_pct"].astype(float)
        x_span = max(float(x.max() - x.min()), 1.0)
        pad_x = 0.12 * x_span
        ax.set_xlim(float(x.min()) - pad_x, float(x.max()) + pad_x)
        ax.set_ylim(ylim)
        ax.axhline(0, color="#888888", linewidth=0.7, linestyle="--", alpha=0.7)
        ax.set_xlabel("Median Tarab confidence (%)")
        ax.grid(True, alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        rho = _spearman_rho(part["confidence_pct"], part["delta_precision"])
        if rho is not None:
            _annotate_spearman_rho(ax, rho, n=len(part))

    axes[0].set_ylabel("Δ precision (dist_155K − baseline)")
    fig.suptitle(
        f"Confidence vs Δ precision by dev support\n{stats_line}",
        y=1.04,
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def _distillation_relationship_df_or_warn(
    loaded: dict[str, dict[str, Any]], compare_labels: tuple[str, str]
):
    df = build_distillation_relationship_df(loaded, compare_labels)
    if df is None:
        st.info("Distillation relationship plot unavailable (missing split or metrics).")
    return df


def _render_distillation_confidence_bubbles(
    loaded: dict[str, dict[str, Any]], compare_labels: tuple[str, str]
) -> None:
    import matplotlib.pyplot as plt

    df = _distillation_relationship_df_or_warn(loaded, compare_labels)
    if df is None:
        return

    st.markdown("**Distillation: confidence, support & Δ precision**")
    st.caption(
        "Bubble size = dev gold count per level. Green = precision gain, red = loss. "
        "All 19 levels shown."
    )

    fig1 = _plot_confidence_vs_delta_precision_bubble(df)
    st.pyplot(fig1, clear_figure=True)
    plt.close(fig1)

    fig2 = _plot_confidence_vs_delta_precision_faceted(df)
    st.pyplot(fig2, clear_figure=True)
    plt.close(fig2)
    _, stats_line = _dev_support_stats_line(df)
    st.caption(
        f"{stats_line}. Split at median dev gold count; bubble size = support. "
        "Association may be stronger within one group than globally."
    )


def _render_per_class_recall_precision_table(
    loaded: dict[str, dict[str, Any]], compare_labels: tuple[str, str]
) -> None:
    df = build_distillation_relationship_df(loaded, compare_labels)
    if df is None:
        return

    with st.expander("Numeric tables (per-class recall & precision)"):
        show = df.sort_values("level").rename(
            columns={
                "level": "level",
                "delta_recall": "Δ recall",
                "delta_precision": "Δ precision",
                "baseline_recall": "baseline recall",
                "baseline_precision": "baseline precision",
                "confidence_pct": "tarab conf median %",
                "support": "dev gold count",
                "class_share_%": "class share %",
            }
        )
        st.dataframe(show, width="stretch", hide_index=True)


def _annotate_dist_count_deltas(
    ax,
    x_positions,
    dist_heights,
    deltas,
    *,
    x_offset: float,
) -> None:
    for xi, height, delta in zip(x_positions, dist_heights, deltas):
        if delta == 0:
            continue
        color = "#3a9e6e" if delta < 0 else "#c94040"
        ax.annotate(
            f"{int(delta):+d}",
            xy=(xi + x_offset, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=color,
        )


def _plot_signed_error_histogram(loaded: dict[str, dict[str, Any]], compare_labels: tuple[str, str]):
    import matplotlib.pyplot as plt
    import numpy as np

    baseline_label, experiment_label = compare_labels
    series = {
        label: signed_error_count_series(entry["cm_df"])
        for label, entry in loaded.items()
    }
    errors = sorted(set().union(*(s.index for s in series.values())))
    x = np.array(errors, dtype=int)

    base_counts = np.array([series[baseline_label].get(err, 0) for err in errors], dtype=float)
    dist_counts = np.array([series[experiment_label].get(err, 0) for err in errors], dtype=float)
    total = float(base_counts.sum()) if base_counts.sum() else 1.0
    base_pct = 100.0 * base_counts / total
    dist_pct = 100.0 * dist_counts / total
    delta_counts = dist_counts - base_counts

    fig, ax = plt.subplots(figsize=(12, 5.8))
    width = 0.38
    dist_offset = width / 2
    for idx, label in enumerate(compare_labels):
        counts = base_counts if label == baseline_label else dist_counts
        pct = base_pct if label == baseline_label else dist_pct
        offset = (idx - 0.5) * width
        ax.bar(
            x + offset,
            pct,
            width=width,
            label=f"{label} (epoch {loaded[label]['epoch']:.0f})",
            color=_color_for_label(label),
            alpha=0.85,
        )

    _annotate_dist_count_deltas(ax, x, dist_pct, delta_counts, x_offset=dist_offset)

    ax.axvline(0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xlabel("Prediction − true (levels)")
    ax.set_ylabel("% of dev set")
    ax.set_title("Signed error distribution")
    ax.set_xlim(-12, 12)
    mistake_pct = dist_pct[np.array([int(e) != 0 for e in errors])]
    zero_pct = float(np.max(dist_pct[x == 0])) if np.any(x == 0) else 0.0
    y_top = max(28.0, float(mistake_pct.max()) * 1.35, zero_pct * 1.06) if len(mistake_pct) else max(28.0, zero_pct * 1.06)
    ax.set_ylim(0, y_top)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _distance_metrics_from_cm(cm_df) -> dict[str, float]:
    stats = compute_confusion_deviation_stats(cm_df)
    summary = {r["metric"]: float(r["value"]) for _, r in stats["summary_df"].iterrows()}
    exact = summary["exact_match_rate"]
    within_1 = summary["within_1_rate"]
    within_2 = summary["within_2_rate"]
    return {
        "pct_exact": 100.0 * exact,
        "pct_abs_1": 100.0 * (within_1 - exact),
        "pct_abs_2": 100.0 * (within_2 - within_1),
        "pct_abs_ge_3": 100.0 * (1.0 - within_2),
        "near_pct": 100.0 * within_1,
        "tail_pct": 100.0 * (1.0 - within_2),
        "mae": summary["mae_levels"],
        "signed_bias": summary["signed_bias_levels"],
    }


def build_distance_decomposition_table(
    loaded: dict[str, dict[str, Any]], compare_labels: tuple[str, str]
):
    import pandas as pd

    baseline_label, experiment_label = compare_labels
    rows: list[dict[str, Any]] = []
    for label in compare_labels:
        m = _distance_metrics_from_cm(loaded[label]["cm_df"])
        rows.append(
            {
                "model": f"{label} (ep {loaded[label]['epoch']:.0f})",
                "pct_|err|=0": round(m["pct_exact"], 2),
                "pct_|err|=1": round(m["pct_abs_1"], 2),
                "pct_|err|=2": round(m["pct_abs_2"], 2),
                "pct_|err|>=3": round(m["pct_abs_ge_3"], 2),
                "near_% (|err|<=1)": round(m["near_pct"], 2),
                "tail_% (|err|>=3)": round(m["tail_pct"], 2),
                "MAE (levels)": round(m["mae"], 3),
                "signed_bias": round(m["signed_bias"], 3),
            }
        )
    base_m = _distance_metrics_from_cm(loaded[baseline_label]["cm_df"])
    exp_m = _distance_metrics_from_cm(loaded[experiment_label]["cm_df"])
    short_exp = experiment_label.replace("dist_", "d").replace("_", " ")
    rows.append(
        {
            "model": f"Δ ({short_exp} − {baseline_label})",
            "pct_|err|=0": round(exp_m["pct_exact"] - base_m["pct_exact"], 2),
            "pct_|err|=1": round(exp_m["pct_abs_1"] - base_m["pct_abs_1"], 2),
            "pct_|err|=2": round(exp_m["pct_abs_2"] - base_m["pct_abs_2"], 2),
            "pct_|err|>=3": round(exp_m["pct_abs_ge_3"] - base_m["pct_abs_ge_3"], 2),
            "near_% (|err|<=1)": round(exp_m["near_pct"] - base_m["near_pct"], 2),
            "tail_% (|err|>=3)": round(exp_m["tail_pct"] - base_m["tail_pct"], 2),
            "MAE (levels)": round(exp_m["mae"] - base_m["mae"], 3),
            "signed_bias": round(exp_m["signed_bias"] - base_m["signed_bias"], 3),
        }
    )
    return pd.DataFrame(rows)


def _shift_profile_narrative(
    loaded: dict[str, dict[str, Any]], compare_labels: tuple[str, str]
) -> str:
    baseline_label, experiment_label = compare_labels
    base_m = _distance_metrics_from_cm(loaded[baseline_label]["cm_df"])
    exp_m = _distance_metrics_from_cm(loaded[experiment_label]["cm_df"])
    parts: list[str] = []

    d_exact = exp_m["pct_exact"] - base_m["pct_exact"]
    if d_exact > 0.2:
        parts.append(f"more mass at exact match (+{d_exact:.1f} pp at |err|=0)")
    elif d_exact < -0.2:
        parts.append(f"less exact match ({d_exact:.1f} pp at |err|=0)")

    d_tail = exp_m["tail_pct"] - base_m["tail_pct"]
    if d_tail < -0.2:
        parts.append(f"long tail shrank (|err|≥3: {d_tail:+.1f} pp)")
    elif d_tail > 0.2:
        parts.append(f"long tail grew (|err|≥3: {d_tail:+.1f} pp)")

    d_near = exp_m["near_pct"] - base_m["near_pct"]
    if abs(d_near) > 0.2:
        parts.append(f"near errors |err|≤1: {d_near:+.1f} pp")

    d_bias = exp_m["signed_bias"] - base_m["signed_bias"]
    if abs(d_bias) > 0.02:
        direction = "over-prediction" if d_bias > 0 else "under-prediction"
        parts.append(f"signed bias shifted toward {direction} ({d_bias:+.3f} levels)")

    if not parts:
        return "Signed-error profile is largely unchanged between models at this checkpoint."
    return " ".join(parts).capitalize() + "."


def build_per_level_shift_table(
    loaded: dict[str, dict[str, Any]], compare_labels: tuple[str, str]
):
    import pandas as pd

    baseline_label, experiment_label = compare_labels
    base_stats = compute_confusion_deviation_stats(loaded[baseline_label]["cm_df"])
    exp_stats = compute_confusion_deviation_stats(loaded[experiment_label]["cm_df"])
    base_pt = base_stats["per_true_summary_df"].set_index("true_level")
    exp_pt = exp_stats["per_true_summary_df"].set_index("true_level")
    levels = sorted(set(base_pt.index) | set(exp_pt.index))

    rows: list[dict[str, Any]] = []
    for lev in levels:
        b = base_pt.loc[lev] if lev in base_pt.index else None
        e = exp_pt.loc[lev] if lev in exp_pt.index else None
        if b is None or e is None:
            continue
        tail_b = 1.0 - float(b["within_2_rate"])
        tail_e = 1.0 - float(e["within_2_rate"])
        rows.append(
            {
                "level": int(lev),
                "support": int(b["support"]),
                "mean_signed_error_base": round(float(b["mean_pred_minus_true"]), 3),
                "mean_signed_error_dist": round(float(e["mean_pred_minus_true"]), 3),
                "Δ_signed_error": round(
                    float(e["mean_pred_minus_true"]) - float(b["mean_pred_minus_true"]), 3
                ),
                "MAE_base": round(float(b["mean_abs_error"]), 3),
                "MAE_dist": round(float(e["mean_abs_error"]), 3),
                "Δ_MAE": round(float(e["mean_abs_error"]) - float(b["mean_abs_error"]), 3),
                "pct_|err|>=2_base": round(100.0 * tail_b, 2),
                "pct_|err|>=2_dist": round(100.0 * tail_e, 2),
                "Δ_pct_|err|>=2": round(100.0 * (tail_e - tail_b), 2),
            }
        )
    return pd.DataFrame(rows)


def _top_levels_by_delta_mae(per_level_df, *, n: int = 4, improved: bool):
    import pandas as pd

    if per_level_df is None or per_level_df.empty:
        return pd.DataFrame()
    ranked = per_level_df.sort_values("Δ_MAE", ascending=improved)
    return ranked.head(n)


def _error_bucket_deltas(loaded: dict[str, dict[str, Any]], compare_labels: tuple[str, str]):
    import pandas as pd

    baseline_label, experiment_label = compare_labels
    rows: list[dict[str, Any]] = []
    for label in (baseline_label, experiment_label):
        stats = compute_confusion_deviation_stats(loaded[label]["cm_df"])
        dist = stats["distance_df"].set_index("|pred-true|")
        total = int(
            stats["summary_df"]
            .loc[stats["summary_df"]["metric"] == "total_samples", "value"]
            .iloc[0]
        )
        for abs_err, row in dist.iterrows():
            d = int(abs_err)
            count = int(row["count"])
            rows.append(
                {
                    "model": label,
                    "abs_error": d,
                    "count": count,
                    "pct": 100.0 * count / total if total else 0.0,
                    "sq_mass": count * d * d,
                }
            )
    long_df = pd.DataFrame(rows)
    base = long_df[long_df["model"] == baseline_label].set_index("abs_error")
    exp = long_df[long_df["model"] == experiment_label].set_index("abs_error")
    all_errs = sorted(set(base.index) | set(exp.index))
    delta_rows = []
    for d in all_errs:
        bc = int(base.loc[d, "count"]) if d in base.index else 0
        ec = int(exp.loc[d, "count"]) if d in exp.index else 0
        bsq = int(base.loc[d, "sq_mass"]) if d in base.index else 0
        esq = int(exp.loc[d, "sq_mass"]) if d in exp.index else 0
        delta_rows.append(
            {
                "abs_error": d,
                "delta_count": ec - bc,
                "delta_sq_mass": esq - bsq,
                "baseline_count": bc,
                "dist_count": ec,
            }
        )
    return pd.DataFrame(delta_rows), int(
        long_df.loc[long_df["model"] == baseline_label, "count"].sum()
    )


def build_per_level_sq_error_contribution(
    loaded: dict[str, dict[str, Any]], compare_labels: tuple[str, str]
):
    """Per-class actual squared-error penalty reduction (baseline − dist)."""
    from tarab_model_experimentation.dev_predictions import (
        build_per_class_qwk_penalty_contribution,
    )

    contrib, err = build_per_class_qwk_penalty_contribution(compare_labels)
    if err or contrib is None:
        return pd.DataFrame()
    if contrib.empty:
        return contrib
    out = contrib.rename(columns={"delta_penalty": "weighted_delta_sq"})
    out["delta_sq_mass"] = out["delta_penalty"]
    return out


def build_far_off_mistake_sanity_table(
    loaded: dict[str, dict[str, Any]], compare_labels: tuple[str, str], *, min_distance: int = 7, top_n: int = 20
):
    import numpy as np
    import pandas as pd

    baseline_label, experiment_label = compare_labels
    base_cm = loaded[baseline_label]["cm_df"].to_numpy(dtype=np.int64)
    exp_cm = loaded[experiment_label]["cm_df"].to_numpy(dtype=np.int64)
    diff = exp_cm - base_cm
    n = base_cm.shape[0]
    level_labels = [int(x) + 1 for x in loaded[baseline_label]["cm_df"].index]

    rows: list[dict[str, Any]] = []
    for r in range(n):
        for c in range(n):
            d = abs(c - r)
            if d < min_distance:
                continue
            delta = int(diff[r, c])
            if delta <= 0:
                continue
            rows.append(
                {
                    "true_level": level_labels[r],
                    "pred_level": level_labels[c],
                    "levels_off": d,
                    "baseline_count": int(base_cm[r, c]),
                    "dist_count": int(exp_cm[r, c]),
                    "new_mistakes": delta,
                    "qwk_weight_per_mistake": d * d,
                    "added_penalty_units": delta * d * d,
                }
            )
    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(
        ["added_penalty_units", "new_mistakes"], ascending=False
    )
    return out.head(top_n)


def _plot_mistake_rate_by_distance(
    loaded: dict[str, dict[str, Any]], compare_labels: tuple[str, str]
):
    import matplotlib.pyplot as plt
    import numpy as np

    baseline_label, experiment_label = compare_labels
    bucket_df, _total = _error_bucket_deltas(loaded, compare_labels)
    if bucket_df.empty:
        return None

    active = bucket_df["baseline_count"] + bucket_df["dist_count"] > 0
    plot_df = bucket_df[active].copy()
    if plot_df.empty:
        plot_df = bucket_df.copy()
    x = plot_df["abs_error"].to_numpy()
    base_pct = 100.0 * plot_df["baseline_count"].to_numpy() / _total
    dist_pct = 100.0 * plot_df["dist_count"].to_numpy() / _total
    width = 0.36

    fig_w = max(11.0, 0.55 * len(x) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    ax.bar(x - width / 2, base_pct, width=width, color=_BASELINE_COLOR, label="baseline", alpha=0.9)
    ax.bar(
        x + width / 2,
        dist_pct,
        width=width,
        color=_color_for_label(experiment_label),
        label=experiment_label.replace("dist_", "d").replace("_", " "),
        alpha=0.9,
    )
    _annotate_dist_count_deltas(
        ax,
        x,
        dist_pct,
        plot_df["delta_count"].to_numpy(),
        x_offset=width / 2,
    )
    ax.set_xlabel("How far off was the prediction? (0 = correct)")
    ax.set_ylabel("% of dev set")
    ax.set_title("Unsigned error distance")
    ax.set_xticks(x)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _plot_per_class_gain_loss_bars(df):
    import matplotlib.pyplot as plt
    import numpy as np

    plot_df = df.sort_values("level")
    x = np.arange(len(plot_df))
    gain = plot_df["gain_positive"].to_numpy(dtype=float)
    loss = plot_df["loss_negative"].to_numpy(dtype=float)
    net = plot_df["delta_penalty"].to_numpy(dtype=float)
    x_labels = [str(int(r["level"])) for _, r in plot_df.iterrows()]
    width = 0.72

    fig, ax = plt.subplots(figsize=(14, 6.0))
    ax.bar(x, gain, color="#3a9e6e", edgecolor="white", linewidth=0.4, width=width, alpha=0.92)
    ax.bar(x, loss, color="#c94040", edgecolor="white", linewidth=0.4, width=width, alpha=0.92)
    ax.plot(
        x,
        net,
        linestyle="none",
        marker="D",
        markersize=5,
        color="#1a1a1a",
        markeredgecolor="white",
        markeredgewidth=0.6,
        zorder=4,
        label="Net (gain + loss)",
    )
    ax.axhline(0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.7)

    y_span = max(gain.max(initial=0.0), abs(loss.min(initial=0.0)), abs(net).max(initial=0.0))
    label_pad = 0.06 * y_span if y_span > 0 else 0.06
    for xi, n in zip(x, net):
        if abs(n) < 1e-6:
            continue
        ax.text(
            xi,
            n + (label_pad if n >= 0 else -label_pad),
            f"{n:+.2f}",
            ha="center",
            va="bottom" if n >= 0 else "top",
            fontsize=7,
            color="#222222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Dev true readability level")
    ax.set_ylabel("Δ squared-error penalty (within true level)")
    ax.set_title(
        "dist_155K vs baseline: row-level gains, losses, and net",
        fontsize=12,
        pad=12,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25)

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#3a9e6e", label="Dist wins on subset of rows"),
            Patch(facecolor="#c94040", label="Dist loses on subset of rows"),
            Patch(facecolor="#1a1a1a", label="Net (gain + loss)"),
        ],
        loc="upper right",
        frameon=True,
        fontsize=8,
    )

    fig.tight_layout()
    return fig


def _render_qwk_contribution_chart(compare_labels: tuple[str, str]) -> None:
    import matplotlib.pyplot as plt

    from tarab_model_experimentation.dev_predictions import build_per_class_gain_loss_vs_baseline

    df, summary, err = build_per_class_gain_loss_vs_baseline(compare_labels)
    if err:
        st.info(err)
        return
    if df is None or df.empty or summary is None:
        st.info("Could not compute per-class gain/loss (dev prediction CSVs needed).")
        return

    _, dist_label = compare_labels
    dist_latex = dist_label.replace("_", r"\_")
    st.markdown("#### Decomposing the QWK gap")
    st.markdown(
        "QWK penalizes errors by *squared distance* on the "
        "level scale. The full penalty for true level *c* is:\n\n"
        r"$$\text{penalty}_c \;=\; \frac{1}{(K-1)^2}\sum_{i\,:\,y_i=c}(\hat{y}_i-y_i)^2$$"
        "\n\nwith $K=19$. Each **dev true readability level** is shown three ways on "
        "the same rows: **gain** (green) = summed squared-error reductions where "
        f"**{dist_label}** beat baseline; **loss** (red) = summed increases where "
        f"**{dist_label}** was worse than baseline; **net** (diamond) = "
        rf"$\Delta_c=\text{{gain}}_c+\text{{loss}}_c="
        rf"\text{{penalty}}^{{\text{{baseline}}}}_c-\text{{penalty}}^{{\text{{{dist_latex}}}}}_c$."
        f" Positive net means **{dist_label}** lowered penalty on that level overall; "
        f"negative net means **{dist_label}** raised penalty. Because QWK is exactly the same "
        r"squared-error penalty normalized by the expected-by-chance penalty"
        " ($\\kappa=1-\\frac{\\sum p_{ij}w_{ij}}{\\sum e_{ij}w_{ij}}$ with "
        r"$w_{ij}=(i-j)^2/(K-1)^2$), these per-class bars are the additive "
        "ingredients of $\\Delta\\text{QWK}$. They decompose where the QWK gap is "
        "actually coming from."
    )
    fig = _plot_per_class_gain_loss_bars(df)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    from tarab_model_experimentation.presentation_insights import (
        render_qwk_contribution_insight,
    )

    render_qwk_contribution_insight(df, summary, compare_labels)


def _qwk_driver_narrative(loaded: dict[str, dict[str, Any]], compare_labels: tuple[str, str]) -> str:
    baseline_label, experiment_label = compare_labels
    short_exp = experiment_label.replace("dist_", "d").replace("_", " ")
    qwk_b = float(loaded[baseline_label]["qwk"])
    qwk_e = float(loaded[experiment_label]["qwk"])
    dqwk = qwk_e - qwk_b

    bucket_df, total = _error_bucket_deltas(loaded, compare_labels)
    helps = bucket_df[bucket_df["delta_sq_mass"] < 0].sort_values("delta_sq_mass")
    hurts = bucket_df[bucket_df["delta_sq_mass"] > 0].sort_values(
        "delta_sq_mass", ascending=False
    )

    def _fmt_bucket(row) -> str:
        d = int(row["abs_error"])
        return f"|err|={d} ({int(row['delta_count']):+d} samples, {int(row['delta_sq_mass']):+d} sq-mass)"

    help_txt = ", ".join(_fmt_bucket(r) for _, r in helps.head(4).iterrows()) or "none"
    hurt_txt = ", ".join(_fmt_bucket(r) for _, r in hurts.head(4).iterrows()) or "none"

    contrib = build_per_level_sq_error_contribution(loaded, compare_labels)
    if contrib.empty:
        top_help = top_hurt = None
    else:
        top_help = contrib.nlargest(1, "weighted_delta_sq").iloc[0]
        top_hurt = contrib.nsmallest(1, "weighted_delta_sq").iloc[0]

    text = (
        f"At each model's **best-QWK epoch**, QWK is **{qwk_b:.4f}** (baseline) vs **{qwk_e:.4f}** "
        f"({short_exp}, **{dqwk:+.4f}**). "
        f"QWK is driven by **squared distance**, not exact-match % alone. "
        f"**Helps:** {help_txt}. "
        f"**Hurts:** {hurt_txt}."
    )
    if top_help is not None and top_hurt is not None:
        text += (
            f" Largest penalty reduction ({short_exp} better): **L{int(top_help['level'])}**; "
            f"largest increase ({short_exp} worse): **L{int(top_hurt['level'])}**."
        )
    return text


def build_confusion_transition_deltas(
    loaded: dict[str, dict[str, Any]],
    compare_labels: tuple[str, str],
    true_levels: list[int],
    *,
    top_n: int = 3,
):
    import numpy as np
    import pandas as pd

    baseline_label, experiment_label = compare_labels
    base_cm = loaded[baseline_label]["cm_df"].to_numpy(dtype=np.int64)
    exp_cm = loaded[experiment_label]["cm_df"].to_numpy(dtype=np.int64)
    diff = exp_cm - base_cm
    index_labels = [int(x) + 1 for x in loaded[baseline_label]["cm_df"].index]

    rows: list[dict[str, Any]] = []
    for true_lev in true_levels:
        if true_lev not in index_labels:
            continue
        r = index_labels.index(true_lev)
        candidates: list[tuple[int, int]] = []
        for c, pred_lev in enumerate(index_labels):
            if pred_lev == true_lev:
                continue
            delta = int(diff[r, c])
            if delta > 0:
                candidates.append((delta, pred_lev))
        candidates.sort(reverse=True)
        if not candidates:
            rows.append(
                {
                    "true_level": true_lev,
                    "pred_level": None,
                    "Δ_count": 0,
                    "note": "no increased off-diagonal errors",
                }
            )
            continue
        for delta, pred_lev in candidates[:top_n]:
            rows.append(
                {
                    "true_level": true_lev,
                    "pred_level": pred_lev,
                    "Δ_count": delta,
                    "note": "",
                }
            )
    return pd.DataFrame(rows)


def _render_far_off_mistakes_section(
    loaded: dict[str, dict[str, Any]], compare_labels: tuple[str, str]
) -> None:
    from tarab_model_experimentation.dev_predictions import (
        build_far_off_mistake_examples_table,
        far_off_mistake_insight_stats,
    )

    st.markdown(
        "#### Sanity Check: Far-off mistakes dist_155K adds vs baseline (|err| ≥ 7)"
    )
    examples, err = build_far_off_mistake_examples_table(loaded, compare_labels)
    if err:
        st.info(err)
    elif examples is not None and not examples.empty:
        st.caption(
            f"{len(examples)} dev examples where **dist_155K** is ≥7 levels off and "
            f"**baseline was less wrong**."
        )
        st.dataframe(examples, width="stretch", hide_index=True)
        stats = far_off_mistake_insight_stats(compare_labels)
        if stats:
            from tarab_model_experimentation.presentation_insights import (
                render_far_off_mistakes_insight,
            )

            render_far_off_mistakes_insight(stats)
    else:
        st.caption(
            "No extra far-off dist_155K mistakes vs baseline at the best-QWK checkpoints."
        )


def _render_qwk_shift_analysis(compare_labels: tuple[str, str]) -> None:
    from tarab_model_experimentation.dev_predictions import (
        build_per_class_qwk_penalty_contribution,
    )

    contrib, contrib_err = build_per_class_qwk_penalty_contribution(compare_labels)
    if contrib_err:
        st.info(contrib_err)
    if contrib is not None and not contrib.empty:
        with st.expander("Per-class squared-error penalty (all levels)"):
            show = contrib.sort_values("delta_penalty", ascending=False)
            st.dataframe(
                show.rename(
                    columns={
                        "delta_penalty": "net (baseline − dist_155K)",
                        "gain_positive": "gain on rows",
                        "loss_negative": "loss on rows",
                        "penalty_baseline": "baseline penalty",
                        "penalty_dist": "dist_155K penalty",
                    }
                ),
                width="stretch",
                hide_index=True,
            )


def _render_one_comparison(
    *,
    log_files: list[str],
    title: str,
    compare_labels: tuple[str, str],
) -> bool:
    import matplotlib.pyplot as plt

    st.markdown(f"### {title}")

    loaded = _load_comparison_epochs(log_files, compare_labels)
    if loaded is None:
        missing = [
            label
            for label in compare_labels
            if log_file_for_chart_label(log_files, label) is None
        ]
        if missing:
            st.warning(f"Missing log files for: {', '.join(missing)}")
        else:
            st.warning(
                f"Could not parse best-QWK confusion matrices for "
                f"{compare_labels[0]} and {compare_labels[1]}."
            )
        return False

    fig1 = _plot_predicted_distribution(loaded, compare_labels)
    st.pyplot(fig1, clear_figure=True)
    plt.close(fig1)
    st.caption(
        "Dashed line: **dev (true)** label mix (same for both models). "
        "Bars: predicted mix at best-QWK checkpoint."
    )

    fig2 = _plot_per_class_accuracy_comparison(loaded, compare_labels)
    st.pyplot(fig2, clear_figure=True)
    plt.close(fig2)
    st.caption(
        "**Recall** (top): of gold level *k*, how often the model predicts *k*. "
        "**Precision** (bottom): when the model predicts *k*, how often gold is *k*."
    )

    _render_per_class_recall_precision_table(loaded, compare_labels)

    from tarab_model_experimentation.presentation_insights import (
        render_shift_vs_confidence_insight,
    )

    render_shift_vs_confidence_insight()

    _render_distillation_confidence_bubbles(loaded, compare_labels)

    st.markdown("#### Distribution-shift error analysis")
    st.caption(
        "ΔC migration, signed error, and far-off mistakes — where gold classes lost "
        "recall to attractors after dist_155K."
    )

    _render_confusion_delta_heatmap(loaded, compare_labels)

    fig3 = _plot_signed_error_histogram(loaded, compare_labels)
    st.pyplot(fig3, clear_figure=True)
    plt.close(fig3)

    fig_unsigned = _plot_mistake_rate_by_distance(loaded, compare_labels)
    if fig_unsigned is not None:
        st.pyplot(fig_unsigned, clear_figure=True)
        plt.close(fig_unsigned)

        from tarab_model_experimentation.presentation_insights import (
            render_unsigned_error_bridge_insight,
        )

        render_unsigned_error_bridge_insight()

    _render_far_off_mistakes_section(loaded, compare_labels)
    _render_qwk_contribution_chart(compare_labels)

    _render_qwk_shift_analysis(compare_labels)
    return True


def render_confusion_comparison_section(*, log_files: list[str]) -> None:
    """Deprecated: use ``focused_analysis.render_focused_baseline_vs_dist_155k_section``."""
    from tarab_model_experimentation.focused_analysis import (
        render_focused_baseline_vs_dist_155k_section,
    )

    render_focused_baseline_vs_dist_155k_section(log_files=log_files)
