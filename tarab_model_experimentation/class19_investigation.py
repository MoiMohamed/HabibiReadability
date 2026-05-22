from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from tarab_model_experimentation.confusion_comparison import (
    best_qwk_epoch_from_log,
    log_file_for_chart_label,
)
from tarab_model_experimentation.constants import DATA_DIR, DIST_ORANGE, SPLITS_DIR, TARAB_SHARE_COLOR
from tarab_model_experimentation.metrics import one_based_confusion_matrix
from tarab_model_experimentation.training_split import load_barec_train_readability_counts

BASELINE_PRETRAIN_CSV = DATA_DIR / "readability_strat_barec_train_aldi_ags.csv"
DIST_155K_SPLIT_CSV = SPLITS_DIR / "barec_tarab_2X_55k_match_distribution_155k.csv"
# DIST_155K_WO_SPLIT_CSV = SPLITS_DIR / "barec_tarab_2X_55k_match_distribution_155k_wo_class19.csv"
LEVELS = list(range(1, 20))
CLASS_19 = 19
ALDI_AXIS_ORDER = ("high", "mid", "low")
AGS_AXIS_ORDER = ("low", "mid", "high")
_ALDI_AXIS_LABELS = ("High", "Mid", "Low")
_AGS_AXIS_LABELS = ("Low", "Mid", "High")
TARAB_CONFIDENCE_STAT_COLUMNS = ("mean", "std", "min", "q1", "median", "q3", "max")
_BASELINE_COLOR = "#1f77b4"
_DIST_COLOR = DIST_ORANGE

_INVESTIGATION_SPECS: tuple[tuple[str, Path, str, tuple[str, str]], ...] = (
    # (
    #     "Class 19 collapse — pretraining investigation",
    #     DIST_155K_SPLIT_CSV,
    #     "dist_155K",
    #     ("baseline", "dist_155K"),
    # ),
    # (
    #     "Class 19 collapse — pretraining investigation (dist_155K_wo_19)",
    #     DIST_155K_WO_SPLIT_CSV,
    #     "dist_155K_wo_19",
    #     ("baseline", "dist_155K_wo_19"),
    # ),
)


@st.cache_data(show_spinner=False)
def _load_baseline_pretrain_df():
    import pandas as pd

    if not BASELINE_PRETRAIN_CSV.exists():
        return None
    df = pd.read_csv(BASELINE_PRETRAIN_CSV, encoding="utf-8")
    if "Readability" not in df.columns or "Sentence" not in df.columns:
        return None
    out = df[["Sentence", "Readability"]].copy()
    out.rename(columns={"Readability": "label"}, inplace=True)
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out = out.dropna(subset=["label", "Sentence"]).copy()
    out["label"] = out["label"].astype(int)
    out = out[(out["label"] >= 1) & (out["label"] <= 19)].copy()
    out["source"] = "barec"
    return out


@st.cache_data(show_spinner=False)
def _load_split_pretrain_df(split_csv: str, *, _schema_version: int = 2):
    import pandas as pd

    path = Path(split_csv)
    if not path.exists():
        return None
    usecols = ["Sentence", "Readability", "source"]
    cols = list(pd.read_csv(path, nrows=0, encoding="utf-8").columns)
    optional = [
        c
        for c in (
            "readability_confidence",
            "type",
            "aldi_score",
            "aldi_label",
            "ags_score",
            "ags_label",
        )
        if c in cols
    ]
    df = pd.read_csv(
        path,
        usecols=[c for c in usecols + optional if c in cols],
        encoding="utf-8",
    )
    if "Readability" not in df.columns or "Sentence" not in df.columns:
        return None
    out = df.copy()
    out.rename(columns={"Readability": "label"}, inplace=True)
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out = out.dropna(subset=["label", "Sentence"]).copy()
    out["label"] = out["label"].astype(int)
    out = out[(out["label"] >= 1) & (out["label"] <= 19)].copy()
    out["source"] = out["source"].astype(str).str.strip().str.lower()
    if "readability_confidence" in out.columns:
        out["readability_confidence"] = pd.to_numeric(
            out["readability_confidence"], errors="coerce"
        )
    if "type" in out.columns:
        out["type"] = out["type"].replace("", pd.NA).astype("string")
        tarab_mask = out["source"] == "tarab"
        if tarab_mask.any():
            out.loc[tarab_mask, "type"] = (
                out.loc[tarab_mask, "type"].astype(str).str.strip().str.lower()
            )
    return out


def _label_distribution(df) -> "pd.Series":
    import pandas as pd

    return (
        df["label"]
        .value_counts()
        .reindex(LEVELS, fill_value=0)
        .astype(int)
    )


def _text_stats(texts) -> dict[str, float | int]:
    import pandas as pd

    series = pd.Series(texts).astype(str)
    lengths = series.str.split().str.len()
    lengths = lengths[lengths > 0]
    if lengths.empty:
        return {
            "count": 0,
            "mean_words": 0.0,
            "median_words": 0.0,
            "std_words": 0.0,
        }
    return {
        "count": int(len(lengths)),
        "mean_words": float(lengths.mean()),
        "median_words": float(lengths.median()),
        "std_words": float(lengths.std()),
    }


def _tarab_confidence_stats_by_level(df) -> "pd.DataFrame":
    import pandas as pd

    empty = pd.DataFrame(
        index=LEVELS,
        columns=list(TARAB_CONFIDENCE_STAT_COLUMNS),
        dtype=float,
    )
    if "readability_confidence" not in df.columns:
        return empty
    tarab = df[df["source"] == "tarab"]
    if tarab.empty:
        return empty

    by_level = tarab.groupby("label", sort=True)["readability_confidence"]
    return pd.DataFrame(
        {
            "mean": by_level.mean(),
            "std": by_level.std(),
            "min": by_level.min(),
            "q1": by_level.quantile(0.25),
            "median": by_level.median(),
            "q3": by_level.quantile(0.75),
            "max": by_level.max(),
        },
        columns=list(TARAB_CONFIDENCE_STAT_COLUMNS),
    ).reindex(LEVELS)


def build_pretraining_label_counts_table(baseline_df, dist_df) -> "pd.DataFrame":
    import pandas as pd

    baseline_counts = _label_distribution(baseline_df)
    dist_counts = _label_distribution(dist_df)
    dist_tarab_counts = _label_distribution(dist_df[dist_df["source"] == "tarab"])
    return pd.DataFrame(
        {
            "baseline_count": baseline_counts.values,
            "dist_155K_count": dist_counts.values,
            "dist_155K_tarab_count": dist_tarab_counts.values,
        },
        index=pd.Index(LEVELS, name="readability_level"),
    )


def build_pretraining_confidence_table(dist_df) -> "pd.DataFrame":
    conf_stats = _tarab_confidence_stats_by_level(dist_df)
    return (conf_stats * 100.0).round(2).add_prefix("tarab_conf_")


def _plot_pretraining_distributions(baseline_df, dist_df, *, dist_name: str):
    import matplotlib.pyplot as plt
    import numpy as np

    dist_counts = _label_distribution(dist_df)
    tarab_counts = _label_distribution(dist_df[dist_df["source"] == "tarab"])
    tarab_share = np.where(
        dist_counts.values > 0,
        100.0 * tarab_counts.values / dist_counts.values,
        0.0,
    )

    datasets = [
        ("baseline (BAREC only)", baseline_df, _BASELINE_COLOR),
        (f"{dist_name} (BAREC + Tarab)", dist_df, _DIST_COLOR),
    ]
    x = np.arange(len(LEVELS))
    edge = {"edgecolor": "white", "linewidth": 0.3}

    fig = plt.figure(figsize=(14, 6.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1], hspace=0.35, wspace=0.1)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left)

    for ax, (title, df, color) in zip((ax_left, ax_right), datasets):
        counts = _label_distribution(df)
        ax.bar(x, counts.values, color=color, **edge)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([str(lev) for lev in LEVELS], fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)

    ax_left.set_ylabel("Count")
    ax_right.set_xlabel("Readability level")

    ax_tarab = fig.add_subplot(gs[1, :], sharex=ax_left)
    ax_tarab.bar(
        x,
        tarab_share,
        color=TARAB_SHARE_COLOR,
        edgecolor="white",
        linewidth=0.35,
    )
    ax_tarab.set_ylabel("Tarab share\nof dist (%)")
    ax_tarab.set_xlabel("Readability level")
    ax_tarab.set_ylim(0, 100)
    ax_tarab.set_xticks(x)
    ax_tarab.set_xticklabels([str(lev) for lev in LEVELS], fontsize=8)
    ax_tarab.grid(True, axis="y", alpha=0.25)

    max_share = float(np.max(tarab_share)) if len(tarab_share) else 0.0
    if max_share > 0:
        ax_tarab.axhline(
            max_share,
            color="#444444",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            zorder=3,
        )
        ax_tarab.annotate(
            f"{max_share:.1f}%",
            xy=(len(LEVELS) - 1, max_share),
            xytext=(4, 4),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8,
            color="#444444",
        )

    fig.suptitle("Training label distribution", fontsize=12, y=0.98)
    fig.subplots_adjust(top=0.9, bottom=0.08, left=0.06, right=0.98)
    return fig


def _plot_tarab_confidence_boxplot_by_level(dist_df, *, dist_name: str):
    import matplotlib.pyplot as plt
    import numpy as np

    if "readability_confidence" not in dist_df.columns:
        return None

    tarab = dist_df[dist_df["source"] == "tarab"].dropna(subset=["readability_confidence"])
    if tarab.empty:
        return None

    x = np.arange(len(LEVELS))
    box_data: list[np.ndarray] = []
    for lev in LEVELS:
        conf = tarab.loc[tarab["label"] == lev, "readability_confidence"].to_numpy(dtype=float)
        box_data.append(100.0 * conf if len(conf) else np.array([], dtype=float))

    flat = np.concatenate([d for d in box_data if len(d)]) if any(len(d) for d in box_data) else None

    fig, ax = plt.subplots(figsize=(14, 7.5))
    bp = ax.boxplot(
        box_data,
        positions=x,
        widths=0.62,
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4, "color": "#555555"},
        medianprops={"color": "#222222", "linewidth": 1.6},
        whiskerprops={"color": "#666666", "linewidth": 1.1},
        capprops={"color": "#666666", "linewidth": 1.1},
        boxprops={"linewidth": 1.1, "edgecolor": "#4a7a7e"},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(TARAB_SHARE_COLOR)
        patch.set_alpha(0.72)

    ax.set_xticks(x)
    ax.set_xticklabels([str(lev) for lev in LEVELS], fontsize=9)
    ax.set_xlabel("Readability level", fontsize=10)
    ax.set_ylabel("Tarab pseudo-label confidence (%)", fontsize=10)
    if flat is not None and len(flat):
        y_pad = max(4.0, 0.04 * (float(np.max(flat)) - float(np.min(flat))))
        ax.set_ylim(max(0.0, float(np.min(flat)) - y_pad), min(100.0, float(np.max(flat)) + y_pad))
    else:
        ax.set_ylim(0, 100)
    ax.set_title(f"Tarab confidence by level ({dist_name} training, Tarab rows only)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.1, top=0.92)
    return fig


def _prepare_sentence_length_columns(df):
    out = df.copy()
    text = out["Sentence"].astype(str).str.strip()
    out["char_length"] = text.str.len()
    out["word_count"] = text.str.split().str.len()
    return out


def _plot_text_metric_boxplot_by_level(
    dist_df,
    *,
    source: str,
    metric: str,
    metric_label: str,
    dist_name: str,
    facecolor: str,
    whisker_percentiles: tuple[float, float] = (5, 95),
    ylim_percentiles: tuple[float, float] = (2, 98),
):
    """Boxplot of ``metric`` per level; y-axis zoomed, outliers hidden."""
    import matplotlib.pyplot as plt
    import numpy as np

    subset = dist_df[dist_df["source"] == source].dropna(subset=[metric])
    if subset.empty:
        return None

    x = np.arange(len(LEVELS))
    box_data: list[np.ndarray] = []
    for lev in LEVELS:
        vals = subset.loc[subset["label"] == lev, metric].to_numpy(dtype=float)
        box_data.append(vals if len(vals) else np.array([], dtype=float))

    flat = (
        np.concatenate([d for d in box_data if len(d)])
        if any(len(d) for d in box_data)
        else None
    )

    fig, ax = plt.subplots(figsize=(14, 7.5))
    bp = ax.boxplot(
        box_data,
        positions=x,
        widths=0.62,
        patch_artist=True,
        whis=whisker_percentiles,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.6},
        whiskerprops={"color": "#666666", "linewidth": 1.1},
        capprops={"color": "#666666", "linewidth": 1.1},
        boxprops={"linewidth": 1.1, "edgecolor": "#4a4a4a"},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(facecolor)
        patch.set_alpha(0.72)

    ax.set_xticks(x)
    ax.set_xticklabels([str(lev) for lev in LEVELS], fontsize=9)
    ax.set_xlabel("Readability level", fontsize=10)
    ax.set_ylabel(metric_label, fontsize=10)
    if flat is not None and len(flat):
        hi = float(np.percentile(flat, ylim_percentiles[1]))
        whisker_lows: list[float] = []
        for d in box_data:
            if len(d) >= 2:
                whisker_lows.append(float(np.percentile(d, whisker_percentiles[0])))
            elif len(d) == 1:
                whisker_lows.append(float(d[0]))
        # Floor: lowest whisker (p5 per level) minus padding — not global p2 alone.
        y_min = min(whisker_lows) if whisker_lows else float(np.percentile(flat, ylim_percentiles[0]))
        y_max = hi
        span = y_max - y_min if y_max > y_min else 1.0
        pad = max(0.5, 0.1 * span)
        ax.set_ylim(max(0.0, y_min - pad), y_max + pad)
    src_title = "BAREC" if source == "barec" else "Tarab"
    wlo, whi = whisker_percentiles
    ax.set_title(
        f"{src_title} {metric_label.lower()} by level ({dist_name}, {src_title} rows)",
        fontsize=11,
    )
    ax.text(
        0.01,
        0.99,
        f"Y max p{ylim_percentiles[1]:.0f}; whiskers p{wlo:.0f}–p{whi:.0f}; "
        f"floor = lowest whisker − padding; outliers hidden",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="#555555",
    )
    ax.grid(True, axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.1, top=0.92)
    return fig


def _build_class19_summary_table(baseline_df, dist_df, *, dist_name: str):
    import pandas as pd

    barec_ref = load_barec_train_readability_counts()
    rows: list[dict[str, Any]] = []

    def _row(name: str, df, *, subset=None) -> None:
        part = df if subset is None else df[subset]
        total = len(part)
        c19 = int((part["label"] == CLASS_19).sum())
        rows.append(
            {
                "dataset": name,
                "total_rows": total,
                "class_19_count": c19,
                "class_19_%": (100.0 * c19 / total) if total else 0.0,
            }
        )

    _row("baseline (BAREC)", baseline_df)
    _row(f"{dist_name} (all)", dist_df)
    _row(f"{dist_name} · barec", dist_df, subset=dist_df["source"] == "barec")
    _row(f"{dist_name} · tarab", dist_df, subset=dist_df["source"] == "tarab")
    rows.append(
        {
            "dataset": "BAREC train reference (counts)",
            "total_rows": int(sum(barec_ref.values())),
            "class_19_count": int(barec_ref.get(CLASS_19, 0)),
            "class_19_%": (
                100.0 * barec_ref.get(CLASS_19, 0) / sum(barec_ref.values())
                if barec_ref
                else 0.0
            ),
        }
    )
    return pd.DataFrame(rows)


def _build_class19_text_stats_table(baseline_df, dist_df, *, dist_name: str):
    import pandas as pd

    slices = [
        ("baseline · class 19", baseline_df[baseline_df["label"] == CLASS_19]),
        (f"{dist_name} · class 19 (all)", dist_df[dist_df["label"] == CLASS_19]),
        (
            f"{dist_name} · class 19 · barec",
            dist_df[(dist_df["label"] == CLASS_19) & (dist_df["source"] == "barec")],
        ),
        (
            f"{dist_name} · class 19 · tarab",
            dist_df[(dist_df["label"] == CLASS_19) & (dist_df["source"] == "tarab")],
        ),
    ]
    rows = []
    for name, part in slices:
        stats = _text_stats(part["Sentence"])
        rows.append({"slice": name, **stats})
    return pd.DataFrame(rows)


def _build_class19_confidence_table(dist_df):
    import pandas as pd

    if "readability_confidence" not in dist_df.columns:
        return pd.DataFrame()

    c19 = dist_df[dist_df["label"] == CLASS_19].copy()
    rows = []
    for source, part in c19.groupby("source", sort=True):
        conf = part["readability_confidence"].dropna()
        rows.append(
            {
                "source": source,
                "count": int(len(part)),
                "mean_confidence": float(conf.mean()) if len(conf) else None,
                "median_confidence": float(conf.median()) if len(conf) else None,
                "min_confidence": float(conf.min()) if len(conf) else None,
                "max_confidence": float(conf.max()) if len(conf) else None,
            }
        )
    return pd.DataFrame(rows)


def _plot_class19_confidence(dist_df, *, dist_name: str):
    import matplotlib.pyplot as plt

    if "readability_confidence" not in dist_df.columns:
        return None

    c19 = dist_df[dist_df["label"] == CLASS_19]
    if c19.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    for source, color in (("barec", "#1f77b4"), ("tarab", DIST_ORANGE)):
        conf = c19.loc[c19["source"] == source, "readability_confidence"].dropna()
        if conf.empty:
            continue
        ax.hist(conf, bins=20, alpha=0.65, label=f"{source} (n={len(conf)})", color=color)
    ax.set_xlabel("readability_confidence")
    ax.set_ylabel("Count")
    ax.set_title(f"Class 19 pseudo-label confidence in {dist_name} pretraining")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _prediction_distribution_when_true_19(cm_df) -> "pd.Series":
    import pandas as pd

    cm = one_based_confusion_matrix(cm_df)
    label = str(CLASS_19)
    if label not in cm.index:
        return pd.Series(dtype=int)
    row = cm.loc[label].astype(int)
    return row[row > 0].sort_values(ascending=False)


def _plot_dev_predictions_when_true_19(
    log_files: list[str], compare_labels: tuple[str, str]
):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 4.5))
    any_data = False
    colors = {"baseline": "#1f77b4", "dist_155K": DIST_ORANGE}  # , "dist_155K_wo_19": "#2ca02c"

    for label in compare_labels:
        log_file = log_file_for_chart_label(log_files, label)
        if log_file is None:
            continue
        epoch = best_qwk_epoch_from_log(log_file)
        if epoch is None:
            continue
        dist = _prediction_distribution_when_true_19(epoch["cm_df"])
        if dist.empty:
            continue
        any_data = True
        preds = dist.index.astype(int).tolist()
        counts = dist.values
        pct = 100.0 * counts / counts.sum()
        ax.plot(
            preds,
            pct,
            marker="o",
            linestyle="-",
            linewidth=1.5,
            color=colors.get(label, "#9467bd"),
            label=f"{label} (epoch {epoch['epoch']:.0f})",
        )

    if not any_data:
        plt.close(fig)
        return None

    ax.axvline(CLASS_19, color="#d62728", linestyle="--", linewidth=1, label="correct (19)")
    ax.set_xlabel("Predicted level (when true = 19)")
    ax.set_ylabel("% of class-19 dev examples")
    ax.set_title("Where dev gold-19 examples are predicted (best QWK checkpoint)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _class19_representation_message(baseline_df, dist_df, *, dist_name: str) -> str | None:
    b19 = int((baseline_df["label"] == CLASS_19).sum())
    d19 = int((dist_df["label"] == CLASS_19).sum())
    tarab19 = int(((dist_df["label"] == CLASS_19) & (dist_df["source"] == "tarab")).sum())

    # if dist_name == "dist_155K_wo_19":
    #     return (
    #         f"**{dist_name}** removed all Tarab pseudo L19 (conf < 0.7). "
    #         f"Pretraining L19 = **{d19}** rows (BAREC only, same {b19} as baseline reference)."
    #     )

    if d19 > b19:
        return (
            f"**{dist_name}** has more level-19 rows than baseline ({d19} vs {b19} BAREC-only). "
            f"The extra {tarab19} Tarab rows are the main change — not absence of class 19."
        )
    if d19 < b19:
        return (
            f"**{dist_name}** has fewer level-19 rows than baseline ({d19} vs {b19}). "
            "Under-representation may contribute to collapse."
        )
    return None


def _render_one_class19_block(
    *,
    log_files: list[str],
    title: str,
    split_csv: Path,
    dist_name: str,
    dev_compare_labels: tuple[str, str],
) -> bool:
    import matplotlib.pyplot as plt

    st.markdown(f"### {title}")
    st.caption(
        f"Baseline = BAREC only (`readability_strat_barec_train_aldi_ags.csv`). "
        f"**{dist_name}** = `{split_csv.name}`."
    )

    baseline_df = _load_baseline_pretrain_df()
    dist_df = _load_split_pretrain_df(str(split_csv))
    if baseline_df is None or dist_df is None:
        st.warning(
            f"Missing data for {dist_name}. Need baseline CSV and `{split_csv}`."
        )
        return False

    st.markdown("#### Step 1 — Class 19 representation in pretraining")
    summary = _build_class19_summary_table(baseline_df, dist_df, dist_name=dist_name)
    st.dataframe(summary, width="stretch", hide_index=True)

    msg = _class19_representation_message(baseline_df, dist_df, dist_name=dist_name)
    if msg:
        st.info(msg)

    st.markdown("#### Step 2 — Text characteristics (class 19 only)")
    text_tbl = _build_class19_text_stats_table(baseline_df, dist_df, dist_name=dist_name)
    st.dataframe(text_tbl, width="stretch", hide_index=True)

    st.markdown(f"#### Step 3 — Source & confidence ({dist_name} class 19)")
    c19 = dist_df[dist_df["label"] == CLASS_19]
    if c19.empty:
        st.info(f"No class-19 rows in **{dist_name}** pretraining split.")
    else:
        src_tbl = c19["source"].value_counts().reset_index()
        src_tbl.columns = ["source", "count"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Counts by source**")
            st.dataframe(src_tbl, width="stretch", hide_index=True)
        with c2:
            conf_tbl = _build_class19_confidence_table(dist_df)
            if not conf_tbl.empty:
                st.markdown("**`readability_confidence` by source**")
                st.dataframe(conf_tbl, width="stretch", hide_index=True)

        conf_fig = _plot_class19_confidence(dist_df, dist_name=dist_name)
        if conf_fig is not None:
            st.pyplot(conf_fig, clear_figure=True)
            plt.close(conf_fig)

    st.markdown("#### Dev set — prediction spread when gold = 19")
    fig_dev = _plot_dev_predictions_when_true_19(log_files, dev_compare_labels)
    if fig_dev is not None:
        st.pyplot(fig_dev, clear_figure=True)
        plt.close(fig_dev)
    else:
        labels = " / ".join(dev_compare_labels)
        st.info(f"Could not load confusion matrices for {labels}.")
    return True


def render_pretraining_label_distribution_section() -> None:
    """Baseline vs dist_155K label counts in training splits (before prediction analysis)."""
    import matplotlib.pyplot as plt

    baseline_df = _load_baseline_pretrain_df()
    dist_df = _load_split_pretrain_df(str(DIST_155K_SPLIT_CSV))
    if baseline_df is None or dist_df is None:
        st.warning(
            "Missing training CSVs for label distribution chart. "
            f"Need `{BASELINE_PRETRAIN_CSV.name}` and `{DIST_155K_SPLIT_CSV.name}`."
        )
        return

    st.markdown("### Training label distribution")
    st.caption(
        "Top: raw counts — baseline (blue) vs dist_155K (orange). "
        "Bottom: Tarab share of dist_155K per level."
    )
    fig = _plot_pretraining_distributions(baseline_df, dist_df, dist_name="dist_155K")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    with st.expander("Numeric tables (label counts)"):
        st.dataframe(
            build_pretraining_label_counts_table(baseline_df, dist_df),
            width="stretch",
        )


def render_dist_155k_confidence_distribution_section() -> None:
    """Tarab pseudo-label confidence by readability level (dist_155K training split)."""
    import matplotlib.pyplot as plt

    dist_df = _load_split_pretrain_df(str(DIST_155K_SPLIT_CSV))
    if dist_df is None:
        st.warning(
            "Missing dist_155K training CSV for confidence chart. "
            f"Need `{DIST_155K_SPLIT_CSV.name}`."
        )
        return

    st.markdown("#### Confidence distribution per class")
    conf_fig = _plot_tarab_confidence_boxplot_by_level(dist_df, dist_name="dist_155K")
    if conf_fig is not None:
        st.pyplot(conf_fig, clear_figure=True)
        plt.close(conf_fig)
        st.caption(
            "Tarab pseudo-label confidence (0–100) per level; y-axis zoomed to data range. "
            "Box = q1–q3, black line = median, whiskers = min/max."
        )
        with st.expander("Numeric tables (confidence per class)"):
            st.caption("Stats are **0–100** (×100 from raw score), Tarab rows only.")
            st.dataframe(build_pretraining_confidence_table(dist_df), width="stretch")
    else:
        st.info("No Tarab `readability_confidence` values in the dist_155K split.")


def _show_text_length_boxplot(
    column,
    dist_df,
    *,
    source: str,
    metric: str,
    metric_label: str,
    dist_name: str,
    facecolor: str,
    empty_msg: str,
) -> None:
    import matplotlib.pyplot as plt

    with column:
        try:
            fig = _plot_text_metric_boxplot_by_level(
                dist_df,
                source=source,
                metric=metric,
                metric_label=metric_label,
                dist_name=dist_name,
                facecolor=facecolor,
            )
        except Exception as exc:
            st.error(f"Could not draw {source} {metric}: {exc}")
            return
        if fig is None:
            st.info(empty_msg)
            return
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def _compute_aldi_ags_pair_pct(split_df, *, source: str) -> tuple[Any, int] | None:
    """3×3 matrix (%): rows ALDi high→low, cols AGS low→high."""
    import numpy as np
    import pandas as pd

    if "aldi_label" not in split_df.columns or "ags_label" not in split_df.columns:
        return None

    sub = split_df[split_df["source"] == source].copy()
    sub["aldi_label"] = sub["aldi_label"].astype(str).str.strip().str.lower()
    sub["ags_label"] = sub["ags_label"].astype(str).str.strip().str.lower()
    sub = sub[
        sub["aldi_label"].isin(ALDI_AXIS_ORDER) & sub["ags_label"].isin(AGS_AXIS_ORDER)
    ]
    if sub.empty:
        return None

    ct = pd.crosstab(sub["aldi_label"], sub["ags_label"])
    ct = ct.reindex(index=list(ALDI_AXIS_ORDER), columns=list(AGS_AXIS_ORDER), fill_value=0)
    total = int(ct.to_numpy().sum())
    if total == 0:
        return None
    pct = (ct.to_numpy(dtype=np.float64) / total) * 100.0
    return pct, total


def _compute_aldi_ags_prob_at_level(
    split_df, *, source: str, level: int
) -> tuple[np.ndarray | None, int]:
    """3×3 probability matrix (sums to 1) for one source and readability level."""
    import pandas as pd

    if "aldi_label" not in split_df.columns or "ags_label" not in split_df.columns:
        return None, 0

    sub = split_df[(split_df["source"] == source) & (split_df["label"] == level)].copy()
    sub["aldi_label"] = sub["aldi_label"].astype(str).str.strip().str.lower()
    sub["ags_label"] = sub["ags_label"].astype(str).str.strip().str.lower()
    sub = sub[
        sub["aldi_label"].isin(ALDI_AXIS_ORDER) & sub["ags_label"].isin(AGS_AXIS_ORDER)
    ]
    if sub.empty:
        return None, 0

    ct = pd.crosstab(sub["aldi_label"], sub["ags_label"])
    ct = ct.reindex(index=list(ALDI_AXIS_ORDER), columns=list(AGS_AXIS_ORDER), fill_value=0)
    total = int(ct.to_numpy().sum())
    if total == 0:
        return None, 0
    prob = ct.to_numpy(dtype=np.float64) / total
    return prob, total


def _aldi_ags_overlap_divergence(p_prob: np.ndarray, q_prob: np.ndarray) -> tuple[float, float]:
    """Histogram intersection (overlap %) and total-variation divergence (%)."""
    overlap_pct = float(np.minimum(p_prob, q_prob).sum()) * 100.0
    divergence_pct = float(0.5 * np.abs(p_prob - q_prob).sum()) * 100.0
    return overlap_pct, divergence_pct


def _marginal_divergences_from_joint(
    barec_prob: np.ndarray, tarab_prob: np.ndarray
) -> tuple[float, float, float, float]:
    """ALDi and AGS 1D marginals from 3×3 joint (rows=ALDi, cols=AGS)."""
    barec_aldi = barec_prob.sum(axis=1)
    tarab_aldi = tarab_prob.sum(axis=1)
    barec_ags = barec_prob.sum(axis=0)
    tarab_ags = tarab_prob.sum(axis=0)
    aldi_overlap, aldi_div = _aldi_ags_overlap_divergence(barec_aldi, tarab_aldi)
    ags_overlap, ags_div = _aldi_ags_overlap_divergence(barec_ags, tarab_ags)
    return aldi_overlap, aldi_div, ags_overlap, ags_div


def _spearman_divergence_vs_penalty(
    overlap_df, *, divergence_col: str, penalty_df
) -> tuple[float | None, float | None, int]:
    from scipy.stats import spearmanr

    merged = overlap_df.merge(penalty_df, on="level", how="inner")
    if len(merged) < 3:
        return None, None, len(merged)
    rho, pval = spearmanr(
        merged[divergence_col].to_numpy(dtype=float),
        merged["delta_penalty"].to_numpy(dtype=float),
    )
    return float(rho), float(pval), len(merged)


@st.cache_data(show_spinner=False)
def _build_dist155k_aldi_ags_level_overlap(split_csv: str, *, _schema_version: int = 3):
    import pandas as pd

    split_df = _load_split_pretrain_df(split_csv, _schema_version=_schema_version)
    if split_df is None:
        return None, "Split CSV not found."
    if "aldi_label" not in split_df.columns or "ags_label" not in split_df.columns:
        return None, "ALDi/AGS columns missing. Run `python3 scripts/fill_splits_aldi_ags.py`."

    rows: list[dict[str, Any]] = []
    for level in LEVELS:
        barec_prob, n_barec = _compute_aldi_ags_prob_at_level(
            split_df, source="barec", level=level
        )
        tarab_prob, n_tarab = _compute_aldi_ags_prob_at_level(
            split_df, source="tarab", level=level
        )
        if barec_prob is None or tarab_prob is None or n_barec == 0 or n_tarab == 0:
            continue
        overlap_pct, joint_div = _aldi_ags_overlap_divergence(barec_prob, tarab_prob)
        aldi_overlap, aldi_div, ags_overlap, ags_div = _marginal_divergences_from_joint(
            barec_prob, tarab_prob
        )
        rows.append(
            {
                "level": level,
                "n_barec": n_barec,
                "n_tarab": n_tarab,
                "overlap_pct": overlap_pct,
                "divergence_pct": joint_div,
                "joint_divergence_pct": joint_div,
                "aldi_overlap_pct": aldi_overlap,
                "aldi_divergence_pct": aldi_div,
                "ags_overlap_pct": ags_overlap,
                "ags_divergence_pct": ags_div,
            }
        )
    if not rows:
        return None, "No per-level ALDi × AGS pairs in the split."
    return pd.DataFrame(rows), None


_DIVERGENCE_METRICS: tuple[tuple[str, str, str], ...] = (
    ("joint_divergence_pct", "ALDi×AGS (joint)", _DIST_COLOR),
    ("ags_divergence_pct", "AGS only", "#8c4a12"),
    ("aldi_divergence_pct", "ALDi only", "#4a6fa5"),
)


def _plot_marginal_divergences_vs_dist_penalty(
    overlap_df,
    penalty_df,
    *,
    spearman_stats: dict[str, tuple[float | None, float | None, int]],
) -> Any:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    merged = overlap_df.merge(penalty_df, on="level", how="inner").sort_values("level")
    if merged.empty:
        return None

    n_metrics = len(_DIVERGENCE_METRICS)
    fig, axes = plt.subplots(
        n_metrics + 1,
        1,
        figsize=(14, 3.2 * (n_metrics + 1)),
        sharex=True,
        gridspec_kw={"hspace": 0.22},
    )
    x = np.arange(len(merged))
    x_labels = [str(int(v)) for v in merged["level"]]
    width = 0.72

    for ax, (col, label, color) in zip(axes[:n_metrics], _DIVERGENCE_METRICS):
        vals = merged[col].to_numpy(dtype=float)
        ax.bar(x, vals, color=color, edgecolor="white", linewidth=0.5, width=width)
        ax.set_ylabel(f"{label}\ndivergence %")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25)
        rho, pval, n = spearman_stats.get(col, (None, None, 0))
        if rho is not None and pval is not None:
            ax.set_title(
                f"{label} — Spearman vs dev penalty: ρ = {rho:+.3f}, p = {pval:.3g} (n={n})",
                fontsize=10,
                loc="left",
            )

    fig.suptitle(
        "BAREC vs Tarab pseudo-label domain overlap per readability level (dist_155K split)",
        fontsize=12,
        y=1.002,
    )

    ax_pen = axes[-1]
    pen_vals = merged["delta_penalty"].to_numpy(dtype=float)
    pen_colors = np.where(pen_vals >= 0, "#3a9e6e", "#c94040")
    ax_pen.bar(x, pen_vals, color=pen_colors, edgecolor="white", linewidth=0.5, width=width)
    ax_pen.axhline(0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_pen.set_xticks(x)
    ax_pen.set_xticklabels(x_labels)
    ax_pen.set_xlabel("Readability level")
    ax_pen.set_ylabel(
        "Dev squared-error savings vs baseline\n"
        "Σ (baseline err² − dist err²) / (K−1)²"
    )
    ax_pen.spines[["top", "right"]].set_visible(False)
    ax_pen.grid(True, axis="y", alpha=0.25)
    ax_pen.legend(
        handles=[
            Patch(facecolor="#3a9e6e", label="Dist wins on dev"),
            Patch(facecolor="#c94040", label="Dist loses on dev"),
        ],
        loc="upper right",
        frameon=True,
        fontsize=8,
    )

    fig.tight_layout()
    return fig


def _plot_ags_divergence_vs_penalty_scatter(
    merged,
    *,
    spearman_r: float | None,
    spearman_p: float | None,
    n: int,
) -> Any:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 6))
    x = merged["ags_divergence_pct"].to_numpy(dtype=float)
    y = merged["delta_penalty"].to_numpy(dtype=float)
    colors = np.where(y >= 0, "#3a9e6e", "#c94040")
    ax.scatter(x, y, c=colors, s=72, edgecolors="white", linewidths=0.6, zorder=3)
    for _, row in merged.iterrows():
        ax.annotate(
            f"L{int(row['level'])}",
            (row["ags_divergence_pct"], row["delta_penalty"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="#333333",
        )

    x_span = max(float(x.max() - x.min()), 1.0)
    y_span = max(float(y.max() - y.min()), 0.05)
    pad_x = 0.12 * x_span
    pad_y = 0.12 * y_span
    ax.set_xlim(float(x.min()) - pad_x, float(x.max()) + pad_x)
    ax.set_ylim(float(y.min()) - pad_y, float(y.max()) + pad_y)
    ax.axhline(0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("AGS divergence % (BAREC vs Tarab at same readability level)")
    ax.set_ylabel(
        "Dev squared-error savings vs baseline\n"
        "Σ (baseline err² − dist err²) / (K−1)²"
    )
    title = "AGS divergence vs distillation outcome (per gold level)"
    if spearman_r is not None and spearman_p is not None:
        title += f"\nSpearman ρ = {spearman_r:+.3f}, p = {spearman_p:.3g} (n = {n})"
    ax.set_title(title, fontsize=11, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25)

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#3a9e6e", label="Dist wins on dev"),
            Patch(facecolor="#c94040", label="Dist loses on dev"),
        ],
        loc="upper right",
        frameon=True,
        fontsize=8,
    )
    fig.tight_layout()
    return fig


def render_dist_155k_aldi_ags_overlap_by_level(
    compare_labels: tuple[str, str] = ("baseline", "dist_155K"),
) -> None:
    """Per-level ALDi×AGS overlap (split) vs dev per-class distillation penalty."""
    import matplotlib.pyplot as plt

    from tarab_model_experimentation.dev_predictions import (
        build_per_class_gain_loss_vs_baseline,
    )

    overlap_df, err = _build_dist155k_aldi_ags_level_overlap(str(DIST_155K_SPLIT_CSV))
    if err:
        st.info(err)
        return
    if overlap_df is None or overlap_df.empty:
        st.info("No per-level ALDi × AGS overlap data.")
        return

    st.markdown("### BAREC vs Tarab domain overlap per level (ALDi × AGS)")
    st.caption(
        f"For each readability level in `{DIST_155K_SPLIT_CSV.name}`, compare **BAREC train** vs "
        "**Tarab pseudo** rows at that level. "
        "**Joint** = 3×3 ALDi×AGS divergence; **AGS only** / **ALDi only** = marginal distributions "
        "(how widely used across varieties vs dialectness). "
        "Divergence = 100% − histogram overlap. "
        "Primary view: **AGS divergence vs dev penalty** scatter (one point per level)."
    )

    contrib, _summary, pen_err = build_per_class_gain_loss_vs_baseline(compare_labels)
    penalty_df = None
    spearman_stats: dict[str, tuple[float | None, float | None, int]] = {}
    if not pen_err and contrib is not None and not contrib.empty:
        penalty_df = contrib[["level", "delta_penalty"]].copy()
        for col, _label, _color in _DIVERGENCE_METRICS:
            spearman_stats[col] = _spearman_divergence_vs_penalty(
                overlap_df, divergence_col=col, penalty_df=penalty_df
            )

    std_joint = float(overlap_df["joint_divergence_pct"].std())
    std_ags = float(overlap_df["ags_divergence_pct"].std())
    std_aldi = float(overlap_df["aldi_divergence_pct"].std())
    st.markdown(
        f"**Cross-level spread (std of divergence):** joint **{std_joint:.2f}** · "
        f"AGS **{std_ags:.2f}** · ALDi **{std_aldi:.2f}**"
    )

    if penalty_df is not None:
        merged_pen = overlap_df.merge(penalty_df, on="level", how="inner").sort_values("level")
        ags_rho, ags_p, ags_n = spearman_stats.get(
            "ags_divergence_pct", (None, None, len(merged_pen))
        )

        fig = _plot_marginal_divergences_vs_dist_penalty(
            overlap_df,
            penalty_df,
            spearman_stats=spearman_stats,
        )
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("#### AGS divergence vs dev penalty (per gold level)")
        fig_scatter = _plot_ags_divergence_vs_penalty_scatter(
            merged_pen,
            spearman_r=ags_rho,
            spearman_p=ags_p,
            n=ags_n,
        )
        st.pyplot(fig_scatter, use_container_width=True)
        plt.close(fig_scatter)

        rows = []
        for col, label, _ in _DIVERGENCE_METRICS:
            rho, pval, n = spearman_stats.get(col, (None, None, 0))
            rows.append(
                {
                    "Metric": label,
                    "ρ vs dev penalty": f"{rho:+.3f}" if rho is not None else "—",
                    "p": f"{pval:.3g}" if pval is not None else "—",
                    "n": n,
                }
            )
        import pandas as pd

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        best = max(
            (s for s in spearman_stats.items() if s[1][0] is not None),
            key=lambda item: abs(item[1][0]),
            default=None,
        )
        if best is not None:
            col, (rho, pval, _n) = best
            label = next(l for c, l, _ in _DIVERGENCE_METRICS if c == col)
            st.markdown(
                f"Strongest |ρ| among splits: **{label}** (ρ = **{rho:+.3f}**, p = {pval:.3g}). "
                "Negative ρ → higher divergence tends to co-occur with dist **losing** on dev."
            )
    else:
        import matplotlib.pyplot as plt

        plot_df = overlap_df.sort_values("level")
        x = np.arange(len(plot_df))
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        for ax, (col, label, color) in zip(axes, _DIVERGENCE_METRICS):
            ax.bar(
                x,
                plot_df[col],
                color=color,
                edgecolor="white",
                linewidth=0.5,
                width=0.72,
            )
            ax.set_ylabel(f"{label} %")
            ax.grid(True, axis="y", alpha=0.25)
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels([str(int(v)) for v in plot_df["level"]])
        axes[-1].set_xlabel("Readability level")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        if pen_err:
            st.info(f"Dev penalty overlay unavailable: {pen_err}")

    with st.expander("Per-level overlap table"):
        show = overlap_df.sort_values("level").copy()
        if penalty_df is not None:
            show = show.merge(penalty_df, on="level", how="left")
        st.dataframe(
            show.rename(
                columns={
                    "level": "Level",
                    "n_barec": "BAREC n",
                    "n_tarab": "Tarab n",
                    "joint_divergence_pct": "Joint div %",
                    "ags_divergence_pct": "AGS div %",
                    "aldi_divergence_pct": "ALDi div %",
                    "overlap_pct": "Joint overlap %",
                    "delta_penalty": "Dev penalty savings",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def _aldi_ags_heatmap_cmap():
    """White → red slice of RdBu_r (same family as confusion Δ heatmap)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colors = plt.cm.RdBu_r(np.linspace(0.52, 1.0, 256))
    return ListedColormap(colors)


def _plot_aldi_ags_heatmap(
    pair_pct,
    *,
    title: str,
    cmap,
    vmax: float | None = None,
) -> Any:
    import matplotlib.pyplot as plt
    import numpy as np

    if vmax is None:
        vmax = float(np.max(pair_pct)) if pair_pct.size else 1.0
    vmax = max(vmax, 1.0)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(
        pair_pct,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        aspect="auto",
        origin="upper",
    )
    ax.set_xticks(range(len(_AGS_AXIS_LABELS)))
    ax.set_yticks(range(len(_ALDI_AXIS_LABELS)))
    ax.set_xticklabels(list(_AGS_AXIS_LABELS))
    ax.set_yticklabels(list(_ALDI_AXIS_LABELS))
    ax.set_xlabel("AGS label", fontsize=10)
    ax.set_ylabel("ALDi label", fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)

    for r in range(pair_pct.shape[0]):
        for c in range(pair_pct.shape[1]):
            v = float(pair_pct[r, c])
            if v <= 0.0:
                continue
            text_color = "white" if v > vmax * 0.45 else "#1a1a1a"
            ax.text(
                c,
                r,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% of rows")
    fig.tight_layout()
    return fig


def render_dist_155k_aldi_ags_heatmaps(split_df) -> None:
    """Side-by-side ALDi × AGS heatmaps for BAREC train vs dist_155K Tarab pseudo rows."""
    import matplotlib.pyplot as plt

    if "type" not in split_df.columns:
        split_df = _load_split_pretrain_df(str(DIST_155K_SPLIT_CSV))
        if split_df is None:
            st.warning(f"Could not reload `{DIST_155K_SPLIT_CSV.name}`.")
            return

    if "aldi_label" not in split_df.columns or "ags_label" not in split_df.columns:
        st.info(
            "ALDi/AGS columns missing on the split CSV. "
            "Run `python3 scripts/fill_splits_aldi_ags.py`."
        )
        return

    barec = _compute_aldi_ags_pair_pct(split_df, source="barec")
    tarab = _compute_aldi_ags_pair_pct(split_df, source="tarab")
    if barec is None and tarab is None:
        st.info("No valid ALDi × AGS pairs in the dist_155K split.")
        return

    st.markdown("#### ALDi × AGS label mix")

    heatmap_cmap = _aldi_ags_heatmap_cmap()
    vmax = None
    if barec is not None and tarab is not None:
        import numpy as np

        vmax = float(max(np.max(barec[0]), np.max(tarab[0])))

    col_barec, col_tarab = st.columns(2)
    if barec is not None:
        pct, n = barec
        with col_barec:
            fig = _plot_aldi_ags_heatmap(
                pct,
                title=f"BAREC train (n={n:,})",
                cmap=heatmap_cmap,
                vmax=vmax,
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    else:
        col_barec.info("No BAREC ALDi × AGS data.")

    if tarab is not None:
        pct, n = tarab
        with col_tarab:
            fig = _plot_aldi_ags_heatmap(
                pct,
                title=f"dist_155K Tarab pseudo (n={n:,})",
                cmap=heatmap_cmap,
                vmax=vmax,
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    else:
        col_tarab.info("No Tarab ALDi × AGS data.")


def _tarab_rows_with_type(split_df):
    if "type" not in split_df.columns:
        return None
    tarab = split_df.loc[split_df["source"] == "tarab"].copy()
    tarab["type"] = tarab["type"].astype(str).str.strip().str.lower()
    tarab = tarab[tarab["type"].isin(("song", "poem"))]
    return tarab if not tarab.empty else None


def _vocab_for_source_level(split_df, *, source: str, level: int) -> set[str]:
    from tarab_model_experimentation.pseudo_label_oov import vocab_from_texts

    sub = split_df[(split_df["source"] == source) & (split_df["label"] == level)]
    if sub.empty:
        return set()
    return vocab_from_texts(sub["Sentence"].astype(str))


@st.cache_data(show_spinner=False)
def _build_per_level_barec_vocab_coverage_by_tarab(
    split_csv: str, *, _schema_version: int = 2
):
    import pandas as pd

    split_df = _load_split_pretrain_df(split_csv)
    if split_df is None:
        return None, "Split CSV not found."

    rows: list[dict[str, Any]] = []
    for level in LEVELS:
        barec_sub = split_df[
            (split_df["source"] == "barec") & (split_df["label"] == level)
        ]
        if barec_sub.empty:
            continue
        barec_vocab = _vocab_for_source_level(split_df, source="barec", level=level)
        if not barec_vocab:
            continue
        tarab_vocab = _vocab_for_source_level(split_df, source="tarab", level=level)
        shared = barec_vocab & tarab_vocab
        barec_n = len(barec_vocab)
        shared_n = len(shared)
        covered_pct = 100.0 * shared_n / barec_n
        rows.append(
            {
                "level": level,
                "barec_support": int(len(barec_sub)),
                "barec_types": barec_n,
                "tarab_types": len(tarab_vocab),
                "shared_types": shared_n,
                "barec_covered_pct": covered_pct,
                "barec_not_in_tarab_pct": 100.0 - covered_pct,
            }
        )
    if not rows:
        return None, "No BAREC vocabulary at any readability level in the split."
    return pd.DataFrame(rows), None


def _plot_barec_vocab_coverage_by_tarab_at_level(coverage_df):
    """100% stacked bar: BAREC word types at level k also seen in Tarab at k."""
    import matplotlib.pyplot as plt
    import numpy as np

    plot_df = coverage_df.set_index("level").reindex(LEVELS)
    x = np.arange(len(LEVELS))
    covered = plot_df["barec_covered_pct"].fillna(0.0).to_numpy(dtype=float)
    not_covered = plot_df["barec_not_in_tarab_pct"].fillna(0.0).to_numpy(dtype=float)
    has_barec = plot_df["barec_types"].notna().to_numpy()

    covered_color = "#2a9d8f"
    gap_color = "#d5d5d5"

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(
        x[has_barec],
        covered[has_barec],
        width=0.72,
        label="In Tarab at same level",
        color=covered_color,
        alpha=0.9,
    )
    ax.bar(
        x[has_barec],
        not_covered[has_barec],
        width=0.72,
        bottom=covered[has_barec],
        label="BAREC-only at level (not in Tarab pseudo)",
        color=gap_color,
        alpha=0.95,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(lev) for lev in LEVELS], fontsize=9)
    ax.set_xlabel("Readability level", fontsize=10)
    ax.set_ylabel("% of BAREC word types at level", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title(
        "BAREC vocabulary covered by Tarab at same readability level (dist_155K split)",
        fontsize=11,
    )
    ax.grid(True, axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


def _plot_tarab_song_poem_by_level(tarab_df):
    """Stacked bar: % song vs % poem at each readability level (Tarab pseudo only)."""
    import matplotlib.pyplot as plt
    import numpy as np

    song_color = "#2a6f97"
    poem_color = "#f4a261"

    x = np.arange(len(LEVELS))
    song_pct: list[float] = []
    poem_pct: list[float] = []
    counts: list[int] = []
    for lev in LEVELS:
        sub = tarab_df[tarab_df["label"] == lev]
        n = len(sub)
        counts.append(n)
        if n == 0:
            song_pct.append(0.0)
            poem_pct.append(0.0)
            continue
        song_n = int((sub["type"] == "song").sum())
        poem_n = int((sub["type"] == "poem").sum())
        song_pct.append(100.0 * song_n / n)
        poem_pct.append(100.0 * poem_n / n)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x, song_pct, width=0.72, label="Song", color=song_color, alpha=0.9)
    ax.bar(x, poem_pct, width=0.72, bottom=song_pct, label="Poem", color=poem_color, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(lev) for lev in LEVELS], fontsize=9)
    ax.set_xlabel("Readability level", fontsize=10)
    ax.set_ylabel("% of Tarab pseudo rows at level", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("Song vs poem by readability level (dist_155K Tarab pseudo)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


def render_dist_155k_song_poem_section() -> None:
    """Tarab pseudo song/poem totals and mix per readability level."""
    import matplotlib.pyplot as plt

    split_df = _load_split_pretrain_df(str(DIST_155K_SPLIT_CSV))
    if split_df is None:
        st.warning(f"Missing dist_155K split. Need `{DIST_155K_SPLIT_CSV.name}`.")
        return

    tarab = _tarab_rows_with_type(split_df)
    if tarab is None:
        st.warning(
            "Tarab `type` (song/poem) not in split — run `scripts/fill_splits_aldi_ags.py`."
        )
        return

    n_tarab = len(tarab)
    counts = tarab["type"].value_counts()
    song_n = int(counts.get("song", 0))
    poem_n = int(counts.get("poem", 0))

    st.markdown("### Tarab song vs poem")

    st.markdown("**Overall**")
    c_song, c_poem = st.columns(2)
    c_song.metric("Song", f"{100.0 * song_n / n_tarab:.1f}%", help=f"{song_n:,} / {n_tarab:,} rows")
    c_poem.metric("Poem", f"{100.0 * poem_n / n_tarab:.1f}%", help=f"{poem_n:,} / {n_tarab:,} rows")

    st.markdown("**By readability level**")
    fig = _plot_tarab_song_poem_by_level(tarab)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_dist_155k_text_length_section() -> None:
    """Character length and word count boxplots by level (BAREC vs Tarab, dist_155K)."""
    dist_df = _load_split_pretrain_df(str(DIST_155K_SPLIT_CSV))
    if dist_df is None:
        st.warning(
            f"Missing dist_155K split for length charts. Need `{DIST_155K_SPLIT_CSV.name}`."
        )
        return

    dist_df = _prepare_sentence_length_columns(dist_df)
    dist_name = "dist_155K"

    st.markdown("### Text length by readability level")
    st.caption(
        "**Characters** = length of `Sentence`; **words** = whitespace-separated token count."
    )

    st.markdown("#### Character length")
    c_barec, c_tarab = st.columns(2)
    _show_text_length_boxplot(
        c_barec,
        dist_df,
        source="barec",
        metric="char_length",
        metric_label="Characters per sentence",
        dist_name=dist_name,
        facecolor=_BASELINE_COLOR,
        empty_msg="No BAREC rows for character length.",
    )
    _show_text_length_boxplot(
        c_tarab,
        dist_df,
        source="tarab",
        metric="char_length",
        metric_label="Characters per sentence",
        dist_name=dist_name,
        facecolor=TARAB_SHARE_COLOR,
        empty_msg="No Tarab rows for character length.",
    )

    st.markdown("#### Word count")
    w_barec, w_tarab = st.columns(2)
    _show_text_length_boxplot(
        w_barec,
        dist_df,
        source="barec",
        metric="word_count",
        metric_label="Words per sentence",
        dist_name=dist_name,
        facecolor=_BASELINE_COLOR,
        empty_msg="No BAREC rows for word count.",
    )
    _show_text_length_boxplot(
        w_tarab,
        dist_df,
        source="tarab",
        metric="word_count",
        metric_label="Words per sentence",
        dist_name=dist_name,
        facecolor=TARAB_SHARE_COLOR,
        empty_msg="No Tarab rows for word count.",
    )

    from tarab_model_experimentation.presentation_insights import (
        render_text_length_insight,
    )

    render_text_length_insight()

    render_dist_155k_aldi_ags_heatmaps(dist_df)


def render_class19_investigation_section(*, log_files: list[str]) -> None:
    # for title, split_csv, dist_name, dev_labels in _INVESTIGATION_SPECS:
    #     _render_one_class19_block(
    #         log_files=log_files,
    #         title=title,
    #         split_csv=split_csv,
    #         dist_name=dist_name,
    #         dev_compare_labels=dev_labels,
    #     )

    from tarab_model_experimentation.pseudo_label_oov import render_pseudo_label_oov_section

    render_pseudo_label_oov_section()
