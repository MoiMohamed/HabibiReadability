from __future__ import annotations

from typing import Any

import streamlit as st

from tarab_model_experimentation.constants import (
    BAREC_PARQUET_DIR,
    DEV_PREDICTIONS_DIR,
    LENGTH_MATCHED_DEV_PREDICTIONS_CSV,
    MIN_L8_DEV_PREDICTIONS_CSV,
)

_DEV_PARQUET = BAREC_PARQUET_DIR / "dev.parquet"

# Local dev exports (row order matches dev.parquet).
_DEV_PREDICTION_CSV_BY_LABEL: dict[str, Path] = {
    "baseline": DEV_PREDICTIONS_DIR / "dev_bert-base-arabertv02_local_19levels_baseline.csv",
    "dist_155K": DEV_PREDICTIONS_DIR / "dev_bert-base-arabertv02_local_19levels_155k.csv",
    "length_matched": LENGTH_MATCHED_DEV_PREDICTIONS_CSV,
    "minL8": MIN_L8_DEV_PREDICTIONS_CSV,
}


def _csv_text_column(df) -> str | None:
    for name in ("text", "Text", "Sentence", "sentence"):
        if name in df.columns:
            return name
    return None


def dev_predictions_csv_path(label: str, epoch: float | None = None) -> Path:
    """Path for a model's dev predictions (epoch ignored for bundled local exports)."""
    if label in _DEV_PREDICTION_CSV_BY_LABEL:
        return _DEV_PREDICTION_CSV_BY_LABEL[label]
    if epoch is not None:
        return DEV_PREDICTIONS_DIR / f"{label}_epoch{int(round(epoch))}_dev.csv"
    return DEV_PREDICTIONS_DIR / f"{label}_dev.csv"


@st.cache_data(show_spinner=False)
def load_dev_gold_frame():
    import pandas as pd

    if not _DEV_PARQUET.exists():
        return None
    df = pd.read_parquet(_DEV_PARQUET, columns=["ID", "Sentence", "Readability_Level_19"])
    df = df.rename(columns={"Readability_Level_19": "true_level"})
    df["true_level"] = df["true_level"].astype(int)
    df["Sentence"] = df["Sentence"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_dev_predictions_for_label(label: str):
    import pandas as pd

    path = dev_predictions_csv_path(label)
    if not path.exists():
        return None

    df = pd.read_csv(path, encoding="utf-8")
    pred_col = next(
        (c for c in ("pred_level", "predicted_level", "prediction", "pred") if c in df.columns),
        None,
    )
    if pred_col is None:
        raise ValueError(f"{path.name} must include prediction (or pred_level / pred).")

    text_col = _csv_text_column(df)

    gold = load_dev_gold_frame()
    if gold is None:
        return None

    if "ID" in df.columns:
        keep = ["ID", pred_col]
        if text_col is not None:
            keep.append(text_col)
        out = df[keep].copy()
        out = out.rename(columns={pred_col: "pred_level"})
        out["pred_level"] = out["pred_level"].astype(int)
        if text_col is not None:
            out = out.rename(columns={text_col: "text"})
            out["text"] = out["text"].astype(str)
        return out

    if len(df) != len(gold):
        raise ValueError(
            f"{path.name} has {len(df)} rows but dev.parquet has {len(gold)}; "
            "expected same row order or an ID column."
        )

    label_col = next((c for c in ("label", "true_level", "Readability_Level_19") if c in df.columns), None)
    if label_col is not None:
        csv_labels = df[label_col].astype(int).to_numpy()
        if not (csv_labels == gold["true_level"].to_numpy()).all():
            raise ValueError(f"{path.name} labels do not match dev.parquet row order.")

    out = gold[["ID"]].copy()
    out["pred_level"] = df[pred_col].astype(int).to_numpy()
    if text_col is not None:
        out["text"] = df[text_col].astype(str).to_numpy()
    return out


READABILITY_LEVELS = 19
_QWK_PENALTY_SCALE = (READABILITY_LEVELS - 1) ** 2


def load_merged_dev_predictions(compare_labels: tuple[str, str]):
    """Gold dev rows with baseline and experiment predictions merged on ID."""
    import pandas as pd

    baseline_label, experiment_label = compare_labels
    gold = load_dev_gold_frame()
    if gold is None:
        return None, f"Dev set not found at `{_DEV_PARQUET}`."

    missing = [
        label
        for label in compare_labels
        if not dev_predictions_csv_path(label).exists()
    ]
    if missing:
        return None, missing_dev_predictions_message(compare_labels)

    base_preds = load_dev_predictions_for_label(baseline_label)
    exp_preds = load_dev_predictions_for_label(experiment_label)
    if base_preds is None or exp_preds is None:
        return None, missing_dev_predictions_message(compare_labels)

    base = base_preds.rename(columns={"pred_level": "pred_baseline"})
    exp = exp_preds.rename(columns={"pred_level": "pred_dist"})
    if "text" in exp.columns:
        exp = exp.drop(columns=["text"])

    merged = gold.merge(base, on="ID", how="inner").merge(exp, on="ID", how="inner")
    if len(merged) != len(gold):
        return None, "Prediction CSVs did not align with the full dev set."

    if "text" in merged.columns:
        merged["Sentence"] = merged["text"].astype(str)
        merged = merged.drop(columns=["text"])

    return merged, None


def build_per_class_qwk_penalty_contribution(
    compare_labels: tuple[str, str],
    *,
    n_classes: int = READABILITY_LEVELS,
):
    """
    Actual squared-error penalty per class (dist vs baseline).

    For true class c: penalty = Σ(pred − true)² on those samples, scaled by (K−1)².
    bar = penalty_baseline − penalty_dist (absolute, not ÷ support).
    Positive = dist made smaller errors on class c than baseline; negative = dist worse.

    gain_positive / loss_negative split the same rows: per dev item with true=c,
    add (base_err² − dist_err²)/(K−1)² to gain_positive if positive else loss_negative.
    delta_penalty = gain_positive + loss_negative (net for the class).
    """
    import pandas as pd

    merged, err = load_merged_dev_predictions(compare_labels)
    if err:
        return None, err
    if merged is None or merged.empty:
        return pd.DataFrame(), None

    scale = (n_classes - 1) ** 2
    rows: list[dict[str, Any]] = []
    for level, grp in merged.groupby("true_level", sort=True):
        n = int(len(grp))
        if n == 0:
            continue
        true = grp["true_level"].to_numpy(dtype=int)
        base_err = grp["pred_baseline"].to_numpy(dtype=int) - true
        dist_err = grp["pred_dist"].to_numpy(dtype=int) - true
        row_delta = (base_err * base_err - dist_err * dist_err).astype(float) / scale
        gain_positive = float(row_delta[row_delta > 0].sum())
        loss_negative = float(row_delta[row_delta < 0].sum())
        penalty_baseline = float((base_err * base_err).sum()) / scale
        penalty_dist = float((dist_err * dist_err).sum()) / scale
        delta_penalty = penalty_baseline - penalty_dist
        rows.append(
            {
                "level": int(level),
                "support": n,
                "penalty_baseline": penalty_baseline,
                "penalty_dist": penalty_dist,
                "delta_penalty": delta_penalty,
                "gain_positive": gain_positive,
                "loss_negative": loss_negative,
            }
        )
    return pd.DataFrame(rows), None


def build_per_class_gain_loss_vs_baseline(compare_labels: tuple[str, str]):
    """
    Per gold class: dist vs baseline on squared error (QWK-aligned ingredient).

    delta_penalty = Σ(base_err² − dist_err²) / (K−1)² on rows with true=level.
    Positive → dist made smaller errors (gain). Negative → dist made larger errors (loss).

    summary includes one QWK counterfactual: revert to baseline on every row where
    dist increased squared error (all negative row-level components), keep dist elsewhere.
    """
    import numpy as np
    from sklearn.metrics import cohen_kappa_score

    contrib, err = build_per_class_qwk_penalty_contribution(compare_labels)
    if err:
        return None, None, err
    if contrib is None or contrib.empty:
        return pd.DataFrame(), None, None

    merged, err = load_merged_dev_predictions(compare_labels)
    if err or merged is None:
        return None, None, err

    labels = merged["true_level"].to_numpy(dtype=int)
    preds_base = merged["pred_baseline"].to_numpy(dtype=int)
    preds_dist = merged["pred_dist"].to_numpy(dtype=int)
    qwk = lambda p: float(cohen_kappa_score(labels, p, weights="quadratic"))
    qwk_dist = qwk(preds_dist)
    qwk_base = qwk(preds_base)

    loss_levels = set(contrib.loc[contrib["delta_penalty"] < 0, "level"].astype(int))
    gain_levels = set(contrib.loc[contrib["delta_penalty"] > 0, "level"].astype(int))

    base_err = preds_base - labels
    dist_err = preds_dist - labels
    row_loss_mask = (dist_err * dist_err) > (base_err * base_err)
    p_fix_losses = preds_dist.copy()
    p_fix_losses[row_loss_mask] = preds_base[row_loss_mask]

    net_positive_mask = contrib["delta_penalty"] > 0
    masked_harm = contrib.loc[net_positive_mask].sort_values("loss_negative")
    summary = {
        "qwk_baseline": qwk_base,
        "qwk_dist": qwk_dist,
        "qwk_fix_losses": qwk(p_fix_losses),
        "n_rows_loss": int(row_loss_mask.sum()),
        "n_gain_classes": len(gain_levels),
        "n_loss_classes": len(loss_levels),
        "gain_pool": float(contrib.loc[contrib["delta_penalty"] > 0, "delta_penalty"].sum()),
        "loss_pool": float(contrib.loc[contrib["delta_penalty"] < 0, "delta_penalty"].sum()),
        "row_gain_pool": float(contrib["gain_positive"].sum()),
        "row_loss_pool": float(contrib["loss_negative"].sum()),
        "masked_harm_levels": [
            {
                "level": int(r["level"]),
                "gain_positive": float(r["gain_positive"]),
                "loss_negative": float(r["loss_negative"]),
                "delta_penalty": float(r["delta_penalty"]),
            }
            for _, r in masked_harm.head(4).iterrows()
            if float(r["loss_negative"]) < -0.05
        ],
    }
    return contrib, summary, None


def build_isolated_class_qwk_impact(compare_labels: tuple[str, str]):
    """
    For each true level k, compute the isolated ΔQWK from reverting class k to baseline:

        isolated_impact_k = QWK(dist with class k → baseline) − QWK(dist)

    Positive  → reverting k improves QWK  → dist is hurting you on class k (loss)
    Negative  → reverting k costs QWK     → dist is genuinely helping on class k (gain)

    Also returns joint counterfactuals:
        qwk_dist, qwk_baseline,
        qwk_fix_losses  (revert all loss classes jointly),
        qwk_keep_gains  (revert all gain classes jointly → lower bound without any gain)
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score

    merged, err = load_merged_dev_predictions(compare_labels)
    if err:
        return None, None, err
    if merged is None or merged.empty:
        return pd.DataFrame(), None, None

    labels = merged["true_level"].to_numpy(dtype=int)
    preds_base = merged["pred_baseline"].to_numpy(dtype=int)
    preds_dist = merged["pred_dist"].to_numpy(dtype=int)

    qwk = lambda p: float(cohen_kappa_score(labels, p, weights="quadratic"))
    qwk_dist = qwk(preds_dist)
    qwk_base = qwk(preds_base)

    levels = sorted(merged["true_level"].unique())
    rows: list[dict[str, Any]] = []
    for lev in levels:
        mask = labels == lev
        p_reverted = preds_dist.copy()
        p_reverted[mask] = preds_base[mask]
        q_reverted = qwk(p_reverted)
        impact = q_reverted - qwk_dist
        rows.append(
            {
                "level": int(lev),
                "support": int(mask.sum()),
                "isolated_qwk_impact": round(impact, 6),
                "direction": "loss" if impact > 0 else "gain",
            }
        )

    df = pd.DataFrame(rows)

    # Joint: revert all loss classes (positive impact → dist hurts there)
    loss_levels = set(df.loc[df["isolated_qwk_impact"] > 0, "level"])
    gain_levels = set(df.loc[df["isolated_qwk_impact"] < 0, "level"])

    p_fix_losses = preds_dist.copy()
    for lev in loss_levels:
        mask = labels == lev
        p_fix_losses[mask] = preds_base[mask]
    qwk_fix_losses = qwk(p_fix_losses)

    p_revert_gains = preds_dist.copy()
    for lev in gain_levels:
        mask = labels == lev
        p_revert_gains[mask] = preds_base[mask]
    qwk_revert_gains = qwk(p_revert_gains)

    summary = {
        "qwk_baseline": qwk_base,
        "qwk_dist": qwk_dist,
        "qwk_fix_losses": qwk_fix_losses,
        "qwk_revert_gains": qwk_revert_gains,
        "n_loss_classes": len(loss_levels),
        "n_gain_classes": len(gain_levels),
    }
    return df, summary, None


def missing_dev_predictions_message(compare_labels: tuple[str, str]) -> str:
    lines = [
        "Per-example analysis needs dev prediction CSVs (with `text` column for examples).",
        "",
        f"Expected under `{DEV_PREDICTIONS_DIR}`:",
    ]
    for label in compare_labels:
        path = dev_predictions_csv_path(label)
        lines.append(f"- `{path.name}` ({label})")
    return "\n".join(lines)


def far_off_inversion_pattern_counts(
    compare_labels: tuple[str, str],
) -> dict[str, int] | None:
    """
    Counts on |err|≥7 rows where dist is worse than baseline (same filter as the table).
    """
    import numpy as np

    merged, err = load_merged_dev_predictions(compare_labels)
    if err or merged is None:
        return None

    true = merged["true_level"].to_numpy(dtype=int)
    pred = merged["pred_dist"].to_numpy(dtype=int)
    base = merged["pred_baseline"].to_numpy(dtype=int)
    dist_off = np.abs(pred - true)
    base_off = np.abs(base - true)
    extra = (dist_off >= 7) & (base_off < dist_off)
    if not extra.any():
        return None

    return {
        "gold_12_15_pred_2_3": int(((true >= 12) & (true <= 15) & (pred <= 3) & extra).sum()),
        "gold_12_16_pred_5": int(((true >= 12) & (true <= 16) & (pred == 5) & extra).sum()),
        "gold_5_pred_12_plus": int(((true == 5) & (pred >= 12) & extra).sum()),
    }


def far_off_mistake_insight_stats(compare_labels: tuple[str, str]) -> dict[str, Any] | None:
    """Summary stats for the far-off mistakes table (dist worse than baseline, |err|≥7)."""
    import numpy as np
    from sklearn.metrics import cohen_kappa_score

    merged, err = load_merged_dev_predictions(compare_labels)
    if err or merged is None:
        return None

    labels = merged["true_level"].to_numpy(dtype=int)
    preds_base = merged["pred_baseline"].to_numpy(dtype=int)
    preds_dist = merged["pred_dist"].to_numpy(dtype=int)
    dist_off = np.abs(preds_dist - labels)
    base_off = np.abs(preds_base - labels)
    extra = (dist_off >= 7) & (base_off < dist_off)
    if not extra.any():
        return None

    qwk = lambda y, p: float(cohen_kappa_score(y, p, weights="quadratic"))
    qwk_base = qwk(labels, preds_base)
    qwk_dist = qwk(labels, preds_dist)
    preds_fixed = preds_dist.copy()
    preds_fixed[extra] = preds_base[extra]
    qwk_if_fixed = qwk(labels, preds_fixed)

    sentences = merged["Sentence"].astype(str)
    short = int((sentences.str.len() < 50).to_numpy()[extra].sum())

    return {
        "n_extra": int(extra.sum()),
        "baseline_exact": int(((base_off == 0) & extra).sum()),
        "true_l12": int(((labels == 12) & extra).sum()),
        "true_l5_pred_high": int(((labels == 5) & extra & (preds_dist >= 12)).sum()),
        "short_under_50_chars": short,
        "mean_dist_err": float(dist_off[extra].mean()),
        "mean_base_err": float(base_off[extra].mean()),
        "qwk_baseline": qwk_base,
        "qwk_dist": qwk_dist,
        "qwk_if_extra_reverted_to_baseline": qwk_if_fixed,
        "dqwk_fix_extra": qwk_if_fixed - qwk_dist,
    }


def build_far_off_mistake_examples_table(
    loaded: dict[str, dict[str, Any]],
    compare_labels: tuple[str, str],
    *,
    min_distance: int = 7,
):
    import pandas as pd

    merged, err = load_merged_dev_predictions(compare_labels)
    if err:
        return None, err
    if merged is None:
        return None, missing_dev_predictions_message(compare_labels)
    merged["levels_off"] = (merged["pred_dist"] - merged["true_level"]).abs()
    merged["baseline_levels_off"] = (merged["pred_baseline"] - merged["true_level"]).abs()
    # Extra far-off harm from dist: dist is ≥min_distance off, baseline was less wrong.
    extra = merged[
        (merged["levels_off"] >= min_distance)
        & (merged["baseline_levels_off"] < merged["levels_off"])
    ].copy()
    if extra.empty:
        return pd.DataFrame(), None

    out = extra.rename(
        columns={
            "Sentence": "example_text",
            "pred_dist": "pred_by_dist155k",
            "pred_baseline": "pred_by_baseline",
            "levels_off": "levels_off_dist",
            "baseline_levels_off": "levels_off_baseline",
        }
    )[
        [
            "pred_by_dist155k",
            "pred_by_baseline",
            "true_level",
            "levels_off_baseline",
            "levels_off_dist",
            "example_text",
        ]
    ]
    sort_cols = [
        "pred_by_dist155k",
        "pred_by_baseline",
        "true_level",
        "levels_off_baseline",
        "levels_off_dist",
    ]
    return out.sort_values(sort_cols, ascending=True), None
