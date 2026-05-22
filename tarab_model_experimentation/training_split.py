from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from tarab_model_experimentation.constants import SPLITS_DIR


@st.cache_data(show_spinner=False)
def load_barec_train_readability_counts():
    return {
        1: 333,
        2: 333,
        3: 1139,
        4: 587,
        5: 2646,
        6: 1206,
        7: 4152,
        8: 4529,
        9: 1597,
        10: 7741,
        11: 4041,
        12: 11318,
        13: 3252,
        14: 8573,
        15: 2016,
        16: 866,
        17: 364,
        18: 67,
        19: 85,
    }


def confidence_bin_edges_20():
    import numpy as np

    edges = np.linspace(0.0, 1.0, 21)
    labels = []
    for i in range(20):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i < 19:
            labels.append(f"[{lo:.2f}, {hi:.2f})")
        else:
            labels.append(f"[{lo:.2f}, {hi:.2f}]")
    return edges, labels


def assign_confidence_bin_idx(conf_series):
    import numpy as np
    import pandas as pd

    c = pd.to_numeric(conf_series, errors="coerce")
    c = c.clip(lower=0.0, upper=1.0)
    idx = np.minimum(np.floor(c.to_numpy(dtype=np.float64) * 20.0).astype(np.int64), 19)
    return pd.Series(idx, index=conf_series.index, dtype="int64")


# _LOG_SPLIT_CSV_ALIASES = {
#     "barec_tarab_2x_55k_match_distribution_155k_wo_pseudolabel19": (
#         "barec_tarab_2X_55k_match_distribution_155k_wo_class19.csv"
#     ),
# }
_LOG_SPLIT_CSV_ALIASES: dict[str, str] = {}


def split_csv_for_experiment_log(log_filename: str) -> str | None:
    """
    Map a training log `*.out` name to a `data/splits/*.csv` training file, e.g.
    `barec_tarab_0.5X_55k_match_distribution_81k-14975941.out`
    -> `barec_tarab_0.5X_55k_match_distribution_81k.csv`
    """
    if not log_filename.endswith(".out"):
        return None
    stem = Path(log_filename).stem
    stem = re.sub(r"-\d+$", "", stem)
    if not SPLITS_DIR.exists():
        return None

    alias = _LOG_SPLIT_CSV_ALIASES.get(stem.lower())
    if alias and (SPLITS_DIR / alias).exists():
        return alias

    exact = SPLITS_DIR / f"{stem}.csv"
    if exact.exists():
        return exact.name
    for p in sorted(SPLITS_DIR.glob("*.csv")):
        if p.stem == stem:
            return p.name
    return None


@st.cache_data(show_spinner=True)
def build_training_split_wide_table(rel_csv: str):
    import numpy as np
    import pandas as pd

    path = SPLITS_DIR / rel_csv
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: data/splits/{rel_csv}")

    cols0 = list(pd.read_csv(path, nrows=0, encoding="utf-8").columns)
    need = ["Readability", "source", "readability_confidence"]
    missing = [c for c in need if c not in cols0]
    if missing:
        raise ValueError(f"{rel_csv} missing columns {missing}. Found: {cols0}")

    df = pd.read_csv(path, usecols=need, encoding="utf-8")
    df["Readability"] = pd.to_numeric(df["Readability"], errors="coerce")
    df["readability_confidence"] = pd.to_numeric(df["readability_confidence"], errors="coerce")
    df = df.dropna(subset=["Readability", "readability_confidence"]).copy()
    df["Readability"] = df["Readability"].astype(int)
    df = df[(df["Readability"] >= 1) & (df["Readability"] <= 19)].copy()

    n_rows = int(len(df))
    src = df["source"].astype(str).str.strip().str.lower()
    df["is_tarab"] = (src == "tarab").astype(np.int64)
    df["is_barec"] = (src == "barec").astype(np.int64)
    df["bin_idx"] = assign_confidence_bin_idx(df["readability_confidence"])

    _, bin_labels = confidence_bin_edges_20()

    level_mean_conf = (
        df.groupby("Readability", sort=True)["readability_confidence"].mean().reindex(range(1, 20))
    )
    global_mean_conf = float(df["readability_confidence"].mean())

    tarab_only = df[df["is_tarab"] == 1]
    barec_only = df[df["is_barec"] == 1]
    tarab_level_in_split = (
        tarab_only.groupby("Readability", sort=True).size().reindex(range(1, 20), fill_value=0).astype(np.int64)
    )
    barec_level_in_split = (
        barec_only.groupby("Readability", sort=True).size().reindex(range(1, 20), fill_value=0).astype(np.int64)
    )
    level_row_counts = (
        df.groupby("Readability", sort=True).size().reindex(range(1, 20), fill_value=0).astype(np.int64)
    )

    if len(tarab_only) == 0:
        tarab_pivot = pd.DataFrame(0, index=list(range(1, 20)), columns=list(range(20)), dtype=np.int64)
    else:
        tarab_bin = (
            tarab_only.groupby(["Readability", "bin_idx"], sort=True).size().reset_index(name="n")
        )
        tarab_pivot = tarab_bin.pivot(index="Readability", columns="bin_idx", values="n")
        tarab_pivot = tarab_pivot.reindex(index=list(range(1, 20)), columns=list(range(20)), fill_value=0)
        tarab_pivot = tarab_pivot.fillna(0).astype(np.int64)

    barec_ref = load_barec_train_readability_counts()
    barec_ref_total = int(sum(barec_ref.values()))
    barec_total_split = int(barec_level_in_split.sum())
    tarab_total_split = int(tarab_level_in_split.sum())

    if len(tarab_only) == 0:
        pool_tarab = np.zeros(20, dtype=np.int64)
    else:
        tb = tarab_only.groupby("bin_idx", sort=True).size().reindex(range(20), fill_value=0).astype(np.int64)
        pool_tarab = tb.to_numpy(dtype=np.int64)

    col_sums = tarab_pivot.sum(axis=0).to_numpy(dtype=np.int64) + pool_tarab
    nonzero_bins = [bi for bi in range(20) if int(col_sums[bi]) > 0]
    bins_desc = sorted(nonzero_bins, reverse=True)
    bin_col_names = [f"bin: {bin_labels[bi]}" for bi in bins_desc]

    col_order = bin_col_names + [
        "count_tarab",
        "count_barec",
        "old_distribution_%",
        "new_distribution_%",
        "avg_readability_confidence",
    ]

    rows: list[dict[str, float | int]] = []
    for lev in range(1, 20):
        row: dict[str, float | int] = {"readability_level": lev}
        for bi in bins_desc:
            bl = bin_labels[bi]
            row[f"bin: {bl}"] = int(tarab_pivot.loc[lev, bi])
        ct = int(tarab_level_in_split.loc[lev])
        cb = int(barec_level_in_split.loc[lev])
        row["count_tarab"] = ct
        row["count_barec"] = cb
        row["old_distribution_%"] = (
            (100.0 * float(barec_ref.get(lev, 0)) / float(barec_ref_total)) if barec_ref_total else 0.0
        )
        n_at_level = int(level_row_counts.loc[lev])
        row["new_distribution_%"] = (100.0 * float(n_at_level) / float(n_rows)) if n_rows else 0.0
        mu = level_mean_conf.loc[lev]
        row["avg_readability_confidence"] = float(mu) if pd.notna(mu) else np.nan
        rows.append(row)

    foot: dict[str, float | int] = {"readability_level": "All levels (pooled by bin)"}
    for bi in bins_desc:
        bl = bin_labels[bi]
        foot[f"bin: {bl}"] = int(pool_tarab[bi])
    foot["count_tarab"] = tarab_total_split
    foot["count_barec"] = barec_total_split
    foot["old_distribution_%"] = 100.0 if barec_ref_total else 0.0
    foot["new_distribution_%"] = 100.0 if n_rows else 0.0
    foot["avg_readability_confidence"] = global_mean_conf
    rows.append(foot)

    wide = pd.DataFrame(rows).set_index("readability_level")
    wide = wide.reindex(columns=col_order)
    wide.index = wide.index.astype(str)
    wide.index.name = "readability_level"
    return wide
