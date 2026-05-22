from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import streamlit as st

from tarab_model_experimentation.constants import SPLITS_DIR

_BAREC_TRAIN_ROWS = 54_845
_LEVELS = 19


@st.cache_data(show_spinner=False)
def _load_split_size_stats() -> list[dict[str, Any]]:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    if not SPLITS_DIR.exists():
        return rows

    for path in sorted(SPLITS_DIR.glob("barec_tarab*.csv")):
        if path.name in {"barec_tarab_2X_55k_match_distribution_155k_wo_class19.csv"}:
            continue
        df = pd.read_csv(path, usecols=["source"], encoding="utf-8")
        src = df["source"].astype(str).str.strip().str.lower()
        rows.append(
            {
                "file": path.name,
                "total": int(len(df)),
                "barec": int((src == "barec").sum()),
                "tarab": int((src == "tarab").sum()),
            }
        )
    return rows


def _match_distribution_table(stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"barec_tarab_(?P<mult>0\.5X|1X|2X|4X)_55k_match_distribution_(?P<nominal>\d+)k",
        re.I,
    )
    mult_label = {"0.5X": "0.5× BAREC", "1X": "1× BAREC", "2X": "2× BAREC", "4X": "4× BAREC"}
    out: list[dict[str, Any]] = []
    for row in stats:
        m = pattern.search(row["file"])
        if not m:
            continue
        mult = m.group("mult")
        nominal_k = int(m.group("nominal"))
        tarab_target = int(_BAREC_TRAIN_ROWS * {"0.5X": 0.5, "1X": 1.0, "2X": 2.0, "4X": 4.0}[mult])
        out.append(
            {
                "Experiment": mult_label.get(mult, mult),
                "Tarab target (design)": f"{tarab_target:,}",
                "Nominal total (design)": f"{_BAREC_TRAIN_ROWS + tarab_target:,}",
                "Actual total": f"{row['total']:,}",
                "BAREC": f"{row['barec']:,}",
                "Tarab pseudo": f"{row['tarab']:,}",
            }
        )
    order = {"0.5× BAREC": 0, "1× BAREC": 1, "2× BAREC": 2, "4× BAREC": 3}
    out.sort(key=lambda r: order.get(r["Experiment"], 99))
    return out


def _uniform_distribution_table(stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pattern = re.compile(r"uniform_(?P<per_class>\d+)k_per_class_(?P<nominal>\d+)k", re.I)
    out: list[dict[str, Any]] = []
    for row in stats:
        m = pattern.search(row["file"])
        if not m:
            continue
        per_k_k = int(m.group("per_class"))  # 3 in "3k_per_class"
        per_class_rows = per_k_k * 1000
        nominal_total = per_class_rows * _LEVELS
        out.append(
            {
                "Experiment": f"{per_k_k}k / class",
                "Per-class target": f"{per_class_rows:,}",
                "Nominal total (19 × per class)": f"{nominal_total:,}",
                "Actual total": f"{row['total']:,}",
                "BAREC": f"{row['barec']:,}",
                "Tarab pseudo": f"{row['tarab']:,}",
            }
        )
    out.sort(key=lambda r: int(r["Experiment"].split("k")[0]))
    return out


def render_experiment_design_overview() -> None:
    """Design narrative + actual split sizes from ``data/splits``."""
    import pandas as pd

    stats = _load_split_size_stats()

    st.markdown(
        """
The **control** run is baseline finetuning:
**AraBERT v02**, cross-entropy, **word** variant, on BAREC train only.

We then run **eight distillation experiments**: same model setup, but each training set is
**BAREC train + Tarab pseudo-labels** sampled differently.
"""
    )

    st.markdown("#### Controls (fixed across experiments)")
    st.markdown(
        """
- Model, finetuning hyperparameters, and BAREC dev / test
- **Tarab confidence fill** (whenever we add Tarab at class *k*): sample **randomly within bands**
  **95–100**, then **90–95**, **85–90**, … until the class quota is met or Tarab runs out.

"""
    )

    st.markdown("#### What we change between runs")
    st.markdown(
        """
1. **Dataset size** — how much Tarab we add (see two experiment sets below).
2. **Label distribution** — match BAREC train vs uniform per readability level.
"""
    )

    st.markdown("#### Properties of each training set")
    st.markdown(
        """
These follow from size and distribution; we chart them to describe each mix:

- Confidence mix per class
- Vocabulary overlap with BAREC (shared word types)
- Pseudo-label **type** (song vs poem)
- **ALDi / AGS** profile vs BAREC at the same level
- **Text length** (characters and words per sentence)
"""
    )

    st.markdown("#### Experiment set A — Match BAREC training distribution")
    st.markdown(
        """
**Filling procedure (match):** at each class *k*:

1. Take **every** BAREC train row at *k*.
2. Set a Tarab quota at *k* so BAREC+Tarab keeps BAREC’s label histogram, scaled by 0.5× / 1× / 2× / 4×.
3. Top up with Tarab pseudo at *k* using the confidence bands until the quota is met or Tarab runs out.
"""
    )
    st.caption(
        "All ~54.8k BAREC train rows kept across the run. Tarab volume scales with BAREC size "
        "(0.5×, 1×, 2×, 4×)."
    )
    match_rows = _match_distribution_table(stats)
    if match_rows:
        st.dataframe(pd.DataFrame(match_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No match-distribution split CSVs found in `data/splits`.")

    st.markdown("#### Experiment set B — Uniform readability distribution")
    st.markdown(
        """
**Filling procedure (uniform):** at each class *k*, target **T** rows (3k / 4k / 5k / 6k):

1. **Randomly sample** BAREC train rows at *k* — up to **T** rows. If BAREC has more than **T** at *k*
   (e.g. ~11k at level 12 when **T** = 3k), subsample BAREC to **T** and stop.
2. If count < **T**, top up with Tarab pseudo at *k* using the confidence bands until **T** or Tarab runs out.
3. If Tarab has no rows at *k*, keep whatever BAREC sample we have.
"""
    )
    st.caption(
        "Nominal total = **3k/4k/5k/6k per class × 19** (e.g. 3k → 57,000); **actual** totals are "
        "lower when Tarab or BAREC cannot fill a class."
    )
    uniform_rows = _uniform_distribution_table(stats)
    if uniform_rows:
        st.dataframe(pd.DataFrame(uniform_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No uniform split CSVs found in `data/splits`.")
