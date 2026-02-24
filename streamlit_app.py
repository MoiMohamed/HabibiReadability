from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

# Silence editor-only missing-import warnings in some setups.
# pyright: reportMissingImports=false


ROOT = Path(__file__).resolve().parent
LEVELS = list(range(1, 20))


DATA_DIR = ROOT / "data"

@st.cache_data(show_spinner=False)
def load_singer_decades_csv(rel_csv_path: str = "singer_decades_final.csv"):
    import numpy as np
    import pandas as pd

    path = DATA_DIR / rel_csv_path
    if not path.exists() and not rel_csv_path.endswith(".gz"):
        # Allow storing as gzip too.
        gz = DATA_DIR / f"{rel_csv_path}.gz"
        if gz.exists():
            path = gz

    if not path.exists():
        raise FileNotFoundError(f"Missing file: data/{rel_csv_path} (or .gz)")

    df = pd.read_csv(path, compression="infer", encoding="utf-8")
    required = {"Singer", "career_start", "peak_decade"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"singer_decades_final.csv missing columns: {sorted(missing)}")

    out = df.copy()
    out["Singer"] = out["Singer"].astype(str).str.strip()
    out["career_start"] = pd.to_numeric(out["career_start"], errors="coerce")
    out["peak_decade"] = pd.to_numeric(out["peak_decade"], errors="coerce")

    # Normalize to decade-like integers (e.g., 2010, 2020)
    out["career_start_decade"] = (
        (np.floor(out["career_start"] / 10) * 10).round().astype("Int64")
    )
    out["peak_decade"] = out["peak_decade"].round().astype("Int64")
    return out

@st.cache_data(show_spinner=False)
def load_jawaher_readability_jsonl(rel_jsonl_path: str):
    """
    Load Jawaher JSONL and normalize to columns:
      - Variety
      - readability_level  (int 1..19)
    Supports either:
      - readability_level
      - readability_level_d3tok
    """
    import pandas as pd

    path = DATA_DIR / rel_jsonl_path
    if not path.exists():
        raise FileNotFoundError(f"Missing file: data/{rel_jsonl_path}")

    df = pd.read_json(path, lines=True, encoding="utf-8")
    if "Variety" not in df.columns:
        raise ValueError(f"`{rel_jsonl_path}` must contain `Variety`.")

    if "readability_level" in df.columns:
        level_col = "readability_level"
    elif "readability_level_d3tok" in df.columns:
        level_col = "readability_level_d3tok"
    else:
        raise ValueError(
            f"`{rel_jsonl_path}` must contain `readability_level` or `readability_level_d3tok`."
        )

    out = df[["Variety", level_col]].copy()
    out.rename(columns={level_col: "readability_level"}, inplace=True)
    out["Variety"] = out["Variety"].astype(str)
    out["readability_level"] = pd.to_numeric(out["readability_level"], errors="coerce")
    out = out.dropna(subset=["readability_level", "Variety"]).copy()
    out["readability_level"] = out["readability_level"].astype(int)
    out = out[(out["readability_level"] >= 1) & (out["readability_level"] <= 19)].copy()
    return out


@st.cache_data(show_spinner=False)
def load_habibi_readability_csv(rel_csv_path: str):
    import pandas as pd

    # Allow using compressed CSVs (e.g., for GitHub size limits).
    # If caller passes "foo.csv", we also try "foo.csv.gz".
    candidates: list[Path] = []
    p0 = DATA_DIR / rel_csv_path
    candidates.append(p0)
    if not rel_csv_path.endswith(".gz"):
        # Common case: keep code referring to *.csv, but store *.csv.gz on disk.
        if rel_csv_path.endswith(".csv"):
            candidates.append(DATA_DIR / f"{rel_csv_path}.gz")  # foo.csv.gz
        else:
            candidates.append(DATA_DIR / f"{rel_csv_path}.gz")

    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        attempted = "\n".join([f"- data/{p.name}" for p in candidates])
        raise FileNotFoundError(f"Missing file. Tried:\n{attempted}")

    cols = list(pd.read_csv(path, nrows=0, compression="infer", encoding="utf-8").columns)
    wanted = ["readability_level", "Singer", "SingerNationality", "SongDialect", "songID"]
    usecols = [c for c in wanted if c in cols]
    if "readability_level" not in usecols:
        raise ValueError(
            f"`{rel_csv_path}` must contain `readability_level`. Found columns: {cols}"
        )
    if "Singer" not in usecols:
        raise ValueError(
            f"`{rel_csv_path}` must contain `Singer` (needed for decade merge). Found columns: {cols}"
        )
    df = pd.read_csv(path, compression="infer", encoding="utf-8", usecols=usecols)
    df["readability_level"] = pd.to_numeric(df["readability_level"], errors="coerce")
    df = df.dropna(subset=["readability_level"]).copy()
    df["readability_level"] = df["readability_level"].astype(int)
    df = df[(df["readability_level"] >= 1) & (df["readability_level"] <= 19)].copy()
    df["Singer"] = df["Singer"].astype(str).str.strip()
    if "SingerNationality" in df.columns:
        df["SingerNationality"] = df["SingerNationality"].astype(str)
    if "SongDialect" in df.columns:
        df["SongDialect"] = df["SongDialect"].astype(str)
    return df


def _country_filter_ui(df, *, country_col: str, label: str, key: str):
    if country_col not in df.columns:
        st.caption(f"Note: `{country_col}` not found; showing all rows.")
        return df
    countries = df[country_col].value_counts().index.tolist()
    selected = st.selectbox(label, ["All"] + countries, index=0, key=key)
    if selected == "All":
        return df
    return df[df[country_col] == selected]


def plot_01_distribution_levels(df, *, title: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 4))
    # Use pre-aggregated counts (faster than seaborn countplot on large frames)
    counts = df["readability_level"].value_counts().reindex(LEVELS, fill_value=0).sort_index()
    ax.bar([str(i) for i in LEVELS], counts.values, color=sns.color_palette("deep")[0])
    ax.set_title(title)
    ax.set_xlabel("Readability level (1–19)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def plot_01_song_mean_hist(df, *, title: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 4))
    # Keep for potential float-based plots; prefer discrete distributions elsewhere.
    sns.histplot(data=df, x="song_readability", bins=30, color=sns.color_palette("deep")[0], ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Song readability (mean verse level)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


@st.cache_data(show_spinner=False)
def get_song_level_from_csv(csv_filename: str, method: str):
    """
    Cached song-level aggregation (groupby songID).
    This is one of the most expensive operations, so caching it makes tab switching much faster.
    """
    df = load_habibi_readability_csv(csv_filename)
    return to_song_level(df, method=method)


def plot_02_dialect_boxplot(df, *, title: str, value_col: str = "readability_level") -> None:
    if "SongDialect" not in df.columns:
        st.warning("Missing column `SongDialect`; cannot render plot 02.")
        return

    plot_02_category_boxplot(
        df,
        title=title,
        category_col="SongDialect",
        category_label="Dialect",
        value_col=value_col,
        cache_key=None,
    )

def plot_02_category_boxplot(
    df,
    *,
    title: str,
    category_col: str,
    category_label: str,
    value_col: str,
    cache_key: str | None,
) -> None:
    if category_col not in df.columns:
        st.warning(f"Missing column `{category_col}`; cannot render plot 02.")
        return

    import matplotlib.pyplot as plt
    import seaborn as sns

    import pandas as pd

    sns.set_theme(style="whitegrid")
    dff = df[[category_col, value_col]].dropna().copy()
    if dff.empty:
        st.warning("No data available for this plot.")
        return

    # Compute Tukey boxplot stats per group (much faster to render than seaborn boxplot on huge frames).
    # Cache the computed stats in session_state to avoid recomputation on every rerun.
    ss_key = f"__boxstats__{cache_key}" if cache_key else None
    if ss_key and ss_key in st.session_state:
        stats = st.session_state[ss_key]
    else:
        grouped = dff.groupby(category_col)[value_col]
        med = grouped.median()
        q1 = grouped.quantile(0.25)
        q3 = grouped.quantile(0.75)
        vmin = grouped.min()
        vmax = grouped.max()
        iqr = (q3 - q1).fillna(0)
        whislo = (q1 - 1.5 * iqr).combine(vmin, max)
        whishi = (q3 + 1.5 * iqr).combine(vmax, min)

        order = med.sort_values().index.tolist()
        stats = [
            {
                "label": str(g),
                "med": float(med[g]),
                "q1": float(q1[g]),
                "q3": float(q3[g]),
                "whislo": float(whislo[g]),
                "whishi": float(whishi[g]),
                "fliers": [],
            }
            for g in order
            if pd.notna(med[g])
        ]
        if ss_key:
            st.session_state[ss_key] = stats

    n_groups = len(stats)
    # ≤ 10 groups: vertical; > 10 groups: horizontal
    vert = n_groups <= 10
    if vert:
        fig, ax = plt.subplots(figsize=(11, 5))
        b = ax.bxp(stats, showfliers=False, vert=True, patch_artist=True)
        ax.set_xticklabels([s["label"] for s in stats], rotation=45, ha="right")
        ax.set_xlabel(category_label)
        ax.set_ylabel("Readability")
    else:
        fig_h = max(4, 0.5 * len(stats))
        fig, ax = plt.subplots(figsize=(12, fig_h))
        b = ax.bxp(stats, showfliers=False, vert=False, patch_artist=True)
        ax.set_yticklabels([s["label"] for s in stats])
        ax.set_xlabel("Readability")
        ax.set_ylabel(category_label)

    # Style boxes to roughly match seaborn palette
    box_face = sns.color_palette("deep")[1]
    for box in b.get("boxes", []):
        try:
            box.set_facecolor(box_face)
            box.set_alpha(0.9)
        except Exception:
            pass
    ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def plot_05_country_boxplot_topN(
    df,
    *,
    title: str,
    country_col: str = "SingerNationality",
    value_col: str = "readability_level",
    top_n: int = 15,
    x_label: str = "Country",
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    if country_col not in df.columns:
        st.warning(f"Missing column `{country_col}`; cannot render plot 05.")
        return

    top = df[country_col].value_counts().head(int(top_n)).index
    dff = df.copy()
    dff["country_top"] = np.where(dff[country_col].isin(top), dff[country_col], "Other")
    order = dff.groupby("country_top")[value_col].median().sort_values().index.tolist()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=dff,
        x="country_top",
        y=value_col,
        order=order,
        showfliers=False,
        color=sns.color_palette("deep")[3],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Readability")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def habibi_groupby_selector(*, key_prefix: str):
    """
    Choose whether Habibi *plot 1 filter* uses dialect or country.
    Returns: (filter_col, filter_label)
    """
    choice = st.radio(
        "Filter by",
        ["Dialect", "Country"],
        horizontal=True,
        key=f"{key_prefix}__groupby_choice",
    )
    if choice == "Dialect":
        return "SongDialect", "Dialect"
    return "SingerNationality", "Country"


def plot_02_variety_boxplot(df, *, title: str, value_col: str = "readability_level") -> None:
    plot_02_category_boxplot(
        df,
        title=title,
        category_col="Variety",
        category_label="Variety",
        value_col=value_col,
        cache_key=None,
    )


def to_song_level(df, *, method: str) -> "pd.DataFrame":
    import pandas as pd
    import numpy as np

    if "songID" not in df.columns:
        raise ValueError("Missing column `songID` required for song-level plots.")
    if method not in {"mean", "max"}:
        raise ValueError("method must be one of: mean, max")

    def first_nonnull(s: "pd.Series"):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else None

    gb = df.groupby("songID", sort=False)
    out = pd.DataFrame({"songID": gb.size().index, "verses_n": gb.size().values})
    if method == "mean":
        out["song_readability"] = gb["readability_level"].mean().values
    else:
        out["song_readability"] = gb["readability_level"].max().values

    if "SongDialect" in df.columns:
        out["SongDialect"] = gb["SongDialect"].apply(first_nonnull).values
    if "SingerNationality" in df.columns:
        out["SingerNationality"] = gb["SingerNationality"].apply(first_nonnull).values
    if "Singer" in df.columns:
        out["Singer"] = gb["Singer"].apply(first_nonnull).values

    out = out.dropna(subset=["song_readability"]).copy()
    # Discrete view for level-based plots.
    # For mean: match scoring rule used elsewhere: level = clip(floor(score + 0.5), 1, 19).
    out["song_level_discrete"] = np.clip(np.floor(out["song_readability"] + 0.5), 1, 19).astype(int)
    return out


def attach_singer_decade(
    habibi_df,
    singer_decades_df,
    *,
    decade_field: str,
    fallback_field: str | None = None,
):
    """
    Left-join singer decade metadata onto Habibi rows via Singer.
    decade_field: 'peak_decade' or 'career_start_decade'
    fallback_field: optional secondary field to use when decade_field is missing
    """
    import pandas as pd

    if decade_field not in {"peak_decade", "career_start_decade"}:
        raise ValueError("decade_field must be one of: peak_decade, career_start_decade")
    if fallback_field is not None and fallback_field not in {"peak_decade", "career_start_decade"}:
        raise ValueError("fallback_field must be one of: peak_decade, career_start_decade")

    left = habibi_df.copy()
    cols = ["Singer", decade_field]
    if fallback_field and fallback_field != decade_field:
        cols.append(fallback_field)
    right = singer_decades_df[cols].copy()

    left["Singer"] = left["Singer"].astype(str).str.strip()
    right["Singer"] = right["Singer"].astype(str).str.strip()

    merged = left.merge(right, on="Singer", how="left")
    primary = pd.to_numeric(merged[decade_field], errors="coerce")
    if fallback_field:
        secondary = pd.to_numeric(merged[fallback_field], errors="coerce")
        decade = primary.where(primary.notna(), secondary)
    else:
        decade = primary
    merged["decade"] = decade.round().astype("Int64")
    return merged


def plot_decade_count(df, *, title: str, decade_col: str = "decade") -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if decade_col not in df.columns:
        st.warning(f"Missing column `{decade_col}`; cannot plot decade counts.")
        return
    dff = df.dropna(subset=[decade_col]).copy()
    if dff.empty:
        st.warning("No rows with decade values after merge/filter.")
        return

    sns.set_theme(style="whitegrid")
    counts = dff[decade_col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar([str(x) for x in counts.index.tolist()], counts.values, color=sns.color_palette("deep")[0])
    ax.set_title(title)
    ax.set_xlabel("Decade")
    ax.set_ylabel("Count")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def plot_decade_mean_bar(df, *, title: str, y_col: str, decade_col: str = "decade") -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    dff = df.dropna(subset=[decade_col, y_col]).copy()
    if dff.empty:
        st.warning("No rows with decade values after merge/filter.")
        return
    order = sorted(dff[decade_col].dropna().unique().tolist())
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.barplot(
        data=dff,
        x=decade_col,
        y=y_col,
        order=order,
        estimator=np.mean,
        errorbar=("ci", 95),
        color=sns.color_palette("deep")[2],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Decade")
    ax.set_ylabel("Mean")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def plot_decade_boxplot(
    df,
    *,
    title: str,
    y_col: str,
    y_label: str,
    decade_col: str = "decade",
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if decade_col not in df.columns or y_col not in df.columns:
        st.warning(f"Missing `{decade_col}` or `{y_col}`; cannot render this plot.")
        return

    dff = df.dropna(subset=[decade_col, y_col]).copy()
    if dff.empty:
        st.warning("No rows with decade values after merge/filter.")
        return
    order = sorted(dff[decade_col].dropna().unique().tolist())
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.boxplot(
        data=dff,
        x=decade_col,
        y=y_col,
        order=order,
        showfliers=False,
        color=sns.color_palette("deep")[1],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Decade")
    ax.set_ylabel(y_label)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def plot_decade_level_distribution(
    df,
    *,
    title: str,
    level_col: str,
    decade_col: str = "decade",
    key: str,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if decade_col not in df.columns or level_col not in df.columns:
        st.warning(f"Missing `{decade_col}` or `{level_col}`; cannot plot distribution by decade.")
        return

    dff = df.dropna(subset=[decade_col, level_col]).copy()
    if dff.empty:
        st.warning("No rows with decade values after merge/filter.")
        return

    decades = sorted(dff[decade_col].dropna().unique().tolist())
    selected = st.selectbox("Choose a decade", decades, key=key)
    sub = dff[dff[decade_col] == selected].copy()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 4))
    counts = sub[level_col].value_counts().reindex(LEVELS, fill_value=0).sort_index()
    ax.bar([str(i) for i in LEVELS], counts.values, color=sns.color_palette("deep")[0])
    ax.set_title(f"{title} — {selected}s")
    ax.set_xlabel("Readability level (1–19)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.caption(f"Rows in selected decade: {len(sub):,}")

def decade_source_selector(*, key_prefix: str):
    """
    Returns (decade_field, fallback_field, label_for_caption)
    """
    mode = st.radio(
        "Decade source (Singer)",
        ["Peak decade (fallback to career start)", "Career-start decade only"],
        horizontal=True,
        key=f"{key_prefix}__decade_source",
    )
    if mode.startswith("Peak"):
        return "peak_decade", "career_start_decade", "peak_decade → career_start_decade"
    return "career_start_decade", None, "career_start_decade"

def render_reproducibility_section() -> None:
    st.markdown(
        """
### How these results were obtained (reproducibility)

The figures in this app are **summaries of readability predictions** produced by the scripts
`preprocess_habibi_d3tok.py` and `predict_readability_camel.py`, then aggregated/visualized by
`analyze_habibi_readability.py` (verse-level) and `analyze_habibi_readability_song_level.py` (song-level).

- **Preprocessing for D3TOK (optional)**: `preprocess_habibi_d3tok.py` adds a `Lyrics_d3tok` column by running
  CAMeL Tools morphological analysis using:
  - Morphology DB: **CALIMA MSA s31** (`calima-msa-s31.db`)
  - Disambiguation model: `camel_tools.disambig.bert.BERTUnfactoredDisambiguator.pretrained(model_name="msa", top=1)`
  - Output token view: uses the analysis field `d3tok` (falls back to `lex`), with Arabic dediacritization via
    `camel_tools.utils.dediac.dediac_ar`.

- **Readability model**: `predict_readability_camel.py` runs the pretrained HuggingFace model
  **`CAMeL-Lab/readability-arabertv2-d3tok-reg`** using `transformers.pipeline("text-classification", ...)`.
  Inference is run with **truncation enabled** (`max_length=512`, `padding=True`, `function_to_apply="none"`).

- **Score: level mapping (1–19)**: the model output `score` is converted to an integer level via:
  `level = clip(round(score + 0.5), 1, 19)`.

        """
    )

    with st.expander("Exact commands (as used by the provided scripts)"):
        st.code(
            """\
# 1) (Optional) Create D3TOK version of lyrics (adds Lyrics_d3tok column)
python preprocess_habibi_d3tok.py \\
  --input habibi.csv \\
  --output habibi_d3tok.csv \\
  --db calima-msa-s31.db

# 2a) Predict readability on D3TOK text (recommended for this model)
python predict_readability_camel.py \\
  --input habibi_d3tok.csv \\
  --output habibi_d3tok_readability.csv \\
  --text_col Lyrics_d3tok

# 2b) Predict readability on raw lyrics (no D3TOK preprocessing)
python predict_readability_camel.py \\
  --input habibi.csv \\
  --output habibi_raw_readability.csv \\
  --text_col Lyrics

# 3) Verse-level plots (01/02/05 are a subset of the outputs)
python analyze_habibi_readability.py \\
  --input habibi_d3tok_readability.csv \\
  --outdir habibi_d3tok_readability_plots

python analyze_habibi_readability.py \\
  --input habibi_raw_readability.csv \\
  --outdir habibi_raw_readability_plots

# 4) Song-level plots (creates mean/ and max/ subfolders)
python analyze_habibi_readability_song_level.py \\
  --input habibi_d3tok_readability.csv \\
  --outdir habibi_d3tok_song_readability_plots \\
  --methods mean max

python analyze_habibi_readability_song_level.py \\
  --input habibi_raw_readability.csv \\
  --outdir habibi_raw_song_readability_plots \\
  --methods mean max
""",
            language="bash",
        )


def render_verse_level_section(*, title: str, csv_filename: str, singer_decades_df) -> None:
    st.subheader(title)

    df = load_habibi_readability_csv(csv_filename)

    st.markdown("**1 — Overall distribution**")
    filter_col, filter_label = habibi_groupby_selector(key_prefix=f"{csv_filename}__verse")
    dff = _country_filter_ui(
        df,
        country_col=filter_col,
        label=f"{filter_label} filter",
        key=f"{csv_filename}__verse__filter",
    )
    plot_01_distribution_levels(dff, title="Habibi readability level distribution")
    st.caption(f"Rows shown: {len(dff):,}")

    st.markdown("**2 — Dialect boxplot**")
    plot_02_category_boxplot(
        df,
        title="Readability by dialect (boxplot)",
        category_col="SongDialect",
        category_label="Dialect",
        value_col="readability_level",
        cache_key=f"{csv_filename}__verse__dialect__readability_level",
    )

    st.markdown("**3 — Country boxplot**")
    plot_02_category_boxplot(
        df,
        title="Readability by country (boxplot)",
        category_col="SingerNationality",
        category_label="Country",
        value_col="readability_level",
        cache_key=f"{csv_filename}__verse__country__readability_level",
    )

    st.divider()
    st.markdown("### Decade analysis")
    if singer_decades_df is None:
        st.error("Missing `data/singer_decades_final.csv` (required for decade plots).")
        return

    decade_field, fallback_field, label = decade_source_selector(
        key_prefix=f"{csv_filename}__verse_decades"
    )
    small = df[["Singer", "readability_level"]].copy()
    merged = attach_singer_decade(
        small,
        singer_decades_df,
        decade_field=decade_field,
        fallback_field=fallback_field,
    )
    match_rate = (merged["decade"].notna().mean() * 100.0) if len(merged) else 0.0
    st.caption(f"Decade match rate: {match_rate:.1f}% (using {label}).")

    st.markdown("**Verses per decade**")
    plot_decade_count(merged, title="Verse count by decade")

    st.markdown("**Readability by decade (boxplot)**")
    plot_decade_boxplot(
        merged,
        title="Verse readability by decade",
        y_col="readability_level",
        y_label="Readability level (1–19)",
    )

    st.markdown("**Distribution within a selected decade**")
    plot_decade_level_distribution(
        merged,
        title="Verse readability distribution",
        level_col="readability_level",
        key=f"{csv_filename}__verse_decades__pick_decade",
    )


def render_song_level_section(*, title: str, csv_filename: str, singer_decades_df) -> None:
    st.subheader(title)
    mean_tab, max_tab = st.tabs(["mean", "max"])

    with mean_tab:
        song_df = get_song_level_from_csv(csv_filename, method="mean")

        st.markdown("**1 — Overall distribution (mean)**")
        filter_col, filter_label = habibi_groupby_selector(key_prefix=f"{csv_filename}__song_mean")
        filtered = _country_filter_ui(
            song_df,
            country_col=filter_col,
            label=f"{filter_label} filter",
            key=f"{csv_filename}__song_mean__filter",
        )
        # Use discrete levels for mean (rounded to readability levels).
        tmp = filtered.rename(columns={"song_level_discrete": "readability_level"})[["readability_level"]].copy()
        plot_01_distribution_levels(tmp, title="Song readability distribution (mean; rounded to level)")
        st.caption(f"Songs shown: {len(filtered):,}")

        st.markdown("**2 — Dialect boxplot (mean)**")
        plot_02_category_boxplot(
            song_df,
            title="Song readability by dialect (boxplot; mean rounded to level)",
            category_col="SongDialect",
            category_label="Dialect",
            value_col="song_level_discrete",
            cache_key=f"{csv_filename}__song_mean__dialect__song_level_discrete",
        )

        st.markdown("**3 — Country boxplot (mean)**")
        plot_02_category_boxplot(
            song_df,
            title="Song readability by country (mean; rounded to level)",
            category_col="SingerNationality",
            category_label="Country",
            value_col="song_level_discrete",
            cache_key=f"{csv_filename}__song_mean__country__song_level_discrete",
        )

        st.divider()
        st.markdown("### Decade analysis")
        if singer_decades_df is None:
            st.error("Missing `data/singer_decades_final.csv` (required for decade plots).")
        else:
            decade_field, fallback_field, label = decade_source_selector(
                key_prefix=f"{csv_filename}__song_mean_decades"
            )
            small = song_df[["Singer", "song_readability", "song_level_discrete", "verses_n"]].copy()
            merged = attach_singer_decade(
                small,
                singer_decades_df,
                decade_field=decade_field,
                fallback_field=fallback_field,
            )
            match_rate = (merged["decade"].notna().mean() * 100.0) if len(merged) else 0.0
            st.caption(f"Decade match rate: {match_rate:.1f}% (using {label}).")

            st.markdown("**Songs per decade**")
            plot_decade_count(merged, title="Song count by decade (mean)")

            st.markdown("**Song readability by decade (boxplot)**")
            plot_decade_boxplot(
                merged,
                title="Song readability by decade (mean)",
                y_col="song_level_discrete",
                y_label="Readability level (1–19)",
            )

            st.markdown("**Distribution within a selected decade**")
            plot_decade_level_distribution(
                merged.rename(columns={"song_level_discrete": "readability_level"}),
                title="Song readability distribution (mean; rounded to level)",
                level_col="readability_level",
                key=f"{csv_filename}__song_mean_decades__pick_decade",
            )

    with max_tab:
        song_df = get_song_level_from_csv(csv_filename, method="max")

        st.markdown("**1 — Overall distribution (max)**")
        filter_col, filter_label = habibi_groupby_selector(key_prefix=f"{csv_filename}__song_max")
        filtered = _country_filter_ui(
            song_df,
            country_col=filter_col,
            label=f"{filter_label} filter",
            key=f"{csv_filename}__song_max__filter",
        )
        # Use discrete level distribution for max (matches original script's intent)
        tmp = filtered.rename(columns={"song_level_discrete": "readability_level"})[["readability_level"]].copy()
        plot_01_distribution_levels(tmp, title="Song readability distribution (max verse level)")
        st.caption(f"Songs shown: {len(filtered):,}")

        st.markdown("**2 — Dialect boxplot (max)**")
        plot_02_category_boxplot(
            song_df,
            title="Song readability by dialect",
            category_col="SongDialect",
            category_label="Dialect",
            value_col="song_readability",
            cache_key=f"{csv_filename}__song_max__dialect__song_readability",
        )

        st.markdown("**3 — Country boxplot (max)**")
        plot_02_category_boxplot(
            song_df,
            title="Song readability by country (max)",
            category_col="SingerNationality",
            category_label="Country",
            value_col="song_readability",
            cache_key=f"{csv_filename}__song_max__country__song_readability",
        )

        st.divider()
        st.markdown("### Decade analysis")
        if singer_decades_df is None:
            st.error("Missing `data/singer_decades_final.csv` (required for decade plots).")
        else:
            decade_field, fallback_field, label = decade_source_selector(
                key_prefix=f"{csv_filename}__song_max_decades"
            )
            small = song_df[["Singer", "song_readability", "song_level_discrete", "verses_n"]].copy()
            merged = attach_singer_decade(
                small,
                singer_decades_df,
                decade_field=decade_field,
                fallback_field=fallback_field,
            )
            match_rate = (merged["decade"].notna().mean() * 100.0) if len(merged) else 0.0
            st.caption(f"Decade match rate: {match_rate:.1f}% (using {label}).")

            st.markdown("**Songs per decade**")
            plot_decade_count(merged, title="Song count by decade (max)")

            st.markdown("**Song readability by decade (boxplot)**")
            plot_decade_boxplot(
                merged,
                title="Song readability by decade (max)",
                y_col="song_readability",
                y_label="Song readability",
            )

            st.markdown("**Distribution within a selected decade**")
            plot_decade_level_distribution(
                merged.rename(columns={"song_level_discrete": "readability_level"}),
                title="Song readability distribution (max verse level)",
                level_col="readability_level",
                key=f"{csv_filename}__song_max_decades__pick_decade",
            )


def render_jawaher_section(*, title: str, jsonl_filename: str, key_prefix: str) -> None:
    st.subheader(title)

    try:
        df = load_jawaher_readability_jsonl(jsonl_filename)
    except Exception as e:
        st.warning(f"Could not load `data/{jsonl_filename}`.\n\n{e}")
        return

    st.markdown("**1 — Overall distribution**")
    dff = _country_filter_ui(
        df,
        country_col="Variety",
        label="Variety",
        key=f"{key_prefix}__variety_filter",
    )
    plot_01_distribution_levels(dff, title="Jawaher readability level distribution")
    st.caption(f"Rows shown: {len(dff):,}")

    st.markdown("**2 — Variety boxplot**")
    plot_02_category_boxplot(
        df,
        title="Readability by variety (boxplot)",
        category_col="Variety",
        category_label="Variety",
        value_col="readability_level",
        cache_key=f"{key_prefix}__variety__readability_level",
    )


def main() -> None:
    st.set_page_config(page_title="Habibi Readability Plots", layout="wide")

    st.title("Habibi + Jawaher Readability Results")

    singer_decades_df = None
    try:
        singer_decades_df = load_singer_decades_csv("singer_decades_final.csv")
    except Exception as e:
        st.warning(f"Could not load singer decades from `data/`: {e}")


    render_reproducibility_section()

    st.divider()
    if not DATA_DIR.exists():
        st.warning(
            "Missing `data/` folder next to `streamlit_app.py`.\n\n"
            "Create it and add:\n"
            "- `data/habibi_d3tok_readability.csv`\n"
            "- `data/habibi_raw_readability.csv`\n"
            "- `data/jawaher_dediac_readability.jsonl`\n"
            "- `data/jawaher_dediac_d3tok_readability.jsonl`"
        )

    verse_d3tok, song_d3tok, verse_raw, song_raw, jaw_dediac, jaw_d3tok = st.tabs(
        [
            "Verse-level (D3TOK)",
            "Song-level (D3TOK)",
            "Verse-level (Raw)",
            "Song-level (Raw)",
            "Jawaher (dediac)",
            "Jawaher (d3tok)",
        ]
    )

    with verse_d3tok:
        render_verse_level_section(
            title="Verse-level readability (D3TOK text)",
            csv_filename="habibi_d3tok_readability.csv",
            singer_decades_df=singer_decades_df,
        )

    with song_d3tok:
        render_song_level_section(
            title="Song-level readability (D3TOK text)",
            csv_filename="habibi_d3tok_readability.csv",
            singer_decades_df=singer_decades_df,
        )

    with verse_raw:
        render_verse_level_section(
            title="Verse-level readability (raw lyrics)",
            csv_filename="habibi_raw_readability.csv",
            singer_decades_df=singer_decades_df,
        )

    with song_raw:
        render_song_level_section(
            title="Song-level readability (raw lyrics)",
            csv_filename="habibi_raw_readability.csv",
            singer_decades_df=singer_decades_df,
        )

    with jaw_dediac:
        render_jawaher_section(
            title="Jawaher readability (dediac)",
            jsonl_filename="jawaher_dediac_readability.jsonl",
            key_prefix="jawaher_dediac",
        )

    with jaw_d3tok:
        render_jawaher_section(
            title="Jawaher readability (d3tok)",
            jsonl_filename="jawaher_dediac_d3tok_readability.jsonl",
            key_prefix="jawaher_d3tok",
        )


if __name__ == "__main__":
    main()

