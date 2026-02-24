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
    wanted = ["readability_level", "SingerNationality", "SongDialect", "songID"]
    usecols = [c for c in wanted if c in cols]
    if "readability_level" not in usecols:
        raise ValueError(
            f"`{rel_csv_path}` must contain `readability_level`. Found columns: {cols}"
        )
    df = pd.read_csv(path, compression="infer", encoding="utf-8", usecols=usecols)
    df["readability_level"] = pd.to_numeric(df["readability_level"], errors="coerce")
    df = df.dropna(subset=["readability_level"]).copy()
    df["readability_level"] = df["readability_level"].astype(int)
    df = df[(df["readability_level"] >= 1) & (df["readability_level"] <= 19)].copy()
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
    sns.countplot(
        data=df,
        x="readability_level",
        order=LEVELS,
        color=sns.color_palette("deep")[0],
        ax=ax,
    )
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
    sns.histplot(
        data=df,
        x="song_readability",
        bins=30,
        color=sns.color_palette("deep")[0],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Song readability (mean verse level)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def plot_02_dialect_boxplot(df, *, title: str, value_col: str = "readability_level") -> None:
    if "SongDialect" not in df.columns:
        st.warning("Missing column `SongDialect`; cannot render plot 02.")
        return

    import matplotlib.pyplot as plt
    import seaborn as sns

    dialect_order = df.groupby("SongDialect")[value_col].median().sort_values().index.tolist()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * len(dialect_order))))
    sns.boxplot(
        data=df,
        y="SongDialect",
        x=value_col,
        order=dialect_order,
        showfliers=False,
        color=sns.color_palette("deep")[1],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Readability")
    ax.set_ylabel("Dialect")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

def plot_02_category_boxplot(
    df,
    *,
    title: str,
    category_col: str,
    category_label: str,
    value_col: str,
) -> None:
    if category_col not in df.columns:
        st.warning(f"Missing column `{category_col}`; cannot render plot 02.")
        return

    import matplotlib.pyplot as plt
    import seaborn as sns

    order = df.groupby(category_col)[value_col].median().sort_values().index.tolist()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * len(order))))
    sns.boxplot(
        data=df,
        y=category_col,
        x=value_col,
        order=order,
        showfliers=False,
        color=sns.color_palette("deep")[1],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Readability")
    ax.set_ylabel(category_label)
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
    if "Variety" not in df.columns:
        st.warning("Missing column `Variety`; cannot render plot 02.")
        return

    import matplotlib.pyplot as plt
    import seaborn as sns

    order = df.groupby("Variety")[value_col].median().sort_values().index.tolist()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * len(order))))
    sns.boxplot(
        data=df,
        y="Variety",
        x=value_col,
        order=order,
        showfliers=False,
        color=sns.color_palette("deep")[1],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Readability level (1–19)")
    ax.set_ylabel("Variety")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def to_song_level(df, *, method: str) -> "pd.DataFrame":
    import pandas as pd

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

    out = out.dropna(subset=["song_readability"]).copy()
    out["song_level_discrete"] = out["song_readability"].round().clip(1, 19).astype(int)
    return out


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


def render_verse_level_section(*, title: str, csv_filename: str) -> None:
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
    plot_02_dialect_boxplot(df, title="Readability by dialect (boxplot)", value_col="readability_level")

    st.markdown("**3 — Country boxplot (top N + Other)**")
    top_n = st.slider(
        "Top N countries",
        min_value=5,
        max_value=30,
        value=15,
        step=1,
        key=f"{csv_filename}__verse__topn",
    )
    plot_05_country_boxplot_topN(
        df,
        title=f"Readability by country (top {top_n} + Other)",
        country_col="SingerNationality",
        value_col="readability_level",
        top_n=top_n,
        x_label="Country",
    )


def render_song_level_section(*, title: str, csv_filename: str) -> None:
    st.subheader(title)
    mean_tab, max_tab = st.tabs(["mean", "max"])

    with mean_tab:
        df = load_habibi_readability_csv(csv_filename)
        song_df = to_song_level(df, method="mean")

        st.markdown("**1 — Overall distribution (mean)**")
        filter_col, filter_label = habibi_groupby_selector(key_prefix=f"{csv_filename}__song_mean")
        filtered = _country_filter_ui(
            song_df,
            country_col=filter_col,
            label=f"{filter_label} filter",
            key=f"{csv_filename}__song_mean__filter",
        )
        plot_01_song_mean_hist(filtered, title="Song readability distribution (mean)")
        st.caption(f"Songs shown: {len(filtered):,}")

        st.markdown("**2 — Dialect boxplot (mean)**")
        plot_02_dialect_boxplot(song_df, title="Song readability by dialect (boxplot)", value_col="song_readability")

        st.markdown("**3 — Country boxplot (mean, top N + Other)**")
        top_n = st.slider(
            "Top N countries (mean)",
            min_value=5,
            max_value=30,
            value=15,
            step=1,
            key=f"{csv_filename}__mean__topn__country",
        )
        plot_05_country_boxplot_topN(
            song_df,
            title=f"Song readability by country (mean; top {top_n} + Other)",
            country_col="SingerNationality",
            value_col="song_readability",
            top_n=top_n,
            x_label="Country",
        )

    with max_tab:
        df = load_habibi_readability_csv(csv_filename)
        song_df = to_song_level(df, method="max")

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
        plot_02_dialect_boxplot(song_df, title="Song readability by dialect (boxplot)", value_col="song_readability")

        st.markdown("**3 — Country boxplot (max, top N + Other)**")
        top_n = st.slider(
            "Top N countries (max)",
            min_value=5,
            max_value=30,
            value=15,
            step=1,
            key=f"{csv_filename}__max__topn__country",
        )
        plot_05_country_boxplot_topN(
            song_df,
            title=f"Song readability by country (max; top {top_n} + Other)",
            country_col="SingerNationality",
            value_col="song_readability",
            top_n=top_n,
            x_label="Country",
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
    plot_02_variety_boxplot(df, title="Readability by variety (boxplot)", value_col="readability_level")

    st.markdown("**3 — Variety boxplot (top N + Other)**")
    top_n = st.slider(
        "Top N varieties",
        min_value=5,
        max_value=30,
        value=15,
        step=1,
        key=f"{key_prefix}__topn",
    )
    plot_05_country_boxplot_topN(
        df,
        title=f"Readability by variety (top {top_n} + Other)",
        country_col="Variety",
        value_col="readability_level",
        top_n=top_n,
        x_label="Variety",
    )


def main() -> None:
    st.set_page_config(page_title="Habibi Readability Plots", layout="wide")

    st.title("Habibi + Jawaher Readability Results")


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
        )

    with song_d3tok:
        render_song_level_section(
            title="Song-level readability (D3TOK text)",
            csv_filename="habibi_d3tok_readability.csv",
        )

    with verse_raw:
        render_verse_level_section(
            title="Verse-level readability (raw lyrics)",
            csv_filename="habibi_raw_readability.csv",
        )

    with song_raw:
        render_song_level_section(
            title="Song-level readability (raw lyrics)",
            csv_filename="habibi_raw_readability.csv",
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

