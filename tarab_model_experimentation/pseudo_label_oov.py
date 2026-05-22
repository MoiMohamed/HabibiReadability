from __future__ import annotations

import json
import re
from typing import Any

import streamlit as st

from tarab_model_experimentation.constants import (
    BAREC_PARQUET_DIR,
    DATA_DIR,
    SPLITS_DIR,
    TARAB_FULL_CORPUS_EMBED,
)
from tarab_model_experimentation.selection import (
    experiment_chart_label,
    experiment_chart_sort_key,
    resolve_split_csv_for_chart_label,
)

_TOKEN_RE = re.compile(r"[^\s]+", re.UNICODE)

TARAB_FULL_CSV = DATA_DIR / "tarab_full.sentence_aldi_ags_readability.csv"

def tokenize_text(text: str) -> list[str]:
    """Whitespace/punctuation tokens (same rule for gold and pseudo)."""
    return _TOKEN_RE.findall(str(text).strip())


def vocab_from_texts(texts) -> set[str]:
    vocab: set[str] = set()
    for text in texts:
        vocab.update(tokenize_text(text))
    return vocab


def vocab_overlap_stats(barec_vocab: set[str], tarab_vocab: set[str]) -> dict[str, Any]:
    """Tarab OOV = types in ``tarab_vocab`` that BAREC gold has never seen."""
    shared = barec_vocab & tarab_vocab
    barec_only = barec_vocab - tarab_vocab
    tarab_oov_set = tarab_vocab - barec_vocab
    barec_n = len(barec_vocab)
    tarab_n = len(tarab_vocab)
    shared_n = len(shared)
    tarab_oov_n = len(tarab_oov_set)
    barec_oov_n = len(barec_only)
    return {
        "barec_word_types": barec_n,
        "tarab_word_types": tarab_n,
        "shared_word_types": shared_n,
        "tarab_oov": tarab_oov_n,
        "tarab_not_in_barec": tarab_oov_n,
        "tarab_in_barec": shared_n,
        "tarab_oov_rate": tarab_oov_n / tarab_n if tarab_n else 0.0,
        "tarab_in_barec_rate": shared_n / tarab_n if tarab_n else 0.0,
        "barec_oov": barec_oov_n,
        "barec_not_in_tarab": barec_oov_n,
        "barec_in_tarab": shared_n,
        "barec_oov_rate": barec_oov_n / barec_n if barec_n else 0.0,
        "barec_in_tarab_rate": shared_n / barec_n if barec_n else 0.0,
        "barec_coverage_rate": shared_n / barec_n if barec_n else 0.0,
        "tarab_coverage_rate": shared_n / tarab_n if tarab_n else 0.0,
    }


@st.cache_data(show_spinner=True)
def load_full_barec_gold_vocab() -> tuple[set[str] | None, dict[str, Any] | None, list[str]]:
    """
    Word types from BAREC train + dev + test parquets.

    Each parquet row is one sentence; the ``Word`` column holds the full
    sentence text (same as ``Sentence``), not a single lexical token.
    """
    import pandas as pd

    texts: list[str] = []
    per_split_records: dict[str, int] = {}
    splits_loaded: list[str] = []
    for split in ("train", "dev", "test"):
        path = BAREC_PARQUET_DIR / f"{split}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["Word"])
        split_texts = df["Word"].astype(str).tolist()
        per_split_records[split] = len(split_texts)
        texts.extend(split_texts)
        splits_loaded.append(split)

    if not texts:
        return None, None, splits_loaded

    meta = {
        "total_records": len(texts),
        "unique_sentence_texts": len(set(texts)),
        "per_split_records": per_split_records,
    }
    return vocab_from_texts(texts), meta, splits_loaded


@st.cache_data(show_spinner=True)
def load_full_tarab_vocab() -> tuple[set[str] | None, int]:
    """Word types from full Tarab corpus CSV (``verse_lyrics``)."""
    import pandas as pd

    if not TARAB_FULL_CSV.exists():
        return None, 0

    vocab: set[str] = set()
    n_rows = 0
    for chunk in pd.read_csv(
        TARAB_FULL_CSV,
        usecols=["verse_lyrics"],
        chunksize=200_000,
        encoding="utf-8",
    ):
        chunk = chunk.dropna(subset=["verse_lyrics"])
        n_rows += len(chunk)
        for text in chunk["verse_lyrics"].astype(str):
            vocab.update(tokenize_text(text))
    return vocab, n_rows


@st.cache_data(show_spinner=False)
def load_embedded_full_tarab_corpus() -> tuple[dict[str, Any] | None, int]:
    """Precomputed full-corpus overlap (see ``scripts/build_tarab_corpus_embed.py``)."""
    if not TARAB_FULL_CORPUS_EMBED.exists():
        return None, 0
    payload = json.loads(TARAB_FULL_CORPUS_EMBED.read_text(encoding="utf-8"))
    stats = payload.get("corpus_stats")
    if not isinstance(stats, dict):
        return None, 0
    return stats, int(payload.get("tarab_sentences") or 0)


@st.cache_data(show_spinner=False)
def _load_split_by_source(rel_csv: str) -> tuple[list[str], list[str]] | None:
    import pandas as pd

    path = SPLITS_DIR / rel_csv
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=["Sentence", "source"], encoding="utf-8")
    df = df.dropna(subset=["Sentence"]).copy()
    df["source"] = df["source"].astype(str).str.strip().str.lower()
    barec_texts = df.loc[df["source"] == "barec", "Sentence"].astype(str).tolist()
    tarab_texts = df.loc[df["source"] == "tarab", "Sentence"].astype(str).tolist()
    return barec_texts, tarab_texts


def _training_split_csvs() -> list[tuple[str, str]]:
    if not SPLITS_DIR.exists():
        return []
    by_label: dict[str, list[str]] = {}
    for path in sorted(SPLITS_DIR.glob("*.csv")):
        label = experiment_chart_label(f"{path.stem}.out")
        by_label.setdefault(label, []).append(path.name)
    out: list[tuple[str, str]] = []
    for label in sorted(by_label, key=experiment_chart_sort_key):
        csv_name = resolve_split_csv_for_chart_label(label, by_label[label])
        out.append((label, csv_name))
    return out


@st.cache_data(show_spinner=True)
def build_per_training_split_vocab_table(*, _schema_version: int = 3):
    import pandas as pd

    barec_vocab, barec_meta, splits_loaded = load_full_barec_gold_vocab()
    if barec_vocab is None or barec_meta is None or not splits_loaded:
        return None

    rows: list[dict[str, Any]] = []
    for label, csv_name in _training_split_csvs():
        loaded = _load_split_by_source(csv_name)
        if loaded is None:
            continue
        barec_texts, tarab_texts = loaded
        if not tarab_texts:
            continue
        tarab_vocab = vocab_from_texts(tarab_texts)
        stats = vocab_overlap_stats(barec_vocab, tarab_vocab)
        rows.append(
            {
                "training_set": label,
                "split_csv": csv_name,
                "barec_gold_records": barec_meta["total_records"],
                "barec_train_rows": len(barec_texts),
                "tarab_rows": len(tarab_texts),
                **stats,
            }
        )
    if not rows:
        return None
    out = pd.DataFrame(rows)
    out["_sort"] = out["training_set"].map(experiment_chart_sort_key)
    return out.sort_values("_sort").drop(columns="_sort")


def _ensure_split_df_columns(df):
    """Backfill columns when Streamlit serves a stale cached frame."""
    import pandas as pd

    out = df.copy()
    if "tarab_oov" not in out.columns and "tarab_not_in_barec" in out.columns:
        out["tarab_oov"] = out["tarab_not_in_barec"]
    if "barec_oov" not in out.columns and "barec_not_in_tarab" in out.columns:
        out["barec_oov"] = out["barec_not_in_tarab"]
    if "barec_in_tarab" not in out.columns and "shared_word_types" in out.columns:
        out["barec_in_tarab"] = out["shared_word_types"]
    if "tarab_in_barec" not in out.columns and "shared_word_types" in out.columns:
        out["tarab_in_barec"] = out["shared_word_types"]
    for rate_col, num_col, den_col in (
        ("tarab_oov_rate", "tarab_oov", "tarab_word_types"),
        ("barec_oov_rate", "barec_oov", "barec_word_types"),
        ("tarab_in_barec_rate", "tarab_in_barec", "tarab_word_types"),
        ("barec_in_tarab_rate", "barec_in_tarab", "barec_word_types"),
    ):
        if rate_col not in out.columns and num_col in out.columns and den_col in out.columns:
            out[rate_col] = out[num_col] / out[den_col].replace(0, pd.NA)
    return out


def _barec_gold_breakdown_caption(meta: dict[str, Any], splits_loaded: list[str]) -> str:
    parts = [
        f"{meta['per_split_records'][s]:,} {s}"
        for s in splits_loaded
        if s in meta["per_split_records"]
    ]
    breakdown = " + ".join(parts)
    unique = meta["unique_sentence_texts"]
    total = meta["total_records"]
    unique_note = (
        f" · {unique:,} unique sentence texts"
        if unique != total
        else ""
    )
    return f"BAREC: {breakdown} = **{total:,} records**{unique_note}."


def render_pseudo_label_oov_section() -> None:
    st.markdown("### Tarab ↔ BAREC vocabulary overlap")
    st.caption(
        "Word types = whitespace/punctuation tokens. "
        "**BAREC covered by Tarab %** = share of BAREC gold types that also appear in the Tarab text."
    )

    barec_vocab, barec_meta, splits_loaded = load_full_barec_gold_vocab()

    if barec_vocab is None or barec_meta is None or not splits_loaded:
        st.warning(
            f"Could not load BAREC parquets from `{BAREC_PARQUET_DIR}` "
            "(expected train/dev/test.parquet)."
        )
        return

    corpus_stats, tarab_sentences = load_embedded_full_tarab_corpus()
    if corpus_stats is None:
        tarab_vocab, tarab_sentences = load_full_tarab_vocab()
        if tarab_vocab is None:
            st.warning(
                f"Could not load Tarab corpus (missing `{TARAB_FULL_CORPUS_EMBED}` "
                f"and `{TARAB_FULL_CSV}`)."
            )
            return
        corpus_stats = vocab_overlap_stats(barec_vocab, tarab_vocab)

    st.markdown("#### Full Tarab corpus × full BAREC")
    st.caption(
        f"{_barec_gold_breakdown_caption(barec_meta, splits_loaded)} "
        f"· Tarab sentences: **{tarab_sentences:,}**"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tarab types", f"{corpus_stats['tarab_word_types']:,}")
    c2.metric("BAREC types", f"{corpus_stats['barec_word_types']:,}")
    c3.metric("Shared types", f"{corpus_stats['shared_word_types']:,}")
    c4.metric(
        "BAREC covered by Tarab %",
        f"{corpus_stats['barec_in_tarab_rate']:.1%}",
        help="Share of BAREC gold word types (train+dev+test) that appear in the Tarab text.",
    )

    split_df = _ensure_split_df_columns(build_per_training_split_vocab_table())
    if split_df is None or split_df.empty:
        st.warning("No training splits found under `data/splits/*.csv`.")
        return

    st.markdown("#### Per training experiment (Tarab pseudo-label text)")
    import pandas as pd

    display = split_df.copy()
    display["barec_covered_by_tarab_%"] = (
        100
        * display["barec_in_tarab"]
        / display["barec_word_types"].replace(0, pd.NA)
    ).round(1)
    st.dataframe(
        display[
            [
                "training_set",
                "tarab_word_types",
                "barec_word_types",
                "shared_word_types",
                "barec_covered_by_tarab_%",
            ]
        ].rename(
            columns={
                "training_set": "Training set",
                "tarab_word_types": "Tarab types",
                "barec_word_types": "BAREC types",
                "shared_word_types": "Shared",
                "barec_covered_by_tarab_%": "BAREC covered by Tarab %",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    from tarab_model_experimentation.presentation_insights import (
        render_vocab_overlap_insight,
    )

    render_vocab_overlap_insight()
