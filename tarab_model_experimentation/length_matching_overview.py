from __future__ import annotations

import streamlit as st

from tarab_model_experimentation.constants import (
    LENGTH_MATCHED_BAREC_CHARS_BY_LEVEL_PNG,
    LENGTH_MATCHED_TARAB_AGGREGATED_CHARS_BY_LEVEL_PNG,
    LENGTH_MATCHED_TARAB_ORIGINAL_CHARS_BY_LEVEL_PNG,
)


def render_length_matching_tab(*, log_files: list[str]) -> None:
    """Methodology overview plus training/dev/test charts for the length-matched run."""
    from tarab_model_experimentation.class19_investigation import (
        render_length_matching_full_analysis_section,
    )

    render_length_matching_overview_section()
    render_length_matching_full_analysis_section(log_files=log_files)


def render_length_matching_overview_section() -> None:
    """Methodology narrative for the length-matching Tarab pseudo-label build."""
    st.markdown(
        """

Length matching builds a Tarab training pool by joining consecutive verses from the
same song/poem into longer pseudo-labeled sentences, so that, per readability level
*k*, the character-length distribution of added Tarab rows is aligned with BAREC’s
distribution at *k*.
"""
    )

    st.markdown("#### Step 1 — Build art-level n-grams and pseudo-labels")

    st.markdown(
        """
For each art (song or poem), verses are ordered by `verse_order`. Within that art we
form consecutive windows of size n = 1, 2, 3, 4, 5:

- n = 1: each verse is its own row (unigram).
- n ≥ 2: slide a window of *n* adjacent verses; lyrics are concatenated with a space. Metadata records which verse orders were used.

For n ≥ 2, the teacher model scores the combined text; the row keeps that
aggregate pseudo-label and confidence.

The new pool contains unigrams through 5-grams (~10M candidate rows in total, with
overlap across sizes because the same verse can appear in several windows).
"""
    )

    st.markdown("#### Step 2 — Per-level length statistics and valid n range")

    st.markdown(
        r"""
For each readability level $k \in \{1,\ldots,19\}$, we compare character length on
BAREC vs Tarab unigram verses:

| Statistic | BAREC | Tarab (unigrams) |
|-----------|--------|------------------|
| Lower band | p25 — 25th percentile of char length | — |
| Target | median | median verse length |
| Upper band | p75 — 75th percentile | — |

Tarab verses are short; BAREC sentences at the same $k$ are longer. If one Tarab verse
has median length $L^{\mathrm{tarab}}_k$ characters, then $n$ verses stitched together
have expected length about $n \cdot L^{\mathrm{tarab}}_k$. We choose how many verses
to glue so that range covers BAREC’s interquartile band at $k$:
"""
    )

    st.latex(
        r"""
n_{\min}(k) = \max\left(1,\ \left\lfloor \frac{L^{\mathrm{barec}}_{k,25}}{L^{\mathrm{tarab}}_k} \right\rfloor \right),
\qquad
n_{\max}(k) = \min\left(5,\ \left\lceil \frac{L^{\mathrm{barec}}_{k,75}}{L^{\mathrm{tarab}}_k} \right\rceil \right)
"""
    )

    st.markdown(
        r"""
Reasoning: $n_{\min}$ is the smallest $n$ whose typical length reaches at least BAREC’s
lower quartile; $n_{\max}$ is the smallest $n$ whose typical length reaches BAREC’s upper
quartile.

Only candidates in $[n_{\min}(k), n_{\max}(k)]$ are eligible when filling
class $k$.
"""
    )

    st.markdown("#### Step 3 — Weights over n within [n_min, n_max]")

    st.markdown(
        """
Among eligible sizes, we do not pick $n$ uniformly. For each $n$, let
$M^{\mathrm{tarab}}_{k,n}$ be the empirical median character length of Tarab
$n$-grams at level $k$. We score how close that length is to BAREC’s median at $k$:
"""
    )

    st.latex(r"\ell_n = M^{\mathrm{tarab}}_{k,n}")

    st.latex(
        r"""
d_n = \left| \ell_n - L^{\mathrm{barec}}_{k,50} \right|
"""
    )

    st.markdown("Weights follow a Gaussian (unnormalized, then normalized to sum to 1):")

    st.latex(
        r"""
w_n \propto \exp\left(-\frac{1}{2} \left(\frac{d_n}{\sigma_k}\right)^2 \right),
\qquad
\sigma_k = \max\left(1,\ \frac{L^{\mathrm{barec}}_{k,75} - L^{\mathrm{barec}}_{k,25}}{2} \right)
"""
    )



    st.markdown("#### Step 4 — Index and caps")

    st.markdown(
        """
Index. All unigram–5-gram rows are merged into one index (~10M rows). Each row stores
its teacher label, confidence, *n*, text, and the list of constituent (art_id,
verse_order) IDs.

Per-class cap. For each *k*, we accept at most 10× the BAREC training count
at *k*, so Tarab additions stay in proportion to the gold label histogram and no single
level can absorb the whole corpus.

*Verse uniqueness*: A verse is never used more than once in the final corpus. When we
accept an aggregate (for instance a 5-gram), any other index row that shares any of those verses — unigram, bigram, trigram,
4-gram, or another 5-gram is marked used and skipped on later picks. So accepting a 5-gram effectively
cancels all smaller (and overlapping larger) windows that contain those verses, which
avoids training on nearly duplicate text and prevents leakage across rows.

Within each ($k$, $n$) pool, rows are sorted by teacher confidence descending before
sampling.
"""
    )

    st.markdown("#### Step 5 — Greedy filling: low levels first, shared verses")

    st.markdown(
        """
Global verse consumption. We maintain a set of used unigram IDs
`(art_id, verse_order)`. Any accepted n-gram marks all its verses used; no later
acceptance may reuse them (across classes and across *n*).

Fill order: low → high. Classes are processed from L1 upward.
Intuition: high levels need larger n and therefore more verses per pick. If we
filled L19 before L1, early picks could burn long verse chains that low levels might
have used as smaller windows, and high levels would still need multi-verse aggregates
from what remains. Filling low first leaves a controlled margin: lower levels mostly
consume short, low-*n* candidates; when we reach level *k*, remaining unused verses still
support the longer windows high *k* needs.

Round-based greedy sampling. For each class, filling proceeds in rounds of fixed
batch size (100 picks per round). In one round:

1. Split the 100 slots across active $n$ values in proportion to $w_n$ (largest
   remainder for integers).
2. For each *n*, take that many picks from the top of its confidence-sorted pool, skipping
   rows that overlap used verses.
3. If any *n* runs out mid-round (cannot fill its slot share), that class stops.

We repeat rounds until the class cap is hit or no progress is possible. Unmet targets
for some *n* are accepted: no forced redistribution.
"""
    )

    with st.expander("…", expanded=False):
        st.markdown(
            """
**Joint fill groups (L11/L12, L13/L14, L16/L17)**

Some neighboring levels share the same verse pools because their constituent filters
overlap heavily. Pairs (L11, L12), (L13, L14), and (L16, L17) are filled together
before sequential low→high continues for the rest: each cycle, every active class in
the group runs one round with a share of the batch proportional to its remaining cap,
all drawing from the same global `used_verses` set. That stops L11 from exhausting
verses L12 still needs for higher-*n* windows (and similarly for 13/14 and 16/17).
When one class hits cap or stalls, the partner can keep filling alone.
"""
        )

    st.markdown("#### Step 6 — Final training split")

    st.markdown(
        """
The accepted aggregates are written to a pseudo corpus CSV and merged with all BAREC
train rows (never subsampled) into:

`barec_tarab_length_matching_train.csv`

"""
    )

    _render_length_matching_corpus_length_comparison()


def _render_length_matching_corpus_length_comparison() -> None:
    """Static boxplot comparison: BAREC gold vs Tarab before/after aggregation."""
    specs = (
        (
            LENGTH_MATCHED_BAREC_CHARS_BY_LEVEL_PNG,
            "BAREC (gold train rows in final split)",
        ),
        (
            LENGTH_MATCHED_TARAB_ORIGINAL_CHARS_BY_LEVEL_PNG,
            "Tarab before aggregation (verse-level pseudo corpus)",
        ),
        (
            LENGTH_MATCHED_TARAB_AGGREGATED_CHARS_BY_LEVEL_PNG,
            "Tarab after length matching (10× caps, accepted aggregates)",
        ),
    )

    st.markdown("#### Character length by level — corpus comparison")
    st.caption(
        "Side-by-side boxplots (characters per sentence) after Step 6: BAREC gold in "
        "the final split vs Tarab pseudo before n-gram aggregation vs the length-matched "
        "pseudo corpus that was merged in."
    )

    cols = st.columns(3)
    for col, (path, caption) in zip(cols, specs):
        with col:
            if path.is_file():
                st.image(str(path), caption=caption, use_container_width=True)
            else:
                st.caption(caption)
                st.info(f"Missing chart: `{path.name}`")
