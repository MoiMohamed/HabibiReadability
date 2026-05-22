from __future__ import annotations

import html
import re
from typing import Any

import streamlit as st

_INSIGHT_STYLE = """
<style>
.habibi-insight {
    border-left: 4px solid #2a6f97;
    background: linear-gradient(90deg, #f0f7fb 0%, #fafbfc 100%);
    padding: 1rem 1.15rem 0.85rem 1.15rem;
    margin: 0.75rem 0 1.25rem 0;
    border-radius: 0 8px 8px 0;
}
.habibi-insight .insight-tag {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #2a6f97;
    margin-bottom: 0.35rem;
}
.habibi-insight .insight-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1a1a1a;
    margin: 0 0 0.5rem 0;
    line-height: 1.35;
}
.habibi-insight ul {
    margin: 0.35rem 0 0.5rem 1.1rem;
    padding: 0;
    color: #333;
    font-size: 0.95rem;
    line-height: 1.5;
}
.habibi-insight .insight-next {
    margin: 0.65rem 0 0 0;
    font-size: 0.88rem;
    color: #555;
    font-style: italic;
}
.habibi-insight .insight-metrics {
    margin: 0.35rem 0 0.15rem 0;
    font-size: 0.9rem;
    color: #444;
    line-height: 1.45;
}
.habibi-insight .insight-metrics-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: #2a6f97;
    margin: 0.45rem 0 0.35rem 0;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.habibi-insight table.insight-metrics-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    margin: 0 0 0.5rem 0;
}
.habibi-insight table.insight-metrics-table th {
    text-align: left;
    color: #2a6f97;
    font-weight: 600;
    font-size: 0.8rem;
    border-bottom: 1px solid #b8d4e3;
    padding: 0.35rem 0.5rem;
}
.habibi-insight table.insight-metrics-table td {
    padding: 0.32rem 0.5rem;
    color: #333;
    border-bottom: 1px solid #e8eef2;
}
.habibi-insight table.insight-metrics-table tr.section-head td {
    font-size: 0.78rem;
    font-weight: 600;
    color: #555;
    background: rgba(42, 111, 151, 0.06);
    border-bottom: 1px solid #d0e3ed;
    padding-top: 0.45rem;
}
.habibi-insight .delta-chip {
    display: inline-block;
    font-weight: 600;
    font-size: 0.82rem;
    white-space: nowrap;
}
.habibi-insight .delta-good { color: #1e6b42; }
.habibi-insight .delta-bad { color: #b33030; }
.habibi-insight .delta-flat { color: #666; }
.habibi-insight td.col-dist { color: #8c4a12; font-weight: 500; }
</style>
"""


def _inline_md_to_html(text: str) -> str:
    """``**bold**`` → ``<strong>`` for HTML insight cards."""
    parts: list[str] = []
    last = 0
    for match in re.finditer(r"\*\*(.+?)\*\*", text):
        parts.append(html.escape(text[last : match.start()]))
        parts.append(f"<strong>{html.escape(match.group(1))}</strong>")
        last = match.end()
    parts.append(html.escape(text[last:]))
    return "".join(parts)


def _delta_chip_html(
    delta: float,
    *,
    higher_is_better: bool = True,
    decimals: int = 2,
    flat_threshold: float = 0.05,
) -> str:
    if abs(delta) < flat_threshold:
        cls, text = "delta-flat", "≈0"
    else:
        good = (delta > 0) == higher_is_better
        cls = "delta-good" if good else "delta-bad"
        sign = "+" if delta >= 0 else ""
        text = f"{sign}{delta:.{decimals}f}"
    return f'<span class="delta-chip {cls}">{html.escape(text)}</span>'


def _metrics_table_row(
    label: str,
    base: float,
    exp: float,
    *,
    as_rate: bool = True,
    higher_is_better: bool = True,
) -> str:
    if as_rate:
        delta = 100.0 * (exp - base)
        chip = _delta_chip_html(delta, higher_is_better=higher_is_better)
        b_disp, e_disp = _pct(base), _pct(exp)
    else:
        delta = exp - base
        chip = _delta_chip_html(
            delta,
            higher_is_better=higher_is_better,
            decimals=3,
            flat_threshold=0.0005,
        )
        b_disp, e_disp = f"{base:.3f}", f"{exp:.3f}"
    return (
        f"<tr><td>{html.escape(label)}</td>"
        f"<td>{html.escape(b_disp)}</td>"
        f'<td class="col-dist">{html.escape(e_disp)}</td>'
        f"<td>{chip}</td></tr>"
    )


def _metrics_comparison_table_html(
    *,
    caption: str,
    sections: list[tuple[str, list[tuple[str, float, float, bool, bool]]]],
) -> str:
    """
    sections: (section_title, rows) where each row is
    (label, baseline, dist_155K, as_rate, higher_is_better).
    """
    body: list[str] = []
    for section_title, rows in sections:
        body.append(
            f'<tr class="section-head"><td colspan="4">{html.escape(section_title)}</td></tr>'
        )
        for label, base, exp, as_rate, hib in rows:
            body.append(
                _metrics_table_row(label, base, exp, as_rate=as_rate, higher_is_better=hib)
            )
    return (
        f'<p class="insight-metrics-label">{html.escape(caption)}</p>'
        "<table class=\"insight-metrics-table\">"
        "<thead><tr><th>Metric</th><th>Baseline</th><th>dist_155K</th><th>Δ</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def render_insight_card(
    *,
    tag: str = "Insight",
    title: str,
    bullets: list[str],
    metrics_table_html: str | None = None,
    next_step: str | None = None,
) -> None:
    """Reusable presentation insight block (styled HTML card + inline bold)."""
    bullet_html = "".join(
        f"<li>{_inline_md_to_html(bullet)}</li>" for bullet in bullets
    )
    metrics_html = metrics_table_html or ""
    next_html = (
        f'<p class="insight-next"><strong>Next:</strong> {_inline_md_to_html(next_step)}</p>'
        if next_step
        else ""
    )
    st.markdown(
        _INSIGHT_STYLE
        + f"""
<div class="habibi-insight">
  <div class="insight-tag">{html.escape(tag)}</div>
  <div class="insight-title">{html.escape(title)}</div>
  <ul>{bullet_html}</ul>
  {metrics_html}
  {next_html}
</div>
""",
        unsafe_allow_html=True,
    )


def _profile_at_qwk(rows: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("experiment") == label:
            return row.get("profiles", {}).get("qwk")
    return None


def _pct(rate: float) -> str:
    """Proportion → display percent (e.g. 0.8041 → 80.41%)."""
    return f"{100.0 * rate:.2f}%"


def render_qwk_profile_match_distribution_insight(
    acc_rows: list[dict[str, Any]],
    *,
    log_files: list[str] | None = None,
) -> None:
    """Insight after accuracy/distance @ QWK-optimal: match-distribution story, focus dist_155K."""
    from tarab_model_experimentation.metric_optimal_profiles import (
        collect_metric_optimal_profile_rows,
    )

    base_p = _profile_at_qwk(acc_rows, "baseline")
    d155_p = _profile_at_qwk(acc_rows, "dist_155K")
    if base_p is None or d155_p is None:
        return

    prf_rows = (
        collect_metric_optimal_profile_rows(log_files or [])
        if log_files
        else []
    )
    f1_base = f1_155 = p_base = p_155 = r_base = r_155 = None
    for row in prf_rows:
        prof = row.get("profiles", {}).get("qwk")
        if prof is None:
            continue
        if row["experiment"] == "baseline":
            f1_base = prof.get("f1")
            p_base = prof.get("precision")
            r_base = prof.get("recall")
        elif row["experiment"] == "dist_155K":
            f1_155 = prof.get("f1")
            p_155 = prof.get("precision")
            r_155 = prof.get("recall")

    b_qwk, e_qwk = float(base_p["qwk"]), float(d155_p["qwk"])
    b_acc, e_acc = float(base_p["acc"]), float(d155_p["acc"])
    b_dist, e_dist = float(base_p["dist"]), float(d155_p["dist"])
    b_acc1, e_acc1 = float(base_p["acc_pm1"]), float(d155_p["acc_pm1"])

    match_qwk: list[tuple[str, float]] = []
    for row in acc_rows:
        lab = row["experiment"]
        if not lab.startswith("dist_"):
            continue
        p = row["profiles"].get("qwk")
        if p and p.get("qwk") is not None:
            match_qwk.append((lab, float(p["qwk"])))
    match_qwk.sort(key=lambda x: x[1], reverse=True)

    table_sections: list[tuple[str, list[tuple[str, float, float, bool, bool]]]] = [
        (
            "At each run’s QWK-best checkpoint",
            [
                ("QWK", b_qwk, e_qwk, True, True),
                ("Acc19", b_acc, e_acc, True, True),
                ("Distance", b_dist, e_dist, False, False),
                ("Acc±1", b_acc1, e_acc1, True, True),
            ],
        ),
    ]
    if f1_base is not None and f1_155 is not None:
        table_sections.append(
            (
                "Weighted P / R / F1 (same epochs)",
                [
                    ("Precision", float(p_base), float(p_155), True, True),
                    ("Recall", float(r_base), float(r_155), True, True),
                    ("F1", float(f1_base), float(f1_155), True, True),
                ],
            )
        )
    metrics_table = _metrics_comparison_table_html(
        caption="Key numbers: dist_155K vs baseline",
        sections=table_sections,
    )

    best_match = match_qwk[0][0] if match_qwk else "dist_155K"
    bullets = [
        "Experiment set A (match distribution) shows the clearest lift, especially "
        "dist_155K, which is the strongest match run on dev among 81K / 108K / 155K / 245K.",
        "At the QWK-optimal checkpoint, dist_155K improves precision, recall, and F1, "
        "which lines up with higher Acc19 / Acc±1 and lower distance, a coherent "
        "“better on almost everything” pattern that is easy to see on F1 and accuracy panels.",
        "QWK barely moves (often slightly below baseline). QWK weights squared error "
        "on the full 19×19 grid. Gains can be real on P/R/F1 while off-diagonal / far-off "
        "mistakes still cap QWK.",
        "Set B (uniform) does not beat match at this checkpoint, but Acc19 and F1 rise "
        "steadily with k per class — roughly linear 3k→6k — so larger caps (7k, 8k, …) "
        "are worth trying.",
    ]

    render_insight_card(
        tag="Insight",
        title="Match-distribution distillation looks promising, but QWK understates it",
        metrics_table_html=metrics_table,
        bullets=bullets,
        next_step=(
            "Focused **performance and error analysis on dist_155K vs baseline** "
            f"(best match label in this chart: **{best_match}**)."
        ),
    )


def render_text_length_insight() -> None:
    """Insight after BAREC vs Tarab text length boxplots."""
    render_insight_card(
        tag="Insight",
        title="Tarab pseudo-labels invert BAREC's length signal",
        bullets=[
            "**BAREC encodes length as a readability signal.** Median words per sentence "
            "scale from ~1 at L1 to ~24 at L14–16; characters grow from ~5 to ~120 "
            "over the same range. Length and readability move together in the gold data.",
            "**Tarab pseudo-labels are short fragments at every level.** Median Tarab "
            "sentences are 2–7 words / 10–27 characters across L1–L19 (song and poem "
            "lines). A 4-word line in BAREC looks like ~L3–L5; the same 4-word line in "
            "Tarab is pseudo-labeled L14+.",
            "**This is a covariate shift on a strong classical feature.** During "
            "distillation the model sees BAREC saying *long means high* and Tarab saying "
            "*short means anything*. That weakens length as a cue and likely contributes to "
            "the prediction-distribution shifts we see at higher levels.",
            "**This is likely a main driver of the catastrophic far-off mistakes we will "
            "see later in the prediction analysis**, and what most contributes to the loss "
            "of gains from pseudo-labeling. Short BAREC fragments at high gold levels "
            "(titles, headers, one-line snippets) lose their length-based anchor after "
            "distillation, so dist_155K pulls them down to low Tarab-like levels, "
            "producing |err| ≥ 7 errors that the baseline does not make.",
        ],
        next_step=(
            "**length-stratified pseudo-labeling** should help."
        ),
    )


def render_training_variant_overview_insight() -> None:
    """Before ALDi × AGS and song/poem charts: context only, not the current fix focus."""
    render_insight_card(
        tag="Insight",
        title="ALDi × AGS and song/poem: training-set overview for later",
        bullets=[
            "The panels below summarize **dist_155K** training mix only: **ALDi × AGS** "
            "label pairs (BAREC vs Tarab pseudo) and **song vs poem** share by readability "
            "level. They are here for completeness, not as the main diagnosis in this update.",
            "Variant-aware matching (pseudo-label type, ALDi/AGS alignment with BAREC) is "
            "worth exploring **after** we address the dominant issues already flagged above: "
            "**text length** covariate shift and **vocabulary** overlap with BAREC.",
        ],
    )


def render_shift_vs_confidence_insight() -> None:
    """After the per-class recall & precision panel in the prediction analysis."""
    render_insight_card(
        tag="Insight",
        title="Pseudo-labeling shifts predictions toward gold, but recall drops",
        bullets=[
            "**Pseudo-labeling shifts the predicted distribution in a clearly positive "
            "direction.** dist_155K predictions move closer to the dev gold mix exactly at "
            "the high-support classes the baseline under-predicts.",
            "**But QWK barely moves and per-class recall drops on several levels even "
            "though we are shifting toward the right predictions.** That tension forces "
            "two competing explanations:",
            "**Hypothesis 1 — Distribution shift.** Match-distribution filling adds large pseudo-label "
            "mass at high-support classes (L10, L12). Decision boundaries move, neighboring "
            "classes get absorbed, and some lose recall not because their own labels were "
            "noisy but because another class became stronger.",
            "**Hypothesis 2 — Pseudo-label confidence noise.** Lower-confidence Tarab labels at "
            "surrounding classes inject wrong supervision, corrupt boundaries, and drag "
            "recall down on those classes directly.",
            "**Why Δ precision is the right test for H2.** If lower-confidence pseudo-labels "
            "were corrupting supervision, predictions of that class would become less "
            "reliable — exactly what precision measures. So if noise dominates, classes with lower median Tarab "
            "confidence should lose precision; if shift dominates, confidence and Δ "
            "precision should be unrelated.",
            "**The confidence vs Δ precision chart rejects H2.** Across all 19 levels, and "
            "within high- and low-support subsets, there is no meaningful relationship: "
            "low-confidence classes can gain precision and high-confidence classes can "
            "lose it. Confidence is not the lever.",
            "**The recall/precision panel shows H1's signature.** Some classes (L19, L10) "
            "lose recall while gaining precision: the model predicts them less often but "
            "more correctly, which indicates boundary movement, not noise. "
            "Their mass goes to attractor classes (L2, L12), which become more aggressive "
            "and gain recall at the cost of precision.",
            "**Conclusion.** Recall losses come from **class-mass redistribution toward "
            "high-support attractors**, not from pseudo-label confidence. The ΔC migration heatmap "
            "below shows exactly where the mass moves.",
        ],
        next_step=(
            "**distribution control**, not confidence bands — cap per-class mass "
            "or rebalance attractors (see below)."
        ),
    )


def render_unsigned_error_bridge_insight() -> None:
    """After unsigned error-distance chart; bridge to far-off catastrophic analysis."""
    render_insight_card(
        tag="Insight",
        title="Pseudo-labels add fixes, but the far tail costs QWK",
        bullets=[
            "Dist_155K wins on **exact match** and trims many small and mid-range mistakes "
            "(green deltas at distances 2–6): the gains from shifting predictions toward gold.",
            "Those gains are **more than erased** by a fatter tail at |err| ≥ 7 (red deltas). "
            "QWK weights squared distance, so a handful of catastrophic errors outweighs "
            "the bulk of near-correct fixes.",
            "The table below isolates those far-off rows, where the tail comes from and "
            "how much of the QWK gap they explain.",
        ],
    )


def render_qwk_contribution_insight(
    df: Any,
    summary: dict[str, Any],
    compare_labels: tuple[str, str] = ("baseline", "dist_155K"),
) -> None:
    """Insight under the QWK decomposition chart."""
    if df is None or len(df) == 0:
        return

    qwk_b = float(summary["qwk_baseline"])
    qwk_d = float(summary["qwk_dist"])
    qwk_ceiling = float(summary["qwk_fix_losses"])
    n_harm = int(summary.get("n_rows_loss", 0))
    n_dev = int(df["support"].sum()) if "support" in df.columns else 0
    harm_pct = 100.0 * n_harm / n_dev if n_dev else 0.0

    worst_row = df.loc[df["delta_penalty"].idxmin()]
    worst_level = int(worst_row["level"])
    best_row = df.loc[df["delta_penalty"].idxmax()]
    best_level = int(best_row["level"])
    deepest_loss_level = int(df.loc[df["loss_negative"].idxmin(), "level"])

    dqwk_ceiling = 100.0 * (qwk_ceiling - qwk_d)
    dqwk_actual = 100.0 * (qwk_d - qwk_b)

    from tarab_model_experimentation.dev_predictions import far_off_inversion_pattern_counts

    inv = far_off_inversion_pattern_counts(compare_labels) or {}
    n_high_low = int(inv.get("gold_12_15_pred_2_3", 0))
    n_high_to_5 = int(inv.get("gold_12_16_pred_5", 0))
    n_l5_high = int(inv.get("gold_5_pred_12_plus", 0))

    metrics_table = _metrics_comparison_table_html(
        caption="QWK on dev",
        sections=[
            (
                "Today",
                [
                    ("QWK", qwk_b, qwk_d, True, True),
                ],
            ),
            (
                "Potential gains: revert every dev row where dist regressed (not deployable)",
                [
                    ("QWK", qwk_d, qwk_ceiling, True, True),
                ],
            ),
        ],
    )

    render_insight_card(
        tag="Insight",
        title="Pseudo-labeling wins rows and loses rows: QWK adds them up",
        bullets=[
            "**The decomposition matches the catastrophic shift story from above.** "
            f"The deepest red is at **L{deepest_loss_level}**; the worst net penalty "
            f"is at **L{worst_level}**. Both are levels where dist often predicts far "
            "from gold. The |err| ≥ 7 table repeats the same pattern: "
            f"**{n_high_low}** rows with gold **L12-L15** as **L2-L3**, "
            f"**{n_high_to_5}** with gold **L12-L16** as **L5**, and "
            f"**{n_l5_high}** with gold **L5** as **L12+**. "
            "That far-off mass drives most of the QWK penalty and most of the "
            "potential-gains room.",
            "**Regression mass is not confined to the far tail.** "
            f"**{n_harm:,}** dev examples (**{harm_pct:.0f}%**) exhibit higher squared "
            "distance under dist_155K than under baseline. Most are moderate "
            "|pred − true| shifts (red on several levels, sometimes alongside green on "
            "the same gold level), distinct "
            "from the |err| ≥ 7 catastrophes. QWK sums squared distance over all rows, "
            "so this diffuse red mass alone is enough to hold the metric near baseline "
            "even where per-level green components improve.",
            "**Counterfactual bound on recoverable QWK.** "
            "The potential-gains row (revert every regressed row) "
            f"attains **{100.0 * qwk_ceiling:.2f}%** QWK (**{dqwk_ceiling:+.2f}** vs dist, "
            f"**{dqwk_actual:+.2f}** today). Full correction is unrealistic, but the "
            "bound indicates that partial reduction of regression mass, distribution "
            "control, length-aligned pseudo-labels, and ultra-short fragment filters, "
            f"could improve QWK while preserving gains at **L{best_level}** and comparable "
            "green levels.",
        ],
        metrics_table_html=metrics_table,
        next_step=(
            "**Distribution control**, **length-matched**, **variant matching** Tarab pseudo-labels shall be the next step and focus."
        ),
    )


def render_far_off_mistakes_insight(stats: dict[str, Any]) -> None:
    """Insight under the table of |err| ≥ 7 dist mistakes vs baseline."""
    n = int(stats["n_extra"])
    already_wrong = n - int(stats["baseline_exact"])
    short = int(stats["short_under_50_chars"])
    mean_base = float(stats["mean_base_err"])
    mean_dist = float(stats["mean_dist_err"])
    qwk_baseline = 100.0 * float(stats["qwk_baseline"])
    qwk_dist = 100.0 * float(stats["qwk_dist"])
    qwk_fixed = 100.0 * float(stats["qwk_if_extra_reverted_to_baseline"])
    dqwk_pp = 100.0 * float(stats["dqwk_fix_extra"])

    render_insight_card(
        tag="Insight",
        title="Far-off errors are amplified borderline cases on ultra-short text",
        bullets=[
            "These rows are mostly **decontextualized fragments** (titles, single words, "
            "bylines, short pedagogical phrases) and **structured markup** (e.g. Wikipedia "
            "`== … ==` headers).",
            f"Baseline was already wrong on **{already_wrong} of {n}** (mean |err| "
            f"~{mean_base:.1f}); dist_155K pushes the same cases to ~{mean_dist:.1f}. "
            "Distillation rarely invents failures from scratch: it amplifies the boundary "
            "movement we saw in the ΔC heatmap.",
            f"**{short} of {n}** are under 50 characters: the same ultra-short, Tarab-shaped "
            "texts from the length insight, now showing up as |err| ≥ 7 catastrophes.",
            f"Almost the whole QWK gap sits here. Reverting only these {n} rows to baseline "
            f"predictions moves dev QWK from **{qwk_dist:.2f}%** to **{qwk_fixed:.2f}%** "
            f"(**{dqwk_pp:+.2f}**), above baseline (**{qwk_baseline:.2f}%**).",
        ],
        next_step=(
            "**length-matched** pseudo-labels and **distribution control** — not "
            "confidence bands."
        ),
    )


def render_vocab_overlap_insight() -> None:
    """Insight after Tarab ↔ BAREC vocabulary overlap (Inoue et al., WANLP 2021)."""
    render_insight_card(
        tag="Insight",
        title="Align Tarab pseudo-label vocabulary with BAREC",
        bullets=[
            "Inoue et al., **The Interplay of Variant, Size, and Task Type in Arabic "
            "Pre-trained Language Models** (WANLP 2021, §5.2): pre-training text closest to the "
            "fine-tuning variant wins more often than larger but mismatched pre-training; lower "
            "downstream OOV tracks better scores (ρ ≈ −0.82).",
            "Future experiments should try vocabulary-matched Tarab pseudo-labels (filter or "
            "rank by BAREC type overlap), not only training size and label distribution.",
            "We cannot read that correlation off this chart yet: training-set size and Tarab "
            "volume move together, so shared-vocabulary % is confounded: hold size fixed and "
            "vary overlap deliberately.",
            "As we will see later, we should not treat lower **readability_confidence** as a "
            "reason to avoid vocabulary matching: confidence bands are not injecting harmful "
            "noise. The shifts we will see come from changed prediction distributions, not from "
            "noisy low-confidence labels alone.",
        ],
    )
