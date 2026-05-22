from __future__ import annotations

import streamlit as st

from tarab_model_experimentation.confusion_comparison import (
    COMPARISON_SPECS,
    _render_one_comparison,
)
from tarab_model_experimentation.class19_investigation import (
    render_dist_155k_confidence_distribution_section,
    render_dist_155k_song_poem_section,
    render_dist_155k_text_length_section,
    render_pretraining_label_distribution_section,
)
from tarab_model_experimentation.test_results import render_primary_test_results_section


def render_focused_baseline_vs_dist_155k_section(*, log_files: list[str]) -> None:
    """Training diagnostics + dev error analysis for baseline vs dist_155K only."""
    st.divider()
    st.header("Focused analysis: baseline vs dist_155K")

    st.subheader("Training data overview")
    render_pretraining_label_distribution_section()
    render_dist_155k_confidence_distribution_section()
    render_dist_155k_text_length_section()
    render_dist_155k_song_poem_section()

    st.subheader("Prediction and error analysis")
    for title, compare_labels in COMPARISON_SPECS:
        _render_one_comparison(
            log_files=log_files, title=title, compare_labels=compare_labels
        )

    render_primary_test_results_section(log_files)
