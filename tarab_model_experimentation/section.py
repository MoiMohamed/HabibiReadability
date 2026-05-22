from __future__ import annotations

import streamlit as st

from tarab_model_experimentation.experiment_overview import render_experiment_design_overview
from tarab_model_experimentation.performance_comparison import render_performance_comparison_section
from tarab_model_experimentation.selection import list_experiment_log_files

def render_tarab_model_experimentation_section() -> None:
    st.subheader("Tarab model experimentation")

    render_experiment_design_overview()

    log_files = list_experiment_log_files()
    if not log_files:
        st.warning("No `.out` log files found in `data/logs`.")
        return

    render_performance_comparison_section(log_files=log_files)
