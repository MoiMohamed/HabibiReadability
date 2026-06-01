from __future__ import annotations

import streamlit as st

from tarab_model_experimentation import render_tarab_model_experimentation_section
from tarab_model_experimentation.length_matching_overview import render_length_matching_tab
from tarab_model_experimentation.selection import list_experiment_log_files


def main() -> None:
    st.set_page_config(page_title="Tarab Model Experimentation", layout="wide")
    st.title("Tarab Model Experimentation")

    model_tab, length_matching_tab = st.tabs(
        ["Model experimentation", "Length-matching"]
    )

    with model_tab:
        render_tarab_model_experimentation_section()

    with length_matching_tab:
        log_files = list_experiment_log_files()
        render_length_matching_tab(log_files=log_files)


if __name__ == "__main__":
    main()
