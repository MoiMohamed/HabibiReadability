from __future__ import annotations

import streamlit as st

from tarab_model_experimentation import render_tarab_model_experimentation_section


def main() -> None:
    st.set_page_config(page_title="Tarab Model Experimentation", layout="wide")
    st.title("Tarab Model Experimentation")
    render_tarab_model_experimentation_section()


if __name__ == "__main__":
    main()
