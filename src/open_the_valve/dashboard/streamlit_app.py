"""Findings dashboard. Run with:

uv run streamlit run src/open_the_valve/dashboard/streamlit_app.py
"""

import os

import pandas as pd
import streamlit as st

from open_the_valve.config_loader import load_app_config

st.set_page_config(page_title="Open The Valve — Findings", layout="wide")

config = load_app_config()
findings_dir = os.path.dirname(config.causal.findings.output_path)

comparison_table = pd.read_csv(os.path.join(findings_dir, "comparison_table.csv"))
cate_slices = pd.read_csv(os.path.join(findings_dir, "cate_slices.csv"))
with open(config.causal.findings.output_path) as f:
    findings_markdown = f.read()

view = st.sidebar.radio("View", ["Overview", "CATE by slice"])

if view == "Overview":
    st.header("Method comparison")
    st.dataframe(comparison_table, use_container_width=True)
    st.divider()
    st.markdown(findings_markdown)
else:
    st.header("CATE by slice")
    col1, col2, col3 = st.columns(3)
    slice_dim = col1.selectbox("Slice dimension", sorted(cate_slices["slice_dim"].unique()))
    estimator = col2.selectbox("Estimator", sorted(cate_slices["estimator"].unique()))
    include_exploratory = col3.checkbox("Include exploratory dims", value=False)

    filtered = cate_slices[
        (cate_slices["slice_dim"] == slice_dim) & (cate_slices["estimator"] == estimator)
    ]
    if not include_exploratory:
        filtered = filtered[~filtered["is_exploratory"]]

    st.bar_chart(filtered.set_index("slice_value")["mean_effect"])
    st.dataframe(filtered, use_container_width=True)
