import streamlit as st
import pandas as pd
import numpy as np


# ===============================
# Inject CSS (SAFE VERSION)
# ===============================
def inject_table_css():
    st.markdown("""
    <style>
    /* ===============================
       TABLE STYLING (HTML TABLES)
       =============================== */
    .table-wrapper {
        max-width: 100%;
        overflow-x: auto;
        margin: 16px auto;
    }

    table.custom-table {
        border-collapse: collapse;
        font-size: 14px;
        font-family: Inter, system-ui, sans-serif;
        white-space: nowrap;
        margin: 0 auto;
    }

    table.custom-table th {
        background-color: #5b5fe8;
        color: white;
        font-weight: 700;
        padding: 10px 18px;
        text-align: center;
    }

    table.custom-table td {
        background-color: #f5f6fa;
        color: #1f2937;
        padding: 10px 18px;
        text-align: center;
        border-top: 1px solid #e5e7eb;
    }

    table.custom-table tbody tr:nth-child(even) td {
        background-color: #eef0f7;
    }

    /* ===============================
       BUTTON STYLING (SAFE)
       =============================== */
    div.stButton > button {
        background-color: #5b5fe8;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #4a4fd8;
        color: white;
    }

    /* File uploader button */
    div[data-testid="stFileUploader"] button {
        background-color: #5b5fe8;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        border: none;
    }

    div[data-testid="stFileUploader"] button:hover {
        background-color: #4a4fd8;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


# ===============================
# Render HTML Table
# ===============================
def render_compact_table(df):
    html = df.to_html(index=False, classes="custom-table", border=0)
    st.markdown(f"<div class='table-wrapper'>{html}</div>", unsafe_allow_html=True)


# ===============================
# Upload Page
# ===============================
def upload_page():
    inject_table_css()

    st.header("📂 Upload Dataset")
    st.write("Upload your customer dataset (CSV format).")

    # Reset
    if "data" in st.session_state:
        if st.button("🔄 Reset Dataset"):
            st.session_state.clear()
            st.experimental_rerun()

    # Upload / reuse
    if "data" in st.session_state:
        df = st.session_state["data"]
        st.success("✅ Dataset already loaded")
    else:
        uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
        if uploaded_file is None:
            st.info("📌 Please upload a CSV file")
            return
        df = pd.read_csv(uploaded_file)
        st.session_state["data"] = df
        st.success("✅ Dataset uploaded successfully!")

    # Preview
    st.subheader("🔍 Preview of Data")
    render_compact_table(df.head())

    st.divider()

    # Overview
    st.subheader("📊 Dataset Overview")
    overview_df = pd.DataFrame({
        "Metric": ["Total Rows", "Total Columns"],
        "Value": [df.shape[0], df.shape[1]]
    })
    render_compact_table(overview_df)

    st.divider()

    # Column Details
    st.subheader("📋 Column Details")

    if "show_num" not in st.session_state:
        st.session_state.show_num = False
    if "show_cat" not in st.session_state:
        st.session_state.show_cat = False

    if st.button("Display Numerical Columns"):
        st.session_state.show_num = not st.session_state.show_num

    if st.session_state.show_num:
        render_compact_table(
            pd.DataFrame({"Numerical Columns": df.select_dtypes(include=np.number).columns})
        )

    if st.button("Display Categorical Columns"):
        st.session_state.show_cat = not st.session_state.show_cat

    if st.session_state.show_cat:
        render_compact_table(
            pd.DataFrame({"Categorical Columns": df.select_dtypes(exclude=np.number).columns})
        )
