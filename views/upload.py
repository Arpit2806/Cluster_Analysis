import streamlit as st
import pandas as pd
import numpy as np


# ===============================
# Inject CSS (Tables + Buttons)
# ===============================
def inject_table_css():
    st.markdown("""
    <style>
        /* ===============================
           TABLE STYLING
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
            color: white !important;
            font-weight: 700;
            padding: 10px 18px;
            text-align: center;
            border-right: 1px solid #4f52d9;
        }

        table.custom-table td {
            background-color: #f5f6fa;
            color: #1f2937;
            padding: 10px 18px;
            text-align: center;
            border-top: 1px solid #e5e7eb;
            border-right: 1px solid #e5e7eb;
        }

        table.custom-table tbody tr:nth-child(even) td {
            background-color: #eef0f7;
        }

        /* ===============================
           BUTTON STYLING (FIXED)
           =============================== */
        div.stButton > button {
            background-color: #5b5fe8 !important;
            color: #ffffff !important;
            font-weight: 600;
            font-size: 15px;
            border-radius: 12px;
            padding: 0.7rem 1.6rem;
            border: none;
        }

        /* IMPORTANT: force text color */
        div.stButton > button span {
            color: #ffffff !important;
        }

        div.stButton > button:hover {
            background-color: #4a4fd8 !important;
            color: #ffffff !important;
        }

        div.stButton > button:hover span {
            color: #ffffff !important;
        }

        div.stButton > button:active,
        div.stButton > button:focus {
            background-color: #4348c9 !important;
            color: #ffffff !important;
            box-shadow: none !important;
        }

        /* File uploader button */
        div[data-testid="stFileUploader"] button {
            background-color: #5b5fe8 !important;
            color: #ffffff !important;
            font-weight: 600;
            border-radius: 12px;
            border: none;
        }

        div[data-testid="stFileUploader"] button span {
            color: #ffffff !important;
        }

        div[data-testid="stFileUploader"] button:hover {
            background-color: #4a4fd8 !important;
        }
    </style>
    """, unsafe_allow_html=True)


# ===============================
# Render Styled Table
# ===============================
def render_compact_table(df):
    html = df.to_html(index=False, classes="custom-table", border=0)
    st.markdown(f"<div class='table-wrapper'>{html}</div>", unsafe_allow_html=True)


# ===============================
# Upload Page (Session Persistent)
# ===============================
def upload_page():
    inject_table_css()

    st.header("📂 Upload Dataset")
    st.write("Upload your customer dataset (CSV format).")

    # ===============================
    # Reset Dataset
    # ===============================
    if "data" in st.session_state:
        if st.button("🔄 Reset Dataset"):
            st.session_state.clear()
            st.experimental_rerun()

    # ===============================
    # Upload / Reuse Dataset
    # ===============================
    if "data" in st.session_state:
        df = st.session_state["data"]
        st.success("✅ Dataset already loaded")
    else:
        uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

        if uploaded_file is None:
            st.info("📌 Please upload a CSV file to continue")
            return

        df = pd.read_csv(uploaded_file)
        st.session_state["data"] = df
        st.success("✅ Dataset uploaded successfully!")

    # ===============================
    # Preview of Data
    # ===============================
    st.subheader("🔍 Preview of Data")
    render_compact_table(df.head())

    st.divider()

    # ===============================
    # Dataset Overview
    # ===============================
    st.subheader("📊 Dataset Overview")

    overview_df = pd.DataFrame({
        "Metric": [
            "Total Rows",
            "Total Columns",
            "Numerical Columns",
            "Categorical Columns"
        ],
        "Value": [
            df.shape[0],
            df.shape[1],
            df.select_dtypes(include=np.number).shape[1],
            df.select_dtypes(exclude=np.number).shape[1]
        ]
    })

    render_compact_table(overview_df)

    st.divider()

    # ===============================
    # Column Details
    # ===============================
    st.subheader("📋 Column Details")

    if "show_num" not in st.session_state:
        st.session_state.show_num = False
    if "show_cat" not in st.session_state:
        st.session_state.show_cat = False

    if st.button("Display Numerical Columns"):
        st.session_state.show_num = not st.session_state.show_num

    if st.session_state.show_num:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        num_df = pd.DataFrame({
            "Index": range(1, len(num_cols) + 1),
            "Numerical Columns": num_cols
        })
        render_compact_table(num_df)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Display Categorical Columns"):
        st.session_state.show_cat = not st.session_state.show_cat

    if st.session_state.show_cat:
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        cat_df = pd.DataFrame({
            "Index": range(1, len(cat_cols) + 1),
            "Categorical Columns": cat_cols
        })
        render_compact_table(cat_df)
