import streamlit as st
import pandas as pd
import numpy as np

# =========================================================
# Inject CSS for compact purple-white tables
# =========================================================
def inject_table_css():
    st.markdown("""
    <style>
        table.custom-table {
            margin: 16px auto;
            border-collapse: collapse;
            font-size: 14px;
            font-family: Inter, system-ui, sans-serif;
            white-space: nowrap;
        }
        table.custom-table th {
            background-color: #5b5fe8;
            color: white !important;
            font-weight: 700;
            padding: 8px 16px;
            text-align: center;
        }
        table.custom-table td {
            padding: 8px 16px;
            text-align: center;
            border-top: 1px solid #e5e7eb;
        }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# Render compact table (NO pandas index)
# =========================================================
def render_compact_table(df):
    html = df.to_html(index=False, classes="custom-table", border=0)
    st.markdown(html, unsafe_allow_html=True)

# =========================================================
# Upload Page
# =========================================================
def upload_page():
    inject_table_css()

    st.header("📂 Upload Dataset")
    st.write("Upload your customer dataset (CSV format).")

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["data"] = df

        st.success("✅ Dataset uploaded successfully!")

        # ===============================
        # Preview of Data (Styled, No Index)
        # ===============================
        st.subheader("🔍 Preview of Data")

        preview_df = df.head().copy()
        render_compact_table(preview_df)

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
        # Column Details (Vertical Flow)
        # ===============================
        st.subheader("📋 Column Details")

        if "show_num" not in st.session_state:
            st.session_state.show_num = False
        if "show_cat" not in st.session_state:
            st.session_state.show_cat = False

        # -------- Button 1: Numerical Columns --------
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

        # -------- Button 2: Categorical Columns --------
        if st.button("Display Categorical Columns"):
            st.session_state.show_cat = not st.session_state.show_cat

        if st.session_state.show_cat:
            cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

            cat_df = pd.DataFrame({
                "Index": range(1, len(cat_cols) + 1),
                "Categorical Columns": cat_cols
            })

            render_compact_table(cat_df)
