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
            border-collapse: collapse;
            font-size: 14px;
            font-family: Inter, system-ui, sans-serif;
            white-space: nowrap;
            margin: 16px;
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
# Render compact table (NO stretch, NO index)
# =========================================================
def render_compact_table(df):
    df = df.reset_index(drop=True)
    html = df.to_html(index=False, classes="custom-table", border=0)
    st.markdown(html, unsafe_allow_html=True)

# =========================================================
# Upload Page
# =========================================================
def upload_page():
    inject_table_css()   # ✅ now defined before use

    st.header("📂 Upload Dataset")
    st.write("Upload your customer dataset (CSV format).")

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["data"] = df

        st.success("✅ Dataset uploaded successfully!")

        # ===============================
        # Preview of Data (styled)
        # ===============================
        st.subheader("🔍 Preview of Data")

        st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
        render_compact_table(df.head())
        st.markdown("</div>", unsafe_allow_html=True)

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

        st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
        render_compact_table(overview_df)
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # ===============================
        # Column Details
        # ===============================
        st.subheader("📋 Column Details")

        if "show_num" not in st.session_state:
            st.session_state.show_num = False
        if "show_cat" not in st.session_state:
            st.session_state.show_cat = False

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Display Numerical Columns"):
                st.session_state.show_num = not st.session_state.show_num

        with col2:
            if st.button("Display Categorical Columns"):
                st.session_state.show_cat = not st.session_state.show_cat

        # Numerical → LEFT
        if st.session_state.show_num:
            num_df = pd.DataFrame({
                "Numerical Columns": df.select_dtypes(include=np.number).columns
            })

            st.markdown("<div style='display:flex; justify-content:flex-start;'>", unsafe_allow_html=True)
            render_compact_table(num_df)
            st.markdown("</div>", unsafe_allow_html=True)

        # Categorical → RIGHT
        if st.session_state.show_cat:
            cat_df = pd.DataFrame({
                "Categorical Columns": df.select_dtypes(exclude=np.number).columns
            })

            st.markdown("<div style='display:flex; justify-content:flex-end;'>", unsafe_allow_html=True)
            render_compact_table(cat_df)
            st.markdown("</div>", unsafe_allow_html=True)
