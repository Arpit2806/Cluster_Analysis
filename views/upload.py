import streamlit as st
import pandas as pd
import numpy as np


# ===============================
# CSS (TABLE + BUTTON STYLING)
# ===============================
def inject_css():
    st.markdown("""
    <style>
        /* ---------- TABLE ---------- */
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
            background: linear-gradient(135deg, #5b5fe8, #6d71ff);
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

        /* ---------- BUTTONS ---------- */
        div.stButton > button {
            width: 100%;
            text-align: left;
            font-weight: 700;
            font-size: 14px;
            color: white !important;
            background: linear-gradient(135deg, #5b5fe8, #6d71ff);
            border: none;
            border-radius: 10px;
            padding: 10px 16px;
            margin-bottom: 10px;
            transition: all 0.2s ease;
        }

        div.stButton > button:hover {
            background: linear-gradient(135deg, #4a4fd8, #5b60ff);
            transform: translateY(-1px);
        }
    </style>
    """, unsafe_allow_html=True)


def render_compact_table(df):
    html = df.to_html(index=False, classes="custom-table", border=0)
    st.markdown(f"<div class='table-wrapper'>{html}</div>", unsafe_allow_html=True)


# ===============================
# UPLOAD PAGE
# ===============================
def upload_page():
    inject_css()

    st.header("📂 Upload Dataset")
    st.write("Upload your customer dataset (CSV format).")

    # ---------- RESET BUTTON ----------
    if "data" in st.session_state:
        with st.form("reset_form"):
            reset = st.form_submit_button("🔁 Reset Dataset")
            if reset:
                st.session_state.clear()
                st.experimental_rerun()

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if not uploaded_file:
        return

    df = pd.read_csv(uploaded_file)
    st.session_state["data"] = df
    st.success("✅ Dataset uploaded successfully!")

    # ===============================
    # Preview
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

    # ---------- BUTTON FORM ----------
    with st.form("column_buttons"):
        show_num = st.form_submit_button("📈 Display Numerical Columns")
        if show_num:
        st.session_state.show_num = not st.session_state.show_num

            # ---------- DISPLAY TABLES ----------
        if st.session_state.show_num:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        num_df = pd.DataFrame({
            "Index": range(1, len(num_cols) + 1),
            "Numerical Columns": num_cols
        })
        render_compact_table(num_df)

        show_cat = st.form_submit_button("🏷 Display Categorical Columns")

        if show_num:
        st.session_state.show_num = not st.session_state.show_num

        if show_cat:
        st.session_state.show_cat = not st.session_state.show_cat

    # ---------- DISPLAY TABLES ----------
    if st.session_state.show_num:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        num_df = pd.DataFrame({
            "Index": range(1, len(num_cols) + 1),
            "Numerical Columns": num_cols
        })
        render_compact_table(num_df)

    if st.session_state.show_cat:
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        cat_df = pd.DataFrame({
            "Index": range(1, len(cat_cols) + 1),
            "Categorical Columns": cat_cols
        })
        render_compact_table(cat_df)
