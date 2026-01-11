import streamlit as st
import pandas as pd
import numpy as np


# ===============================
# Purple + White Styling (ONLY UI)
# ===============================
def inject_purple_white_css():
    st.markdown("""
    <style>
    /* Page headers */
    h1, h2, h3 {
        color: #5b5fe8;
        font-weight: 700;
    }

    /* Dataframe header */
    thead tr th {
        background-color: #5b5fe8 !important;
        color: white !important;
        font-weight: 700;
        text-align: center;
    }

    /* Dataframe cells */
    tbody tr td {
        background-color: #f7f8ff;
        color: #1f2937;
        text-align: center;
    }

    /* Remove index column */
    .row_heading.level0 {
        display: none;
    }
    .blank {
        display: none;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #5b5fe8;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
    }

    div.stButton > button:hover {
        background-color: #4a4fd8;
        color: white;
    }

    /* Alerts */
    .stAlert {
        border-left: 6px solid #5b5fe8;
    }
    </style>
    """, unsafe_allow_html=True)


# ===============================
# Preprocessing Page
# ===============================
def preprocessing_page():
    inject_purple_white_css()

    st.header("🛠️ Preprocessing Stage")

    if "data" not in st.session_state:
        st.warning("⚠️ Upload dataset first.")
        return

    df = st.session_state["data"]

    # =========================
    # Column-wise Data Quality Summary
    # =========================
    st.subheader("📌 Column-wise Data Quality Summary")

    summary = []

    for col in df.columns:
        col_data = df[col]
        missing_count = col_data.isnull().sum()

        # Outlier detection using IQR (numeric only)
        outliers = 0
        if pd.api.types.is_numeric_dtype(col_data):
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)].count()

        summary.append({
            "Column Name": col,
            "Data Type": col_data.dtype,
            "Missing Values": missing_count,
            "Outliers (IQR)": outliers,
            "Unique Values": col_data.nunique()
        })

    summary_df = pd.DataFrame(summary)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # =========================
    # Duplicate Records
    # =========================
    st.subheader("🧬 Duplicate Records")

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        st.warning(f"⚠️ Duplicate Rows Found: {duplicate_count}")
        st.dataframe(df[df.duplicated()], use_container_width=True, hide_index=True)
    else:
        st.success("✅ No duplicate rows found")
