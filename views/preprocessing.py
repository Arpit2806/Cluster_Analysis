import streamlit as st
import pandas as pd
import numpy as np

# ===============================
# Preprocessing Page
# ===============================
def preprocessing_page():
    inject_table_css()

    st.header("🛠️ Preprocessing Stage")

    if "data" not in st.session_state:
        st.warning("⚠️ Upload dataset first.")
        return

    df = st.session_state["data"]

    # ===============================
    # Dataset Preview
    # ===============================
    st.subheader("🔍 Dataset Preview")
    render_compact_table(df.head())

    st.divider()

    # ===============================
    # Dataset Health Overview
    # ===============================
    st.subheader("📊 Dataset Health Overview")

    overview_df = pd.DataFrame({
        "Metric": [
            "Total Rows",
            "Total Columns",
            "Total Missing Values",
            "Duplicate Rows"
        ],
        "Value": [
            df.shape[0],
            df.shape[1],
            df.isnull().sum().sum(),
            df.duplicated().sum()
        ]
    })

    render_compact_table(overview_df)

    st.divider()

    # ===============================
    # Column-wise Data Quality Summary
    # ===============================
    st.subheader("📌 Column-wise Data Quality Summary")

    summary = []

    for col in df.columns:
        col_data = df[col]
        missing_count = col_data.isnull().sum()
        missing_pct = round((missing_count / len(df)) * 100, 2)

        outliers = 0
        if pd.api.types.is_numeric_dtype(col_data):
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = col_data[(col_data < lower) | (col_data > upper)].count()

        summary.append({
            "Column Name": col,
            "Data Type": col_data.dtype,
            "Missing Values": missing_count,
            "Missing %": missing_pct,
            "Outliers (IQR)": outliers,
            "Unique Values": col_data.nunique()
        })

    summary_df = pd.DataFrame(summary)
    render_compact_table(summary_df)

    st.divider()

    # ===============================
    # Duplicate Records
    # ===============================
    st.subheader("🧬 Duplicate Records")

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        dup_info_df = pd.DataFrame({
            "Metric": ["Duplicate Rows Found"],
            "Value": [duplicate_count]
        })
        render_compact_table(dup_info_df)

        st.markdown("**Duplicate Rows Preview**")
        render_compact_table(df[df.duplicated()])
    else:
        render_compact_table(pd.DataFrame({
            "Status": ["No duplicate rows found"]
        }))

    st.divider()

    # ===============================
    # Missing Values Table
    # ===============================
    st.subheader("❓ Missing Values per Column")

    missing_df = (
        df.isnull()
        .sum()
        .reset_index()
        .rename(columns={"index": "Column Name", 0: "Missing Count"})
    )

    render_compact_table(missing_df)

    st.divider()

    # ===============================
    # Handle Missing Values
    # ===============================
    if st.button("⚙️ Handle Missing Values"):
        df_processed = df.copy()

        for col in df_processed.columns:
            if df_processed[col].dtype == "object":
                df_processed[col].fillna("Unknown", inplace=True)
            else:
                df_processed[col].fillna(df_processed[col].mean(), inplace=True)

        st.session_state["processed_data"] = df_processed
        st.success("✅ Missing values handled and stored as processed data")
