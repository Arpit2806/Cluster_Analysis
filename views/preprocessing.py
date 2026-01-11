import streamlit as st
import pandas as pd
import numpy as np

def preprocessing_page():
    st.header("🛠️ Preprocessing Stage")

    if "data" not in st.session_state:
        st.warning("⚠️ Upload dataset first.")
        return

    df = st.session_state["data"]

    # =========================
    # Dataset Overview
    # =========================
    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Total Missing Values", df.isnull().sum().sum())

    # =========================
    # Dataset Preview
    # =========================
    st.subheader("👀 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # =========================
    # Column-wise Data Quality Summary
    # =========================
    st.subheader("📌 Column-wise Data Quality Summary")

    summary = []

    for col in df.columns:
        col_data = df[col]
        missing_count = col_data.isnull().sum()
        missing_pct = (missing_count / len(df)) * 100

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
            "Missing %": round(missing_pct, 2),
            "Outliers (IQR)": outliers,
            "Unique Values": col_data.nunique()
        })

    summary_df = pd.DataFrame(summary)
    st.dataframe(summary_df, use_container_width=True)

    # =========================
    # Duplicate Records
    # =========================
    st.subheader("🧬 Duplicate Records")

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        st.warning(f"⚠️ Duplicate Rows Found: {duplicate_count}")
        st.dataframe(df[df.duplicated()], use_container_width=True)
    else:
        st.success("✅ No duplicate rows found")

    # =========================
    # Missing Value Details
    # =========================
    st.subheader("❓ Missing Values per Column")
    st.dataframe(df.isnull().sum().reset_index().rename(
        columns={"index": "Column", 0: "Missing Count"}
    ), use_container_width=True)

    # =========================
    # Handle Missing Values
    # =========================
    if st.button("⚙️ Handle Missing Values"):
        df_processed = df.copy()

        for col in df_processed.columns:
            if df_processed[col].dtype == "object":
                df_processed[col].fillna("Unknown", inplace=True)
            else:
                df_processed[col].fillna(df_processed[col].mean(), inplace=True)

        st.session_state["processed_data"] = df_processed
        st.success("✅ Missing values handled and saved as processed data")
