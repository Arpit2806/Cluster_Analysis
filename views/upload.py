import streamlit as st
import pandas as pd
import numpy as np

def upload_page():
    st.header("📂 Upload Dataset")
    st.write("Upload your customer dataset (CSV format).")

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["data"] = df

        st.success("✅ Dataset uploaded successfully!")

        # ===============================
        # 1. PREVIEW OF DATA
        # ===============================
        st.subheader("🔍 Preview of Data")
        st.dataframe(df.head(), use_container_width=True)

        # ===============================
        # 2. DATASET OVERVIEW
        # ===============================
        st.subheader("📊 Dataset Overview")

        total_rows = df.shape[0]
        total_columns = df.shape[1]

        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        overview_df = pd.DataFrame({
            "Metric": [
                "Total Rows",
                "Total Columns",
                "Numerical Columns",
                "Categorical Columns"
            ],
            "Value": [
                total_rows,
                total_columns,
                len(numerical_cols),
                len(categorical_cols)
            ]
        })

        st.table(overview_df)

        # ===============================
        # 3. COLUMN LIST (INFORMATIONAL)
        # ===============================
        with st.expander("📋 View Column Names"):
            col_df = pd.DataFrame({
                "Column Name": df.columns
            })
            st.dataframe(col_df, use_container_width=True)
