import streamlit as st
import pandas as pd

def preprocessing_page():
    st.header("🛠️ Preprocessing Stage")

    # Check if data exists
    if "data" not in st.session_state:
        st.warning("⚠️ Please upload a dataset first.")
        return

    df = st.session_state["data"]

    st.subheader("Dataset Overview")
    st.write("Shape of dataset:", df.shape)
    st.dataframe(df.head())

    st.subheader("Missing Value Analysis")
    missing_values = df.isnull().sum()
    st.dataframe(missing_values)

    if st.button("Handle Missing Values"):
        df_processed = df.copy()

        for col in df_processed.columns:
            if df_processed[col].dtype == "object":
                df_processed[col].fillna("Unknown", inplace=True)
            else:
                df_processed[col].fillna(df_processed[col].mean(), inplace=True)

        st.session_state["processed_data"] = df_processed

        st.success("✅ Missing values handled successfully!")
        st.dataframe(df_processed.head())
