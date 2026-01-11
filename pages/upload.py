import streamlit as st
import pandas as pd

def upload_page():
    st.header("📂 Upload Dataset")
    st.write("Upload your customer dataset in CSV format.")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Store dataset in session state
        st.session_state["data"] = df

        st.success("✅ Dataset uploaded successfully!")

        st.subheader("Preview of Dataset")
        st.dataframe(df.head())
