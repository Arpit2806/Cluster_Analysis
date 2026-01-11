import streamlit as st

def eda_page():
    st.header("📊 Exploratory Data Analysis")

    if "data" not in st.session_state:
        st.warning("⚠️ Upload dataset first.")
        return

    st.write("EDA visuals will be added here.")

