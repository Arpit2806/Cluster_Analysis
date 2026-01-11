import streamlit as st

def preprocessing_page():
    st.header("🛠️ Preprocessing Stage")

    if "data" not in st.session_state:
        st.warning("⚠️ Please upload dataset first.")
        return

    df = st.session_state["data"]
    st.dataframe(df.head())
