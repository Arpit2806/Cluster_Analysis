import streamlit as st

def preprocessing_page():
    st.header("🛠️ Preprocessing Stage")

    if "data" not in st.session_state:
        st.warning("⚠️ Upload dataset first.")
        return

    df = st.session_state["data"]

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    if st.button("Handle Missing Values"):
        df_processed = df.copy()

        for col in df_processed.columns:
            if df_processed[col].dtype == "object":
                df_processed[col].fillna("Unknown", inplace=True)
            else:
                df_processed[col].fillna(df_processed[col].mean(), inplace=True)

        st.session_state["processed_data"] = df_processed
        st.success("✅ Missing values handled")
