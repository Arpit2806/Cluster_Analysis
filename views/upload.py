import streamlit as st
import pandas as pd
import numpy as np

def upload_page():
    st.header("📂 Upload Dataset")
    st.write("Upload your customer dataset (CSV format).")

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["data"] = df

        st.success("✅ Dataset uploaded successfully!")

        # ================= PREVIEW =================
        st.subheader("🔍 Preview of Data")
        st.dataframe(df.head(), use_container_width=True)

        st.divider()

        # ================= OVERVIEW =================
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

        # ================= COLUMN DETAILS =================
        st.subheader("📋 Column Details")

        if "show_num" not in st.session_state:
            st.session_state.show_num = False
        if "show_cat" not in st.session_state:
            st.session_state.show_cat = False

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Display Numerical Columns"):
                st.session_state.show_num = not st.session_state.show_num

        with col2:
            if st.button("Display Categorical Columns"):
                st.session_state.show_cat = not st.session_state.show_cat

        if st.session_state.show_num:
            num_df = pd.DataFrame({
                "Numerical Columns": df.select_dtypes(include=np.number).columns
            })
            render_compact_table(num_df)

        if st.session_state.show_cat:
            cat_df = pd.DataFrame({
                "Categorical Columns": df.select_dtypes(exclude=np.number).columns
            })
            render_compact_table(cat_df)
