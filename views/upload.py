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

        # ==================================================
        # 1. PREVIEW OF DATA
        # ==================================================
        st.subheader("🔍 Preview of Data")
        st.dataframe(df.head(), use_container_width=True)

        st.divider()

        # ==================================================
        # 2. DATASET OVERVIEW (CLEAN TABLE)
        # ==================================================
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

        # Remove index & display compact centered table
        st.markdown(
            """
            <div style="display:flex; justify-content:center;">
            """,
            unsafe_allow_html=True
        )

        st.dataframe(
            overview_df,
            hide_index=True,
            use_container_width=False
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # ==================================================
        # 3. DISPLAY COLUMN TYPES
        # ==================================================
        st.subheader("📋 Column Details")

        col1, col2 = st.columns(2)

        # ---- Numerical Columns ----
        with col1:
            if st.button("Display Numerical Columns"):
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                num_df = pd.DataFrame({"Numerical Columns": num_cols})

                st.markdown(
                    "<div style='display:flex; justify-content:center;'>",
                    unsafe_allow_html=True
                )

                st.dataframe(
                    num_df,
                    hide_index=True,
                    use_container_width=False
                )

                st.markdown("</div>", unsafe_allow_html=True)

        # ---- Categorical Columns ----
        with col2:
            if st.button("Display Categorical Columns"):
                cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
                cat_df = pd.DataFrame({"Categorical Columns": cat_cols})

                st.markdown(
                    "<div style='display:flex; justify-content:center;'>",
                    unsafe_allow_html=True
                )

                st.dataframe(
                    cat_df,
                    hide_index=True,
                    use_container_width=False
                )

                st.markdown("</div>", unsafe_allow_html=True)
