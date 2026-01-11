import streamlit as st
import pandas as pd
import numpy as np

PURPLE = "#5b5fe8"

def styled_table(df):
    return (
        df.style
        .set_properties(**{
            "text-align": "center",
            "white-space": "nowrap"
        })
        .set_table_styles([
            {
                # 🔥 ONLY COLUMN HEADERS
                "selector": "th.col_heading",
                "props": [
                    ("background-color", PURPLE),
                    ("color", "white"),              # ✅ WHITE TEXT
                    ("font-weight", "700"),
                    ("text-align", "center"),
                    ("border", "1px solid #ffffff20")
                ]
            },
            {
                # Optional: body cell borders
                "selector": "td",
                "props": [
                    ("border", "1px solid #e5e7eb")
                ]
            }
        ])
        .hide(axis="index")   # ❌ removes 0,1,2,3
    )

def upload_page():
    st.header("📂 Upload Dataset")
    st.write("Upload your customer dataset (CSV format).")

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["data"] = df

        st.success("✅ Dataset uploaded successfully!")

        # ==================================================
        # PREVIEW
        # ==================================================
        st.subheader("🔍 Preview of Data")
        st.dataframe(df.head(), use_container_width=True)

        st.divider()

        # ==================================================
        # DATASET OVERVIEW
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

        st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
        st.table(styled_table(overview_df))
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # ==================================================
        # COLUMN DETAILS (TOGGLE BUTTONS)
        # ==================================================
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

        # ==================================================
        # NUMERICAL COLUMNS TABLE
        # ==================================================
        if st.session_state.show_num:
            num_df = pd.DataFrame({
                "Numerical Columns": df.select_dtypes(include=np.number).columns
            })

            st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
            st.table(styled_table(num_df))
            st.markdown("</div>", unsafe_allow_html=True)

        # ==================================================
        # CATEGORICAL COLUMNS TABLE
        # ==================================================
        if st.session_state.show_cat:
            cat_df = pd.DataFrame({
                "Categorical Columns": df.select_dtypes(exclude=np.number).columns
            })

            st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
            st.table(styled_table(cat_df))
            st.markdown("</div>", unsafe_allow_html=True)
