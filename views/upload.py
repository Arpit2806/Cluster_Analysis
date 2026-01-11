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
        # PREVIEW
        # ===============================
        st.subheader("🔍 Preview of Data")
        st.dataframe(df.head(), use_container_width=True)

        st.divider()

        # ===============================
        # DATASET OVERVIEW
        # ===============================
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

        styled_overview = (
            overview_df.style
            .set_properties(**{
                "text-align": "center"
            })
            .set_table_styles([
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#5b5fe8"),
                        ("color", "white"),
                        ("text-align", "center"),
                        ("font-weight", "600")
                    ]
                }
            ])
        )

        st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
        st.dataframe(styled_overview, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # ===============================
        # COLUMN DETAILS (BUTTON LOGIC)
        # ===============================
        st.subheader("📋 Column Details")

        # Initialize session flags
        if "show_num" not in st.session_state:
            st.session_state.show_num = False
        if "show_cat" not in st.session_state:
            st.session_state.show_cat = False

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Display Numerical Columns"):
                st.session_state.show_num = True

        with col2:
            if st.button("Display Categorical Columns"):
                st.session_state.show_cat = True

        # ---- Numerical Columns Table ----
        if st.session_state.show_num:
            num_df = pd.DataFrame({
                "Numerical Columns": df.select_dtypes(include=np.number).columns
            })

            styled_num = (
                num_df.style
                .set_properties(**{"text-align": "center"})
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#5b5fe8"),
                            ("color", "white"),
                            ("text-align", "center"),
                            ("font-weight", "600")
                        ]
                    }
                ])
            )

            st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
            st.dataframe(styled_num, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ---- Categorical Columns Table ----
        if st.session_state.show_cat:
            cat_df = pd.DataFrame({
                "Categorical Columns": df.select_dtypes(exclude=np.number).columns
            })

            styled_cat = (
                cat_df.style
                .set_properties(**{"text-align": "center"})
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#5b5fe8"),
                            ("color", "white"),
                            ("text-align", "center"),
                            ("font-weight", "600")
                        ]
                    }
                ])
            )

            st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
            st.dataframe(styled_cat, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
