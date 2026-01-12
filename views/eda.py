import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder


def eda_page():

    st.header("📊 Exploratory Data Analysis (EDA)")

    if "data" not in st.session_state:
        st.warning("⚠️ Upload dataset first.")
        return

    # ==================================================
    # WORKING COPY
    # ==================================================
    df = st.session_state["data"].copy()

    # ==================================================
    # 0. GLOBAL DROP COLUMNS
    # ==================================================
    st.subheader("🗑 Drop Irrelevant Columns (EDA Scope)")

    global_drop_cols = st.multiselect(
        "Select columns to remove from all analysis (e.g. ID, Index):",
        df.columns.tolist()
    )

    if global_drop_cols:
        df.drop(columns=global_drop_cols, inplace=True)

    st.divider()

    # ==================================================
    # 1. TARGET VARIABLE SELECTION
    # ==================================================
    st.subheader("🎯 Target Variable Selection")

    if "target_var" not in st.session_state:
        st.session_state.target_var = None

    target = st.selectbox(
        "Select target variable:",
        ["-- Select --"] + df.columns.tolist()
    )

    if target != "-- Select --":
        st.session_state.target_var = target
        st.success(f"Target variable set to: {target}")
    else:
        st.session_state.target_var = None

    st.divider()

    # ==================================================
    # 2. UNIVARIATE ANALYSIS
    # ==================================================
    st.subheader("📊 Univariate Analysis")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # ---------- Numerical ----------
    st.markdown("### 🔢 Numerical Features")

    for i in range(0, len(num_cols), 3):
        cols = st.columns(3)
        for j, col in enumerate(num_cols[i:i + 3]):
            with cols[j]:
                fig, ax = plt.subplots(figsize=(3.5, 2.4))

                # 🔧 FIX: Binary / discrete handled separately
                if df[col].nunique() <= 5:
                    counts = df[col].value_counts().sort_index()
                    ax.bar(counts.index.astype(str), counts.values)
                else:
                    ax.hist(df[col].dropna(), bins=20)

                ax.set_title(col, fontsize=9)
                ax.set_xlabel("")
                ax.set_ylabel("")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    st.divider()

    # ---------- Categorical ----------
    st.markdown("### 🏷 Categorical Features")

    for i in range(0, len(cat_cols), 3):
        cols = st.columns(3)
        for j, col in enumerate(cat_cols[i:i + 3]):
            with cols[j]:
                counts = df[col].value_counts().head(6)
                fig, ax = plt.subplots(figsize=(3.5, 2.4))
                ax.barh(counts.index.astype(str), counts.values)
                ax.set_title(col, fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    st.divider()

    # ==================================================
    # 3. DROP COLUMNS (CORRELATION ONLY)
    # ==================================================
    st.subheader("🗑 Exclude Columns from Correlation Only")

    if st.session_state.target_var:
        corr_drop_cols = st.multiselect(
            "Select columns to exclude (correlation only):",
            [c for c in df.columns if c != st.session_state.target_var]
        )
    else:
        corr_drop_cols = []

    corr_df = df.drop(columns=corr_drop_cols, errors="ignore")

    st.divider()

    # ==================================================
    # 4. PEARSON CORRELATION (NUMERICAL)
    # ==================================================
    st.subheader("🔥 Pearson Correlation with Target")

    target = st.session_state.target_var

    if target and target in corr_df.columns:

        num_df = corr_df.select_dtypes(include=np.number)
        num_df = num_df.loc[:, num_df.nunique() > 1]

        # 🔧 FIX: Guard against missing target
        if target not in num_df.columns:
            st.warning("Target variable not available for numerical correlation.")
        else:
            corr_vals = (
                num_df.corr()[target]
                .drop(target, errors="ignore")
                .sort_values(ascending=False)
                .to_frame(name=target)
            )

            fig_height = max(4, len(corr_vals) * 0.3)
            fig, ax = plt.subplots(figsize=(3.2, fig_height))

            sns.heatmap(
                corr_vals,
                annot=True,
                cmap="coolwarm",
                center=0,
                fmt=".2f",
                annot_kws={"size": 8},
                ax=ax
            )

            plt.tight_layout()
            _, center_col, _ = st.columns([1, 2, 1])
            with center_col:
                st.pyplot(fig)

            plt.close(fig)
    else:
        st.info("Select a valid target variable to view correlation.")

    st.divider()

    # ==================================================
    # 5. SPEARMAN CORRELATION (CATEGORICAL)
    # ==================================================
    st.subheader("📐 Spearman Correlation with Target (Categorical)")

    if target and target in corr_df.columns and pd.api.types.is_numeric_dtype(corr_df[target]):

        cat_corr_cols = corr_df.select_dtypes(exclude=np.number).columns.tolist()

        if cat_corr_cols:
            enc = OrdinalEncoder()
            encoded = pd.DataFrame(
                enc.fit_transform(corr_df[cat_corr_cols]),
                columns=cat_corr_cols
            )

            encoded = encoded.loc[:, encoded.nunique() > 1]

            spearman_vals = (
                encoded.apply(
                    lambda x: x.corr(corr_df[target], method="spearman")
                )
                .dropna()
                .sort_values(ascending=False)
                .to_frame(name=target)
            )

            fig_height = max(3, len(spearman_vals) * 0.35)
            fig, ax = plt.subplots(figsize=(3.2, fig_height))

            sns.heatmap(
                spearman_vals,
                annot=True,
                cmap="coolwarm",
                center=0,
                fmt=".2f",
                annot_kws={"size": 8},
                ax=ax
            )

            plt.tight_layout()
            _, center_col, _ = st.columns([1, 2, 1])
            with center_col:
                st.pyplot(fig)

            plt.close(fig)
        else:
            st.info("No categorical columns available.")
    else:
        st.info("Spearman correlation requires a numerical target.")

    st.divider()

    # ==================================================
    # 6. FINAL DATAFRAME PREVIEW (FOR NEXT PAGES)
    # ==================================================
    st.subheader("📄 Final Dataset After EDA")

    df_temp = corr_df.copy()
    st.session_state["df_temp"] = df_temp

    st.write("This dataset will be used for modeling and predictions.")
    st.dataframe(df_temp.head(), use_container_width=True)

    st.divider()

    # ==================================================
    # 7. INSIGHTS
    # ==================================================
    st.subheader("🧠 Key Insights")

    st.markdown("""
    - Binary variables are visualized correctly  
    - Target-aware correlation is safely computed  
    - Irrelevant columns are excluded cleanly  
    - Final dataset (`df_temp`) is ready for modeling  
    """)
