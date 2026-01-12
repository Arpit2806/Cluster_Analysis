import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
import shap


def eda_page():

    st.header("📊 Exploratory Data Analysis (EDA)")

    if "data" not in st.session_state:
        st.warning("⚠️ Upload dataset first.")
        return

    # ==================================================
    # BASE COPY
    # ==================================================
    df = st.session_state["data"].copy()

    # ==================================================
    # 0. GLOBAL DROP (IDs, INDEX ETC.)
    # ==================================================
    st.subheader("🗑 Drop Irrelevant Columns (Global)")

    global_drop = st.multiselect(
        "Remove from all analysis (IDs, Index, LoanID):",
        df.columns.tolist()
    )

    if global_drop:
        df.drop(columns=global_drop, inplace=True)

    st.divider()

    # ==================================================
    # 1. DROP CATEGORICAL COLUMNS
    # ==================================================
    st.subheader("🗑 Drop Categorical Columns")

    cat_cols_all = df.select_dtypes(exclude=np.number).columns.tolist()

    cat_drop = st.multiselect(
        "Remove categorical columns from analysis:",
        cat_cols_all
    )

    if cat_drop:
        df.drop(columns=cat_drop, inplace=True)

    st.divider()

    # ==================================================
    # 2. TARGET VARIABLE
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

    st.divider()

    # ==================================================
    # 3. UNIVARIATE ANALYSIS
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
                fig, ax = plt.subplots(figsize=(3.4, 2.3))
                ax.hist(df[col].dropna(), bins=20)
                ax.set_title(col, fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    st.divider()

    # ---------- Categorical ----------
    if cat_cols:
        st.markdown("### 🏷 Categorical Features")

        for i in range(0, len(cat_cols), 3):
            cols = st.columns(3)
            for j, col in enumerate(cat_cols[i:i + 3]):
                with cols[j]:
                    counts = df[col].value_counts().head(6)
                    fig, ax = plt.subplots(figsize=(3.4, 2.3))
                    ax.barh(counts.index, counts.values)
                    ax.set_title(col, fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
    else:
        st.info("No categorical columns available after dropping.")

    st.divider()

    # ==================================================
    # 4. CORRELATION-ONLY DROP
    # ==================================================
    st.subheader("🗑 Exclude Columns (Correlation Only)")

    if st.session_state.target_var:
        corr_drop = st.multiselect(
            "Exclude columns only for correlation:",
            [c for c in df.columns if c != target]
        )
    else:
        corr_drop = []

    corr_df = df.drop(columns=corr_drop, errors="ignore")

    st.divider()

    # ==================================================
    # 5. PEARSON CORRELATION
    # ==================================================
    st.subheader("🔥 Pearson Correlation with Target")

    if target and pd.api.types.is_numeric_dtype(corr_df[target]):

        num_df = corr_df.select_dtypes(include=np.number)
        num_df = num_df.loc[:, num_df.nunique() > 1]

        corr_vals = (
            num_df.corr()[target]
            .drop(target, errors="ignore")
            .sort_values(ascending=False)
            .to_frame(name=target)
        )

        fig_h = max(4, len(corr_vals) * 0.3)
        fig, ax = plt.subplots(figsize=(3.2, fig_h))

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
        _, c, _ = st.columns([1, 2, 1])
        with c:
            st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Pearson correlation requires a numerical target.")

    st.divider()

    # ==================================================
    # 6. SPEARMAN CORRELATION (CATEGORICAL)
    # ==================================================
    st.subheader("📐 Spearman Correlation with Target (Categorical)")

    if target and pd.api.types.is_numeric_dtype(corr_df[target]):

        cat_corr_cols = corr_df.select_dtypes(exclude=np.number).columns.tolist()

        if cat_corr_cols:
            enc = OrdinalEncoder()
            encoded = pd.DataFrame(
                enc.fit_transform(corr_df[cat_corr_cols]),
                columns=cat_corr_cols
            )

            encoded = encoded.loc[:, encoded.nunique() > 1]

            spearman_vals = (
                encoded.apply(lambda x: x.corr(corr_df[target], method="spearman"))
                .dropna()
                .sort_values(ascending=False)
                .to_frame(name=target)
            )

            fig_h = max(3, len(spearman_vals) * 0.35)
            fig, ax = plt.subplots(figsize=(3.2, fig_h))

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
            _, c, _ = st.columns([1, 2, 1])
            with c:
                st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No categorical columns available.")
    else:
        st.info("Spearman correlation requires a numerical target.")

    st.divider()

    # ==================================================
    # 7. SHAP (AUTO-UPDATED)
    # ==================================================
    st.subheader("🧠 SHAP Feature Importance")

    if target and pd.api.types.is_numeric_dtype(df[target]):

        X = df.select_dtypes(include=np.number).drop(columns=[target], errors="ignore")
        y = df[target]

        if X.shape[1] >= 2:
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=6,
                random_state=42
            )
            model.fit(X, y)

            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)

            max_feats = min(10, X.shape[1])

            fig = plt.figure(figsize=(4, 3))
            shap.plots.bar(shap_values, max_display=max_feats, show=False)

            _, c, _ = st.columns([1, 2, 1])
            with c:
                st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Not enough numerical features for SHAP.")
    else:
        st.info("SHAP available only for numerical target.")

    st.divider()

    # ==================================================
    # 8. INSIGHTS
    # ==================================================
    st.subheader("🧠 Key Insights")

    st.markdown("""
    - Global and categorical drops immediately affect all downstream analysis  
    - Correlation-only drop allows flexible what-if analysis  
    - Pearson and Spearman capture numerical and categorical relationships  
    - SHAP dynamically adapts to selected features  
    """)
