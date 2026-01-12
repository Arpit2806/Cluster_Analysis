import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder


def eda_page():

    st.header("📊 Exploratory Data Analysis (EDA)")

    if "data" not in st.session_state:
        st.warning("⚠️ Upload dataset first.")
        return

    # --------------------------------------------------
    # Work on a COPY (safe)
    # --------------------------------------------------
    df = st.session_state["data"].copy()

    # ==================================================
    # 0. DROP COLUMNS (USER CONTROL)
    # ==================================================
    st.subheader("🗑 Drop Columns (EDA Scope Only)")

    drop_cols = st.multiselect(
        "Select columns to exclude from analysis:",
        df.columns.tolist()
    )

    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

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

    st.divider()

    # ==================================================
    # 2. UNIVARIATE ANALYSIS (ALL AT ONCE)
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
                fig, ax = plt.subplots(figsize=(3.5, 2.3))
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
                fig, ax = plt.subplots(figsize=(3.5, 2.3))
                ax.barh(counts.index, counts.values)
                ax.set_title(col, fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    st.divider()

    # ==================================================
    # 3. CORRELATION (NUMERICAL vs TARGET)
    # ==================================================
    st.subheader("🔥 Correlation with Target (Numerical)")

    if st.session_state.target_var and pd.api.types.is_numeric_dtype(df[target]):

        corr = (
            df.select_dtypes(include=np.number)
            .corr()[target]
            .drop(target)
            .sort_values(key=abs, ascending=False)
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(corr.index, corr.values)
        ax.set_title("Pearson Correlation with Target", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Numerical correlation available only for numerical target.")

    st.divider()

    # ==================================================
    # 4. SPEARMAN CORRELATION (CATEGORICAL vs TARGET)
    # ==================================================
    st.subheader("📐 Spearman Correlation (Categorical)")

    if st.session_state.target_var and pd.api.types.is_numeric_dtype(df[target]) and cat_cols:

        enc = OrdinalEncoder()
        cat_encoded = pd.DataFrame(
            enc.fit_transform(df[cat_cols]),
            columns=cat_cols
        )

        spearman = cat_encoded.apply(
            lambda x: x.corr(df[target], method="spearman")
        ).sort_values(key=abs, ascending=False)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(spearman.index, spearman.values)
        ax.set_title("Spearman Correlation with Target", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Spearman correlation requires numerical target and categorical features.")

    st.divider()

    # ==================================================
    # 5. SHAP (LIGHTWEIGHT, SAFE)
    # ==================================================
    st.subheader("🧠 SHAP Feature Importance (EDA-level)")

    if st.session_state.target_var and pd.api.types.is_numeric_dtype(df[target]):

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

            fig = plt.figure()
            shap.plots.bar(shap_values, max_display=10, show=False)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Not enough numerical features for SHAP.")
    else:
        st.info("SHAP available only for numerical target.")

    st.divider()

    # ==================================================
    # 6. INSIGHTS
    # ==================================================
    st.subheader("🧠 Key Insights")

    st.markdown("""
    - Numerical and categorical distributions vary significantly  
    - Several features show measurable association with the target  
    - Correlation and Spearman analysis help shortlist predictors  
    - SHAP highlights feature importance from a model perspective  
    """)
