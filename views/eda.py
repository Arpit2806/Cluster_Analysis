import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor


def eda_page():

    st.header("📊 Exploratory Data Analysis (EDA)")

    if "data" not in st.session_state:
        st.warning("⚠️ Upload dataset first.")
        return

    df = st.session_state["data"].copy()

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
    # 2. UNIVARIATE ANALYSIS (UNCHANGED)
    # ==================================================
    st.subheader("📊 Univariate Analysis")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    st.markdown("### 🔢 Numerical Features")
    for i in range(0, len(num_cols), 3):
        cols = st.columns(3)
        for j, col in enumerate(num_cols[i:i + 3]):
            with cols[j]:
                fig, ax = plt.subplots(figsize=(3.5, 2.3))
                ax.hist(df[col].dropna(), bins=20)
                ax.set_title(col, fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    st.divider()

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
    # 3. DROP COLUMNS (FOR CORRELATION ONLY)
    # ==================================================
    st.subheader("🗑 Exclude Columns from Correlation")

    if st.session_state.target_var:
        corr_drop_cols = st.multiselect(
            "Select columns to exclude (correlation only):",
            [c for c in df.columns if c != target]
        )
    else:
        corr_drop_cols = []

    corr_df = df.drop(columns=corr_drop_cols, errors="ignore")

    st.divider()

    # ==================================================
    # 4. CORRELATION (NUMERICAL – HEATMAP STYLE)
    # ==================================================
    st.subheader("🔥 Correlation with Target (Numerical)")

    if st.session_state.target_var and pd.api.types.is_numeric_dtype(corr_df[target]):

        num_df = corr_df.select_dtypes(include=np.number)

        corr_vals = (
            num_df.corr()[[target]]
            .sort_values(by=target, ascending=False)
        )

        fig, ax = plt.subplots(figsize=(4, len(corr_vals) * 0.35))
        sns.heatmap(
            corr_vals,
            annot=True,
            cmap="coolwarm",
            center=0,
            cbar=True,
            fmt=".2f",
            ax=ax
        )
        ax.set_title("Pearson Correlation with Target")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Numerical correlation requires a numerical target.")

    st.divider()

    # ==================================================
    # 5. SPEARMAN CORRELATION (CATEGORICAL – HEATMAP STYLE)
    # ==================================================
    st.subheader("📐 Spearman Correlation with Target (Categorical)")

    if st.session_state.target_var and pd.api.types.is_numeric_dtype(corr_df[target]):

        cat_cols_corr = corr_df.select_dtypes(exclude=np.number).columns.tolist()

        if cat_cols_corr:
            enc = OrdinalEncoder()
            encoded = pd.DataFrame(
                enc.fit_transform(corr_df[cat_cols_corr]),
                columns=cat_cols_corr
            )

            spearman_vals = encoded.apply(
                lambda x: x.corr(corr_df[target], method="spearman")
            ).to_frame(name=target).sort_values(by=target, ascending=False)

            fig, ax = plt.subplots(figsize=(4, len(spearman_vals) * 0.35))
            sns.heatmap(
                spearman_vals,
                annot=True,
                cmap="coolwarm",
                center=0,
                cbar=True,
                fmt=".2f",
                ax=ax
            )
            ax.set_title("Spearman Correlation with Target")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No categorical columns available.")
    else:
        st.info("Spearman correlation requires a numerical target.")

    st.divider()

    # ==================================================
    # 6. SHAP (SAFE)
    # ==================================================
    st.subheader("🧠 SHAP Feature Importance")

    try:
        import shap

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

    except ModuleNotFoundError:
        st.warning(
            "SHAP is not installed in this environment. "
            "To enable SHAP, add `shap` to requirements.txt."
        )

    st.divider()

    # ==================================================
    # 7. INSIGHTS
    # ==================================================
    st.subheader("🧠 Key Insights")

    st.markdown("""
    - Correlation analysis focuses strictly on the target variable  
    - Both numerical (Pearson) and categorical (Spearman) relationships are examined  
    - Column exclusion allows flexible what-if analysis  
    - SHAP provides model-based interpretability when available  
    """)
