import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def eda_page():

    st.header("📊 Exploratory Data Analysis (EDA)")

    if "data" not in st.session_state:
        st.warning("⚠️ Upload dataset first.")
        return

    df = st.session_state["data"]

    # ==================================================
    # 1. TARGET VARIABLE SELECTION
    # ==================================================
    st.subheader("🎯 Target Variable Selection")

    if "target_var" not in st.session_state:
        st.session_state.target_var = None

    target = st.selectbox(
        "Select the target variable:",
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

    # ---------- Numerical ----------
    st.markdown("### 🔢 Numerical Features")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    for i in range(0, len(num_cols), 3):
        cols = st.columns(3)
        for j, col in enumerate(num_cols[i:i+3]):
            with cols[j]:
                fig, ax = plt.subplots(figsize=(3.5, 2.5))
                ax.hist(df[col].dropna(), bins=20)
                ax.set_title(col, fontsize=9)
                ax.set_xlabel("")
                ax.set_ylabel("")
                st.pyplot(fig)
                plt.close(fig)

    st.divider()

    # ---------- Categorical ----------
    st.markdown("### 🏷 Categorical Features")
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    for i in range(0, len(cat_cols), 3):
        cols = st.columns(3)
        for j, col in enumerate(cat_cols[i:i+3]):
            with cols[j]:
                counts = df[col].value_counts().head(6)
                fig, ax = plt.subplots(figsize=(3.5, 2.5))
                ax.barh(counts.index, counts.values)
                ax.set_title(col, fontsize=9)
                st.pyplot(fig)
                plt.close(fig)

    st.divider()

    # ==================================================
    # 3. BIVARIATE ANALYSIS (ALL vs TARGET)
    # ==================================================
    st.subheader("🔗 Bivariate Analysis")

    if st.session_state.target_var is None:
        st.warning("Select target variable first.")
    else:
        target = st.session_state.target_var
        target_is_num = pd.api.types.is_numeric_dtype(df[target])

        features = [c for c in df.columns if c != target]

        for i in range(0, len(features), 3):
            cols = st.columns(3)
            for j, feat in enumerate(features[i:i+3]):
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(3.5, 2.5))

                    if target_is_num:
                        if pd.api.types.is_numeric_dtype(df[feat]):
                            ax.scatter(df[feat], df[target], s=8)
                        else:
                            top = df[feat].value_counts().head(5).index
                            plot_df = df[df[feat].isin(top)]
                            plot_df.boxplot(
                                column=target,
                                by=feat,
                                ax=ax
                            )
                            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, fontsize=7)
                    else:
                        if pd.api.types.is_numeric_dtype(df[feat]):
                            df.boxplot(
                                column=feat,
                                by=target,
                                ax=ax
                            )
                        else:
                            top = df[feat].value_counts().head(5).index
                            plot_df = df[df[feat].isin(top)]
                            pd.crosstab(plot_df[feat], plot_df[target]).plot(
                                kind="bar",
                                stacked=True,
                                ax=ax,
                                legend=False
                            )

                    ax.set_title(feat, fontsize=9)
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    st.pyplot(fig)
                    plt.close(fig)

    st.divider()

    # ==================================================
    # 4. CORRELATION WITH TARGET ONLY
    # ==================================================
    st.subheader("🔥 Correlation with Target Variable")

    if st.session_state.target_var and pd.api.types.is_numeric_dtype(df[st.session_state.target_var]):

        corr = (
            df.select_dtypes(include=np.number)
            .corr()[st.session_state.target_var]
            .drop(st.session_state.target_var)
            .sort_values(key=abs, ascending=False)
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(corr.index, corr.values)
        ax.set_title("Correlation with Target")
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("Correlation available only for numerical target variables.")

    st.divider()

    # ==================================================
    # 5. INSIGHTS
    # ==================================================
    st.subheader("🧠 Key Insights from EDA")

    st.markdown("""
    - Distributions vary significantly across numerical features  
    - Categorical variables show dominant customer segments  
    - Several features show visible association with the target  
    - Target-based correlation helps shortlist important predictors  
    """)
