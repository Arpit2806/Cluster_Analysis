import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
        options=["-- Select --"] + df.columns.tolist()
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

    if num_cols:
        rows = [num_cols[i:i + 3] for i in range(0, len(num_cols), 3)]

        for row in rows:
            cols = st.columns(3)
            for i, col in enumerate(row):
                with cols[i]:
                    fig, ax = plt.subplots(figsize=(3.5, 2.5))
                    sns.histplot(df[col].dropna(), kde=True, ax=ax)
                    ax.set_title(col, fontsize=10)
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    st.pyplot(fig)
    else:
        st.info("No numerical columns available.")

    st.divider()

    # ---------- Categorical ----------
    st.markdown("### 🏷 Categorical Features")

    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if cat_cols:
        rows = [cat_cols[i:i + 3] for i in range(0, len(cat_cols), 3)]

        for row in rows:
            cols = st.columns(3)
            for i, col in enumerate(row):
                with cols[i]:
                    counts = df[col].value_counts().head(10)
                    fig, ax = plt.subplots(figsize=(3.5, 2.5))
                    sns.barplot(x=counts.values, y=counts.index, ax=ax)
                    ax.set_title(col, fontsize=10)
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    st.pyplot(fig)
    else:
        st.info("No categorical columns available.")

    st.divider()

    # ==================================================
    # 3. BIVARIATE ANALYSIS (ALL vs TARGET)
    # ==================================================
    st.subheader("🔗 Bivariate Analysis")

    if st.session_state.target_var is None:
        st.warning("Please select a target variable.")
    else:
        target = st.session_state.target_var
        target_is_num = pd.api.types.is_numeric_dtype(df[target])

        features = [c for c in df.columns if c != target]

        rows = [features[i:i + 3] for i in range(0, len(features), 3)]

        for row in rows:
            cols = st.columns(3)
            for i, feature in enumerate(row):
                with cols[i]:
                    fig, ax = plt.subplots(figsize=(3.5, 2.5))

                    if target_is_num:
                        if pd.api.types.is_numeric_dtype(df[feature]):
                            sns.scatterplot(
                                x=df[feature], y=df[target], ax=ax, s=10
                            )
                        else:
                            sns.boxplot(
                                x=df[feature], y=df[target], ax=ax
                            )
                    else:
                        if pd.api.types.is_numeric_dtype(df[feature]):
                            sns.boxplot(
                                x=df[target], y=df[feature], ax=ax
                            )
                        else:
                            ct = pd.crosstab(df[feature], df[target])
                            ct.plot(kind="bar", stacked=True, ax=ax, legend=False)

                    ax.set_title(feature, fontsize=10)
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    st.pyplot(fig)

    st.divider()

    # ==================================================
    # 4. CLEAN CORRELATION HEATMAP
    # ==================================================
    st.subheader("🔥 Correlation Analysis (Top Relationships)")

    num_df = df.select_dtypes(include=np.number)

    if num_df.shape[1] < 2:
        st.info("Not enough numerical columns.")
    else:
        corr = num_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        top_corr = (
            upper.stack()
            .sort_values(ascending=False)
            .head(12)
        )

        corr_features = list(set(top_corr.index.get_level_values(0)) |
                             set(top_corr.index.get_level_values(1)))

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            num_df[corr_features].corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            square=True,
            ax=ax
        )
        st.pyplot(fig)

    st.divider()

    # ==================================================
    # 5. INSIGHTS SUMMARY
    # ==================================================
    st.subheader("🧠 Key Insights from EDA")

    st.markdown("""
    - Numerical features show varied distributions and skewness  
    - Categorical variables highlight dominant customer segments  
    - Several features exhibit strong relationships with the target  
    - Highly correlated numerical features indicate possible redundancy  
    """)
