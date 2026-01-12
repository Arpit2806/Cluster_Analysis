import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ===============================
# EDA PAGE
# ===============================
def eda_page():

    st.header("📊 Exploratory Data Analysis (EDA)")

    if "data" not in st.session_state:
        st.warning("⚠️ Upload dataset first.")
        return

    df = st.session_state["data"]

    # ==================================================
    # 1. TARGET VARIABLE SELECTION (GLOBAL)
    # ==================================================
    st.subheader("🎯 Target Variable Selection")

    if "target_var" not in st.session_state:
        st.session_state.target_var = None

    target = st.selectbox(
        "Select the target (dependent) variable:",
        options=["-- Select --"] + df.columns.tolist()
    )

    if target != "-- Select --":
        st.session_state.target_var = target
        st.success(f"Target variable set to: {target}")

    st.divider()

    # ==================================================
    # 2. UNIVARIATE ANALYSIS
    # ==================================================
    st.subheader("📊 Univariate Analysis")

    uni_type = st.radio(
        "Select variable type:",
        ["Numerical", "Categorical"],
        horizontal=True
    )

    if uni_type == "Numerical":
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not num_cols:
            st.info("No numerical columns available.")
        else:
            col = st.selectbox("Select numerical column:", num_cols)

            st.markdown(f"**Distribution of `{col}`**")

            fig, ax = plt.subplots(1, 2, figsize=(10, 4))

            sns.histplot(df[col].dropna(), kde=True, ax=ax[0])
            ax[0].set_title("Histogram")

            sns.boxplot(x=df[col].dropna(), ax=ax[1])
            ax[1].set_title("Boxplot")

            st.pyplot(fig)

            st.markdown("**Summary Statistics**")
            st.write(df[col].describe())

    else:
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        if not cat_cols:
            st.info("No categorical columns available.")
        else:
            col = st.selectbox("Select categorical column:", cat_cols)

            st.markdown(f"**Distribution of `{col}`**")

            counts = df[col].value_counts()

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=counts.index, y=counts.values, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_title("Category Count")

            st.pyplot(fig)
            st.write(counts.to_frame("Count"))

    st.divider()

    # ==================================================
    # 3. BIVARIATE ANALYSIS
    # ==================================================
    st.subheader("🔗 Bivariate Analysis")

    if st.session_state.target_var is None:
        st.warning("Please select a target variable first.")
    else:
        target = st.session_state.target_var
        target_is_numeric = pd.api.types.is_numeric_dtype(df[target])

        feature = st.selectbox(
            "Select feature variable:",
            [c for c in df.columns if c != target]
        )

        st.markdown(f"**Relationship: `{feature}` vs `{target}`**")

        fig, ax = plt.subplots(figsize=(7, 4))

        if target_is_numeric:
            if pd.api.types.is_numeric_dtype(df[feature]):
                sns.scatterplot(x=df[feature], y=df[target], ax=ax)
            else:
                sns.boxplot(x=df[feature], y=df[target], ax=ax)
        else:
            if pd.api.types.is_numeric_dtype(df[feature]):
                sns.boxplot(x=df[target], y=df[feature], ax=ax)
            else:
                ct = pd.crosstab(df[feature], df[target])
                ct.plot(kind="bar", stacked=True, ax=ax)

        st.pyplot(fig)

    st.divider()

    # ==================================================
    # 4. CORRELATION ANALYSIS
    # ==================================================
    st.subheader("🔥 Correlation Analysis")

    num_df = df.select_dtypes(include=np.number)

    if num_df.shape[1] < 2:
        st.info("Not enough numerical columns for correlation.")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.divider()

    # ==================================================
    # 5. INSIGHTS SUMMARY
    # ==================================================
    st.subheader("🧠 Key Insights from EDA")

    st.markdown("""
    - Dataset structure and variable distributions were analyzed  
    - Univariate analysis highlighted skewness and dominant categories  
    - Bivariate analysis revealed relationships with the target variable  
    - Correlation analysis identified strongly related numerical features  
    """)

