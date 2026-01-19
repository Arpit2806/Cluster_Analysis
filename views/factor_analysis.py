def factor_analysis_page():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.preprocessing import StandardScaler
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import (
        calculate_kmo,
        calculate_bartlett_sphericity
    )

    # --------------------------------------------------
    # HEADER & CONTEXT
    # --------------------------------------------------
    st.header("📉 Factor Analysis")

    st.markdown("""
    **Why Factor Analysis?**  
    Factor Analysis helps reduce a large number of correlated variables into a smaller
    set of meaningful latent factors. It is commonly used for survey and Likert-scale data
    to uncover hidden behavioral dimensions.
    """)

    # --------------------------------------------------
    # CHECK DATA
    # --------------------------------------------------
    if "data" not in st.session_state:
        st.warning("No dataset found. Please upload a dataset first.")
        return

    df = st.session_state["data"]

    # --------------------------------------------------
    # DECISION GATE
    # --------------------------------------------------
    run_fa = st.radio(
        "Do you want to run Factor Analysis on this dataset?",
        ["Yes, run Factor Analysis", "No, I want to upload another dataset"]
    )

    # ==================================================
    # IF YES
    # ==================================================
    if run_fa == "Yes, run Factor Analysis":

        # --------------------------------------------------
        # FEATURE SELECTION
        # --------------------------------------------------
        st.subheader("🔧 Select Variables (Likert / Numeric)")

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        if len(numeric_cols) < 3:
            st.error("Factor Analysis requires at least 3 numeric / Likert-scale variables.")
            return

        features = st.multiselect(
            "Select variables for Factor Analysis:",
            numeric_cols,
            default=numeric_cols
        )

        if len(features) < 3:
            st.warning("Please select at least 3 variables.")
            return

        # --------------------------------------------------
        # DATA CLEANING (CRITICAL FIX)
        # --------------------------------------------------
        data = df[features].copy()

        # Force numeric & remove invalid values
        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.dropna()

        # Remove constant (zero-variance) columns
        data = data.loc[:, data.nunique() > 1]

        if data.shape[1] < 3:
            st.error("After cleaning, fewer than 3 valid variables remain for Factor Analysis.")
            return

        if data.shape[0] < 10:
            st.error("Not enough observations to perform Factor Analysis.")
            return

        # --------------------------------------------------
        # STANDARDIZATION
        # --------------------------------------------------
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

        # --------------------------------------------------
        # CORRELATION HEATMAP
        # --------------------------------------------------
        st.subheader("📊 Correlation Heatmap")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(data_scaled.corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # --------------------------------------------------
        # KMO TEST
        # --------------------------------------------------
        st.subheader("📐 KMO Test")

        kmo_all, kmo_model = calculate_kmo(data_scaled)
        st.metric("KMO Value", round(kmo_model, 3))

        # --------------------------------------------------
        # BARTLETT TEST
        # --------------------------------------------------
        st.subheader("📐 Bartlett’s Test of Sphericity")

        chi_square_value, p_value = calculate_bartlett_sphericity(data_scaled)

        st.write(f"Chi-Square Value: **{round(chi_square_value, 2)}**")
        st.write(f"P-Value: **{round(p_value, 6)}**")

        if kmo_model < 0.6 or p_value >= 0.05:
            st.error(
                "Data is not suitable for Factor Analysis "
                "(KMO < 0.6 or Bartlett p-value ≥ 0.05)."
            )
            return

        st.success("Data is suitable for Factor Analysis.")

        # --------------------------------------------------
        # EIGENVALUES & SCREE PLOT
        # --------------------------------------------------
        st.subheader("📈 Scree Plot & Eigenvalues")

        fa = FactorAnalyzer(rotation=None)
        fa.fit(data_scaled)

        eigen_values, _ = fa.get_eigenvalues()

        fig, ax = plt.subplots()
        ax.plot(range(1, len(eigen_values) + 1), eigen_values, marker="o")
        ax.axhline(y=1, color="red", linestyle="--")
        ax.set_xlabel("Factor Number")
        ax.set_ylabel("Eigenvalue")
        ax.set_title("Scree Plot")
        st.pyplot(fig)

        # --------------------------------------------------
        # SELECT NUMBER OF FACTORS
        # --------------------------------------------------
        n_factors = st.slider(
            "Select number of factors",
            min_value=1,
            max_value=min(10, data.shape[1]),
            value=min(3, data.shape[1])
        )

        # --------------------------------------------------
        # APPLY FACTOR ANALYSIS
        # --------------------------------------------------
        st.subheader("🔄 Factor Extraction (Varimax Rotation)")

        fa = FactorAnalyzer(
            n_factors=n_factors,
            rotation="varimax"
        )
        fa.fit(data_scaled)

        # --------------------------------------------------
        # FACTOR LOADINGS
        # --------------------------------------------------
        loadings = pd.DataFrame(
            fa.loadings_,
            index=data.columns,
            columns=[f"Factor {i+1}" for i in range(n_factors)]
        )

        st.subheader("📋 Factor Loadings")
        st.dataframe(loadings.style.background_gradient(cmap="coolwarm"))

        # --------------------------------------------------
        # FACTOR SCORES
        # --------------------------------------------------
        st.subheader("📌 Factor Scores (Optional Output)")

        factor_scores = fa.transform(data_scaled)
        factor_scores_df = pd.DataFrame(
            factor_scores,
            columns=[f"Factor {i+1}" for i in range(n_factors)]
        )

        st.dataframe(factor_scores_df.head())

        st.download_button(
            "⬇️ Download Factor Scores",
            factor_scores_df.to_csv(index=False),
            file_name="factor_scores.csv",
            mime="text/csv"
        )

    # ==================================================
    # IF NO
    # ==================================================
    else:
        st.info("Upload another clean dataset suitable for Factor Analysis.")

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            new_df = pd.read_csv(uploaded_file)
            st.session_state["data"] = new_df
            st.success("New dataset uploaded successfully!")
            st.dataframe(new_df.head())
