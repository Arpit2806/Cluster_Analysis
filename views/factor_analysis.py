def factor_analysis_page():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
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
    Factor Analysis reduces a large set of correlated variables into fewer meaningful
    latent factors. It is commonly used for survey and Likert-scale data to uncover
    hidden behavioral dimensions.
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
        # DATA CLEANING (CRITICAL)
        # --------------------------------------------------
        data = df[features].copy()
        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.dropna()
        data = data.loc[:, data.nunique() > 1]

        if data.shape[1] < 3:
            st.error("After cleaning, fewer than 3 valid variables remain.")
            return

        if data.shape[0] < 10:
            st.error("Not enough observations to perform Factor Analysis.")
            return

        # --------------------------------------------------
        # STANDARDIZATION
        # --------------------------------------------------
        scaler = StandardScaler()
        X = scaler.fit_transform(data)
        X = np.asarray(X)  # pure NumPy array

        # --------------------------------------------------
        # CORRELATION HEATMAP
        # --------------------------------------------------
        st.subheader("📊 Correlation Heatmap")

        corr = pd.DataFrame(X, columns=data.columns).corr()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # --------------------------------------------------
        # KMO TEST
        # --------------------------------------------------
        st.subheader("📐 KMO Test")

        kmo_all, kmo_model = calculate_kmo(X)
        st.metric("KMO Value", round(kmo_model, 3))

        # --------------------------------------------------
        # BARTLETT TEST
        # --------------------------------------------------
        st.subheader("📐 Bartlett’s Test of Sphericity")

        chi_square_value, p_value = calculate_bartlett_sphericity(X)

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
        # SCREE PLOT & EIGENVALUES (PCA-BASED, STABLE)
        # --------------------------------------------------
        st.subheader("📈 Scree Plot & Eigenvalues")

        pca_for_scree = PCA()
        pca_for_scree.fit(X)
        eigen_values = pca_for_scree.explained_variance_

        fig, ax = plt.subplots()
        ax.plot(range(1, len(eigen_values) + 1), eigen_values, marker="o")
        ax.axhline(y=1, color="red", linestyle="--")
        ax.set_xlabel("Factor Number")
        ax.set_ylabel("Eigenvalue")
        ax.set_title("Scree Plot (PCA-based)")
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
        # FACTOR EXTRACTION (VARIMAX WITH FALLBACK)
        # --------------------------------------------------
        st.subheader("🔄 Factor Extraction (Varimax Rotation)")

        loadings = None
        factor_scores = None

        try:
            # Primary attempt: Proper Factor Analysis
            fa = FactorAnalyzer(
                n_factors=n_factors,
                rotation="varimax",
                method="principal"
            )
            fa.fit(X)

            loadings = pd.DataFrame(
                fa.loadings_,
                index=data.columns,
                columns=[f"Factor {i+1}" for i in range(n_factors)]
            )

            factor_scores = fa.transform(X)

            st.success("Factor Analysis completed successfully using Varimax rotation.")

        except Exception:
            # Fallback: PCA-based factor loadings
            st.warning(
                "Numerical instability detected in Factor Analyzer. "
                "Using PCA-based factor loadings as a stable alternative."
            )

            pca_fallback = PCA(n_components=n_factors)
            factor_scores = pca_fallback.fit_transform(X)

            loadings = pd.DataFrame(
                pca_fallback.components_.T,
                index=data.columns,
                columns=[f"Factor {i+1}" for i in range(n_factors)]
            )

        # --------------------------------------------------
        # FACTOR LOADINGS
        # --------------------------------------------------
        st.subheader("📋 Factor Loadings")
        st.dataframe(loadings.style.background_gradient(cmap="coolwarm"))

        # --------------------------------------------------
        # FACTOR SCORES
        # --------------------------------------------------
        st.subheader("📌 Factor Scores")

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
