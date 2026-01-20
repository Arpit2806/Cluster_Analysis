def pca_page():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # --------------------------------------------------
    # HEADER & CONTEXT
    # --------------------------------------------------
    st.header("📉 Principal Component Analysis (PCA)")

    st.markdown("""
    **Why PCA?**  
    Principal Component Analysis is used to reduce high-dimensional data into a smaller
    number of uncorrelated components while preserving maximum variance.  
    It helps in **dimensionality reduction**, **noise reduction**, and **data visualization**.
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
    run_pca = st.radio(
        "Do you want to run PCA on this dataset?",
        ["Yes, run PCA", "No, I want to upload another dataset"]
    )

    # ==================================================
    # IF YES
    # ==================================================
    if run_pca == "Yes, run PCA":

        # --------------------------------------------------
        # FEATURE SELECTION
        # --------------------------------------------------
        st.subheader("🔧 Select Numeric Variables")

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("PCA requires at least 2 numeric variables.")
            return

        features = st.multiselect(
            "Select variables for PCA:",
            numeric_cols,
            default=numeric_cols
        )

        if len(features) < 2:
            st.warning("Please select at least two variables.")
            return

        X = df[features].dropna()

        # --------------------------------------------------
        # STANDARDIZATION
        # --------------------------------------------------
        st.subheader("⚖️ Data Standardization")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.success("Data has been standardized successfully.")

        # --------------------------------------------------
        # PCA FIT (ALL COMPONENTS)
        # --------------------------------------------------
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        # --------------------------------------------------
        # EXPLAINED VARIANCE (BAR CHART)
        # --------------------------------------------------
        st.subheader("📊 Explained Variance by Principal Components")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(
            range(1, len(explained_variance) + 1),
            explained_variance,
            color="#5b5fe8"
        )
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")
        st.pyplot(fig)
        plt.close(fig)

        # --------------------------------------------------
        # CUMULATIVE EXPLAINED VARIANCE
        # --------------------------------------------------
        st.subheader("📈 Cumulative Explained Variance")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            range(1, len(cumulative_variance) + 1),
            cumulative_variance,
            marker="o",
            color="#4a4fd8"
        )
        ax.axhline(y=0.8, linestyle="--", color="red", label="80% Variance")
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        # --------------------------------------------------
        # SCREE PLOT (EIGENVALUES)
        # --------------------------------------------------
        st.subheader("📉 Scree Plot")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            range(1, len(pca.explained_variance_) + 1),
            pca.explained_variance_,
            marker="o"
        )
        ax.set_xlabel("Component Number")
        ax.set_ylabel("Eigenvalue")
        st.pyplot(fig)
        plt.close(fig)

        # --------------------------------------------------
        # SELECT NUMBER OF COMPONENTS
        # --------------------------------------------------
        n_components = st.slider(
            "Select number of principal components",
            min_value=2,
            max_value=min(10, len(features)),
            value=2
        )

        pca_final = PCA(n_components=n_components)
        X_pca_final = pca_final.fit_transform(X_scaled)

        # --------------------------------------------------
        # PCA 2D SCATTER PLOT
        # --------------------------------------------------
        if n_components >= 2:
            st.subheader("🧭 PCA 2D Projection (PC1 vs PC2)")

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(
                X_pca_final[:, 0],
                X_pca_final[:, 1],
                alpha=0.7,
                color="#6d71ff"
            )
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            st.pyplot(fig)
            plt.close(fig)

        # --------------------------------------------------
        # PCA LOADINGS
        # --------------------------------------------------
        st.subheader("📋 PCA Loadings")

        loadings = pd.DataFrame(
            pca_final.components_.T,
            index=features,
            columns=[f"PC{i+1}" for i in range(n_components)]
        )

        st.markdown(
            loadings.to_html(classes="data-table", index=True),
            unsafe_allow_html=True
        )

        # --------------------------------------------------
        # DOWNLOAD PCA OUTPUT
        # --------------------------------------------------
        pca_df = pd.DataFrame(
            X_pca_final,
            columns=[f"PC{i+1}" for i in range(n_components)]
        )

        st.download_button(
            "⬇️ Download PCA Transformed Data",
            pca_df.to_csv(index=False),
            file_name="pca_transformed_data.csv",
            mime="text/csv"
        )

    # ==================================================
    # IF NO
    # ==================================================
    else:
        st.info("Upload another dataset suitable for PCA.")

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            new_df = pd.read_csv(uploaded_file)
            st.session_state["data"] = new_df
            st.success("New dataset uploaded successfully!")
            st.dataframe(new_df.head())
