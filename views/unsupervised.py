import streamlit as st

def kmeans_clustering_page():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    st.header("📊 K-Means Clustering")

    # --------------------------------------------------
    # WHY & WHAT
    # --------------------------------------------------
    st.markdown("""
    **Why K-Means Clustering?**  
    K-Means clustering groups similar data points together based on their behavior.  
    It helps identify distinct segments that can be used for targeted strategies and better decision-making.
    """)

    # --------------------------------------------------
    # CHECK IF DATA EXISTS
    # --------------------------------------------------
    if "data" not in st.session_state:
        st.warning("No dataset found. Please upload a dataset first.")
        return

    df = st.session_state["data"]

    # --------------------------------------------------
    # DECISION GATE
    # --------------------------------------------------
    decision = st.radio(
        "Do you want to run cluster analysis on this dataset?",
        ["Yes, run clustering", "No, I want to use another dataset"]
    )

    # ==================================================
    # YES → RUN CLUSTERING
    # ==================================================
    if decision == "Yes, run clustering":

        st.subheader("🔧 Select Features for Clustering")

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("At least two numeric features are required for clustering.")
            return

        features = st.multiselect(
            "Choose numeric features:",
            numeric_cols,
            default=numeric_cols
        )

        if len(features) < 2:
            st.warning("Please select at least two features.")
            return

        # --------------------------------------------------
        # SCALING
        # --------------------------------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])

        # --------------------------------------------------
        # ELBOW METHOD
        # --------------------------------------------------
        st.subheader("📈 Elbow Method (Optimal Number of Clusters)")

        inertia = []
        K_range = range(1, 11)

        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertia.append(km.inertia_)

        fig, ax = plt.subplots()
        ax.plot(K_range, inertia, marker="o")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Plot")
        st.pyplot(fig)

        # --------------------------------------------------
        # SELECT K
        # --------------------------------------------------
        k = st.slider(
            "Select number of clusters (K)",
            min_value=2,
            max_value=10,
            value=3
        )

        # --------------------------------------------------
        # RUN K-MEANS
        # --------------------------------------------------
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        df_clustered = df.copy()
        df_clustered["Cluster"] = clusters

        # --------------------------------------------------
        # PCA FOR VISUALIZATION
        # --------------------------------------------------
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(X_scaled)

        df_clustered["PCA1"] = pca_components[:, 0]
        df_clustered["PCA2"] = pca_components[:, 1]

        # --------------------------------------------------
        # CLUSTER VISUALIZATION
        # --------------------------------------------------
        st.subheader("🧭 Cluster Visualization (PCA Reduced)")

        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df_clustered,
            x="PCA1",
            y="PCA2",
            hue="Cluster",
            palette="tab10",
            ax=ax
        )
        ax.set_title("Customer Segments")
        st.pyplot(fig)

        # --------------------------------------------------
        # CLUSTER SIZE DISTRIBUTION
        # --------------------------------------------------
        st.subheader("📊 Cluster Size Distribution")

        cluster_counts = df_clustered["Cluster"].value_counts().sort_index()

        fig, ax = plt.subplots()
        cluster_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of Records")
        st.pyplot(fig)

        # --------------------------------------------------
        # CLUSTER PROFILES
        # --------------------------------------------------
        st.subheader("📋 Cluster Profiles (Average Values)")

        profile = df_clustered.groupby("Cluster")[features].mean()
        st.dataframe(profile.style.background_gradient(cmap="coolwarm"))

        # --------------------------------------------------
        # DOWNLOAD DATA
        # --------------------------------------------------
        st.download_button(
            "⬇️ Download Clustered Dataset",
            df_clustered.to_csv(index=False),
            file_name="clustered_data.csv",
            mime="text/csv"
        )

    # ==================================================
    # NO → UPLOAD NEW DATASET
    # ==================================================
    else:
        st.info("If you have another clean dataset suitable for clustering, upload it below.")

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            new_df = pd.read_csv(uploaded_file)
            st.session_state["data"] = new_df
            st.success("New dataset uploaded successfully!")
            st.dataframe(new_df.head())
