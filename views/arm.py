def arm_page():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    # --------------------------------------------------
    # HEADER & CONTEXT
    # --------------------------------------------------
    st.header("🧺 Association Rule Mining (ARM)")

    st.markdown("""
    **Why Association Rule Mining?**  
    Association Rule Mining identifies hidden co-occurrence patterns between variables
    using *If–Then* rules based on **support, confidence, and lift**.
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
    run_arm = st.radio(
        "Do you want to run Association Rule Mining on this dataset?",
        ["Yes, run ARM", "No, I want to upload another dataset"]
    )

    # ==================================================
    # IF YES
    # ==================================================
    if run_arm == "Yes, run ARM":

        # --------------------------------------------------
        # COLUMN SELECTION
        # --------------------------------------------------
        st.subheader("🔧 Select Columns for ARM")

        cols = st.multiselect(
            "Select categorical / survey columns:",
            df.columns.tolist()
        )

        if len(cols) < 2:
            st.warning("Please select at least two columns.")
            return

        arm_df = df[cols].astype(str)

        # --------------------------------------------------
        # TRANSACTIONS & ENCODING
        # --------------------------------------------------
        transactions = arm_df.values.tolist()

        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)

        st.subheader("📦 Encoded Transactions (Preview)")
        st.dataframe(df_encoded.head())

        # --------------------------------------------------
        # ARM PARAMETERS
        # --------------------------------------------------
        st.subheader("⚙️ ARM Parameters")

        min_support = st.slider("Minimum Support", 0.01, 0.5, 0.05, 0.01)
        min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.6, 0.05)
        min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.2, 0.1)

        # --------------------------------------------------
        # FREQUENT ITEMSETS
        # --------------------------------------------------
        frequent_itemsets = apriori(
            df_encoded,
            min_support=min_support,
            use_colnames=True
        )

        if frequent_itemsets.empty:
            st.warning("No frequent itemsets found. Try lowering support.")
            return

        st.subheader("📊 Frequent Itemsets")
        st.dataframe(frequent_itemsets.sort_values("support", ascending=False))

        # --------------------------------------------------
        # ASSOCIATION RULES
        # --------------------------------------------------
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence
        )

        rules = rules[rules["lift"] >= min_lift]

        if rules.empty:
            st.warning("No association rules found. Adjust thresholds.")
            return

        # Formatting
        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
        rules["antecedent_len"] = rules["antecedents"].apply(len)

        # --------------------------------------------------
        # RULES TABLE
        # --------------------------------------------------
        st.subheader("📜 All Association Rules")
        st.dataframe(
            rules[
                ["antecedents_str", "consequents_str", "support", "confidence", "lift"]
            ].sort_values("lift", ascending=False)
        )

        # ==================================================
        # 🔥 VISUAL 1: TOP-10 RULES (STRICT TOP-10)
        # ==================================================
        st.subheader("🏆 Top 10 Association Rules (by Lift)")

        top_10 = rules.sort_values("lift", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh(
            top_10["antecedents_str"] + " → " + top_10["consequents_str"],
            top_10["lift"],
            color=sns.color_palette("viridis", 10)
        )
        ax.set_xlabel("Lift")
        ax.set_title("Top 10 Rules by Lift")
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close(fig)

        # ==================================================
        # 🔥 VISUAL 2: SUPPORT vs CONFIDENCE (LIFT COLORFUL)
        # ==================================================
        st.subheader("📈 Support vs Confidence (Lift Highlighted)")

        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(
            rules["support"],
            rules["confidence"],
            c=rules["lift"],
            s=rules["lift"] * 40,
            cmap="plasma",
            alpha=0.7
        )
        ax.set_xlabel("Support")
        ax.set_ylabel("Confidence")
        plt.colorbar(scatter, ax=ax, label="Lift")
        st.pyplot(fig)
        plt.close(fig)

        # ==================================================
        # 🔥 VISUAL 3: CONFIDENCE DISTRIBUTION
        # ==================================================
        st.subheader("📊 Confidence Distribution")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(rules["confidence"], bins=10, color="#4CAF50", edgecolor="black")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        plt.close(fig)

        # ==================================================
        # 🔥 VISUAL 4: RULE COMPLEXITY (ANTECEDENT LENGTH)
        # ==================================================
        st.subheader("📏 Rule Complexity (Antecedent Length)")

        fig, ax = plt.subplots(figsize=(6, 4))
        rules["antecedent_len"].value_counts().sort_index().plot(
            kind="bar",
            ax=ax,
            color=sns.color_palette("Set2")
        )
        ax.set_xlabel("Number of Items in Antecedent")
        ax.set_ylabel("Number of Rules")
        st.pyplot(fig)
        plt.close(fig)

        # --------------------------------------------------
        # DOWNLOAD RULES
        # --------------------------------------------------
        st.download_button(
            "⬇️ Download Association Rules",
            rules.to_csv(index=False),
            file_name="association_rules.csv",
            mime="text/csv"
        )

        # --------------------------------------------------
        # SUMMARY INSIGHTS (NOTEBOOK-STYLE)
        # --------------------------------------------------
        st.subheader("🧠 Summary Insights")

        st.markdown(f"""
        - **Total rules generated:** {rules.shape[0]}
        - **Average confidence:** {rules['confidence'].mean():.2f}
        - **Maximum lift observed:** {rules['lift'].max():.2f}
        - **Most common antecedent size:** {rules['antecedent_len'].mode()[0]}
        """)

    # ==================================================
    # IF NO
    # ==================================================
    else:
        st.info("Upload another clean dataset suitable for Association Rule Mining.")

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            new_df = pd.read_csv(uploaded_file)
            st.session_state["data"] = new_df
            st.success("New dataset uploaded successfully!")
            st.dataframe(new_df.head())
