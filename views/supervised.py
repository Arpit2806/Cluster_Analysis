import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def supervised_learning_page():

    st.header("🤖 Supervised Learning")

    # ==================================================
    # CHECKS
    # ==================================================
    if "df_temp" not in st.session_state or "target_var" not in st.session_state:
        st.warning("⚠️ Complete EDA first (df_temp / target variable missing).")
        return

    df = st.session_state["df_temp"].copy()
    target = st.session_state["target_var"]

    if target not in df.columns:
        st.error("❌ Target variable not found in dataset.")
        return

    # ==================================================
    # PROBLEM TYPE DETECTION
    # ==================================================
    if pd.api.types.is_numeric_dtype(df[target]):
        problem_type = "Regression"
    else:
        problem_type = "Classification"

    st.success(f"Detected problem type: **{problem_type}**")

    # ==================================================
    # FEATURE / TARGET SPLIT
    # ==================================================
    X = df.drop(columns=[target])
    y = df[target]

    # Encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    st.divider()

    # ==================================================
    # TRAIN-TEST SPLIT
    # ==================================================
    st.subheader("🔀 Train-Test Split")

    test_size = st.slider("Test size (%)", 20, 40, 30) / 100

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # ==================================================
    # SCALING
    # ==================================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==================================================
    # MODEL SELECTION
    # ==================================================
    st.subheader("🧠 Select Models")

    if problem_type == "Regression":
        model_options = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=100, random_state=42
            )
        }
    else:
        model_options = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Random Forest Classifier": RandomForestClassifier(
                n_estimators=100, random_state=42
            )
        }

    selected_models = st.multiselect(
        "Choose models to train:",
        list(model_options.keys()),
        default=list(model_options.keys())
    )

    if not selected_models:
        st.warning("⚠️ Select at least one model.")
        return

    st.divider()

    # ==================================================
    # TRAINING & EVALUATION
    # ==================================================
    st.subheader("📊 Model Performance")

    results = []

    for model_name in selected_models:
        model = model_options[model_name]

        # Scale-dependent models
        if model_name in [
            "Linear Regression",
            "Ridge Regression",
            "Logistic Regression",
            "KNN"
        ]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        if problem_type == "Regression":
            results.append({
                "Model": model_name,
                "R2": r2_score(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "MAE": mean_absolute_error(y_test, y_pred)
            })
        else:
            results.append({
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted"),
                "Recall": recall_score(y_test, y_pred, average="weighted"),
                "F1 Score": f1_score(y_test, y_pred, average="weighted")
            })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)

    st.divider()

    # ==================================================
    # BEST MODEL SELECTION
    # ==================================================
    st.subheader("🏆 Best Model")

    if problem_type == "Regression":
        best_row = results_df.sort_values(by="R2", ascending=False).iloc[0]
    else:
        best_row = results_df.sort_values(by="F1 Score", ascending=False).iloc[0]

    st.success(f"Best Model: **{best_row['Model']}**")

    st.session_state["best_model_name"] = best_row["Model"]
    st.session_state["model_comparison"] = results_df

    st.markdown("""
    ### 🧠 What this page does
    - Automatically detects regression vs classification  
    - Encodes categorical features  
    - Trains multiple supervised models  
    - Compares models using appropriate metrics  
    - Selects the best-performing model  
    """)
