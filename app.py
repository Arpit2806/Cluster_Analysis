import streamlit as st

from pages.upload import upload_page
from pages.preprocessing import preprocessing_page

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Customer Profiling Dashboard",
    layout="wide"
)

# ---------------------- SIDEBAR ----------------------
st.sidebar.title("📊 Customer_Profiling_Dashboard")
st.sidebar.info("🔷 Logo will be added here")

page = st.sidebar.radio(
    "Navigation",
    [
        "1. Upload Dataset",
        "2. Preprocessing Stage",
        "3. EDA",
        "4. Feature Engineering",
        "5. Model Building",
        "6. Prediction & Insights"
    ]
)

# ---------------------- ROUTING ----------------------
if page == "1. Upload Dataset":
    upload_page()

elif page == "2. Preprocessing Stage":
    preprocessing_page()

elif page == "3. EDA":
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.info("EDA logic will be added later.")

elif page == "4. Feature Engineering":
    st.header("⚙️ Feature Engineering")
    st.info("Feature engineering logic will be added later.")

elif page == "5. Model Building":
    st.header("🤖 Model Building")
    st.info("Model training logic will be added later.")

elif page == "6. Prediction & Insights":
    st.header("📈 Prediction & Insights")
    st.info("Prediction and insights logic will be added later.")
