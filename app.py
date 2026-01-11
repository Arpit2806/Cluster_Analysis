import streamlit as st

from views.upload import upload_page
from views.preprocessing import preprocessing_page
from views.eda import eda_page
from views.feature_engineering import feature_engineering_page
from views.model import model_page
from views.prediction import prediction_page

st.set_page_config(
    page_title="Customer Profiling Dashboard",
    layout="wide"
)

# ---------- SIDEBAR ----------
st.sidebar.title("📊 Customer_Profiling_Dashboard")
st.sidebar.info("🔷 Logo will be added here")

page = st.sidebar.selectbox(
    "Pages",
    (
        "📂 Upload Dataset",
        "🛠️ Preprocessing Stage",
        "📊 EDA",
        "⚙️ Feature Engineering",
        "🤖 Model Building",
        "📈 Prediction & Insights"
    )
)

# ---------- ROUTING ----------
if page == "📂 Upload Dataset":
    upload_page()

elif page == "🛠️ Preprocessing Stage":
    preprocessing_page()

elif page == "📊 EDA":
    eda_page()

elif page == "⚙️ Feature Engineering":
    feature_engineering_page()

elif page == "🤖 Model Building":
    model_page()

elif page == "📈 Prediction & Insights":
    prediction_page()
