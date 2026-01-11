import streamlit as st

from views.upload import upload_page
from views.preprocessing import preprocessing_page
from views.eda import eda_page
from views.feature_engineering import feature_engineering_page
from views.model import model_page
from views.prediction import prediction_page

import os
# from PIL import Image

# logo_path = os.path.join(
#     os.path.dirname(__file__),
#     "assets",
#     "logo1.png"
# )

# logo = Image.open(logo_path)
# st.sidebar.image(logo, use_container_width=True)



st.set_page_config(
    page_title="Customer Profiling Dashboard",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
import os

logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo1.png")
st.sidebar.image(logo_path, use_container_width=True)
st.sidebar.title("📊 Customer Profiling Dashboard")
# st.sidebar.info("🔷 Logo will be added here")

page = st.sidebar.radio(
    "Pages",
    [
        "📂 Upload Dataset",
        "🛠️ Preprocessing Stage",
        "📊 EDA",
        "⚙️ Feature Engineering",
        "🤖 Model Building",
        "📈 Prediction & Insights"
    ]
)

# ---------------- ROUTING ----------------
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
