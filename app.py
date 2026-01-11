import streamlit as st
import os
from PIL import Image

from views.upload import upload_page
from views.preprocessing import preprocessing_page
from views.eda import eda_page
from views.feature_engineering import feature_engineering_page
from views.model import model_page
from views.prediction import prediction_page


# ---------------- PAGE CONFIG (MUST BE FIRST) ----------------
st.set_page_config(
    page_title="Customer Profiling Dashboard",
    layout="wide"
)


# ---------------- LOAD CSS (THIS WAS MISSING ❗) ----------------
def load_css():
    css_path = os.path.join(
        os.path.dirname(__file__),
        "assets",
        "styles.css"
    )
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()   # ✅ DO NOT REMOVE


# ---------------- SIDEBAR ----------------

# ---- LOGO (ABOVE TITLE, RESIZED, SAFE) ----
logo_path = os.path.join(
    os.path.dirname(__file__),
    "assets",
    "logo1.png"
)

if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.sidebar.image(logo, width=160)
else:
    st.sidebar.warning("Logo not found")

# ---- TITLE ----
st.sidebar.title("Customer Profiling Dashboard")


# ---- NAVIGATION ----
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
