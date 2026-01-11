import streamlit as st
import os
from PIL import Image   # ✅ REQUIRED

from views.upload import upload_page
from views.preprocessing import preprocessing_page
from views.eda import eda_page
from views.feature_engineering import feature_engineering_page
from views.model import model_page
from views.prediction import prediction_page


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Profiling Dashboard",
    layout="wide"
)


# ---------------- SIDEBAR ----------------

# ---- LOGO ABOVE TITLE (SAFE METHOD) ----
logo_path = os.path.join(
    os.path.dirname(__file__),
    "assets",
    "logo1.png"
)

if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.sidebar.image(logo, width=160)   # 👈 adjust size here
else:
    st.side
