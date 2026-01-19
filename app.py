import streamlit as st
import os
from PIL import Image, ImageDraw

# ================= IMPORT VIEWS =================
from views.upload import upload_page
from views.preprocessing import preprocessing_page
from views.eda import eda_page
from views.supervised import supervised_learning_page
from views.kmeans_clustering import kmeans_clustering_page
from views.factor_analysis import factor_analysis_page
from views.arm import arm_page
from views.model import model_page
from views.prediction import prediction_page


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Customer Profiling Dashboard",
    layout="wide"
)


# ================= LOAD CSS =================
def load_css():
    css_path = os.path.join(
        os.path.dirname(__file__),
        "assets",
        "styles.css"
    )
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# ================= SESSION STATE =================
if "active_page" not in st.session_state:
    st.session_state.active_page = "Upload"


# ================= SIDEBAR =================

# ---- CIRCULAR PROFILE LOGO ----
logo_path = os.path.join(
    os.path.dirname(__file__),
    "assets",
    "logo1.png"
)

if os.path.exists(logo_path):
    img = Image.open(logo_path).convert("RGBA")

    size = (110, 110)
    img = img.resize(size)

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size[0], size[1]), fill=255)
    img.putalpha(mask)

    col1, col2, col3 = st.sidebar.columns([1, 2, 1])
    with col2:
        st.image(img)
else:
    st.sidebar.war
