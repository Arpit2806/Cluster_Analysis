import streamlit as st
import os
from PIL import Image, ImageDraw

from views.upload import upload_page
from views.preprocessing import preprocessing_page
from views.eda import eda_page
from views.supervised import supervised_learning_page
from views.unsupervised import unsupervised_learning_page
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
    st.sidebar.warning("Logo not found")


# ---- TITLE ----
st.sidebar.title("Customer Profiling Dashboard")


# ================= SIDEBAR NAVIGATION =================

if st.sidebar.button("📂 Upload Dataset", use_container_width=True):
    st.session_state.active_page = "Upload"

if st.sidebar.button("🛠️ Preprocessing Stage", use_container_width=True):
    st.session_state.active_page = "Preprocessing"

if st.sidebar.button("📊 EDA", use_container_width=True):
    st.session_state.active_page = "EDA"

if st.sidebar.button("⚙️ Supervised Learning", use_container_width=True):
    st.session_state.active_page = "Supervised"

if st.sidebar.button("🧩 Unsupervised Learning", use_container_width=True):
    st.session_state.active_page = "Unsupervised"

if st.sidebar.button("🤖 Model Building", use_container_width=True):
    st.session_state.active_page = "Model"

if st.sidebar.button("📈 Prediction & Insights", use_container_width=True):
    st.session_state.active_page = "Prediction"


# ================= MAIN ROUTING =================
if st.session_state.active_page == "Upload":
    upload_page()

elif st.session_state.active_page == "Preprocessing":
    preprocessing_page()

elif st.session_state.active_page == "EDA":
    eda_page()

elif st.session_state.active_page == "Supervised":
    supervised_learning_page()

elif st.session_state.active_page == "Unsupervised":
    unsupervised_learning_page()

elif st.session_state.active_page == "Model":
    model_page()

elif st.session_state.active_page == "Prediction":
    prediction_page()


# ================= FOOTER =================
st.markdown(
    """
    <div class="app-footer">
        DMUSL End-Term Hackathon
    </div>
    """,
    unsafe_allow_html=True
)
