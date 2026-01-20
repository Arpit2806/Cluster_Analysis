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
from views.pca import pca_page          # ✅ PCA ADDED
from views.model import model_page
from views.prediction import prediction_page


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="DatathonX Dashboard",
    layout="wide"
)


# ================= LOAD CSS =================
def load_css():
    css_path = os.path.join(
        os.path.dirname(__file__),
        "assets",
        "styles.css"
    )
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# ================= SIDEBAR =================

# ---- CIRCULAR PROFILE LOGO ----
logo_path = os.path.join(
    os.path.dirname(__file__),
    "assets",
    "logo1.png"
)

if os.path.exists(logo_path):
    img = Image.open(logo_path).convert("RGBA")
    img = img.resize((110, 110))

    mask = Image.new("L", (110, 110), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, 110, 110), fill=255)
    img.putalpha(mask)

    st.sidebar.image(img)
else:
    st.sidebar.warning("Logo not found")

st.sidebar.title("Customer Profiling Dashboard")


# ================= SIDEBAR NAVIGATION =================

page = st.sidebar.radio(
    "Navigation",
    [
        "📂 Upload Dataset",
        "🛠️ Preprocessing Stage",
        "📊 EDA",
        "📉 Factor Analysis",
        "📉 PCA",                        # ✅ PCA OPTION
        "📊 K-Means Clustering",
        "🧺 Association Rule Mining",
        "⚙️ Supervised Learning",
        "🤖 Model Building",
        "📈 Prediction & Insights",
    ],
    index=0
)


# ================= MAIN ROUTING =================

if page == "📂 Upload Dataset":
    upload_page()

elif page == "🛠️ Preprocessing Stage":
    preprocessing_page()

elif page == "📊 EDA":
    eda_page()

elif page == "📉 Factor Analysis":
    factor_analysis_page()

elif page == "📉 PCA":                 # ✅ PCA ROUTE
    pca_page()

elif page == "📊 K-Means Clustering":
    kmeans_clustering_page()

elif page == "🧺 Association Rule Mining":
    arm_page()

elif page == "⚙️ Supervised Learning":
    supervised_learning_page()

elif page == "🤖 Model Building":
    model_page()

elif page == "📈 Prediction & Insights":
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

st.markdown(
    '<div class="footer-ribbon"></div>',
    unsafe_allow_html=True
)
