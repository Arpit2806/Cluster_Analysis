import streamlit as st

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

# ---------------- SESSION STATE ----------------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Upload"

# ---------------- SIDEBAR (VERTICAL TABS) ----------------
st.sidebar.title("📊 Customer_Profiling_Dashboard")
st.sidebar.info("🔷 Logo will be added here")

st.sidebar.markdown("### Workflow")

if st.sidebar.button("📂 Upload Dataset", use_container_width=True):
    st.session_state.active_tab = "Upload"

if st.sidebar.button("🛠️ Preprocessing Stage", use_container_width=True):
    st.session_state.active_tab = "Preprocessing"

if st.sidebar.button("📊 EDA", use_container_width=True):
    st.session_state.active_tab = "EDA"

if st.sidebar.button("⚙️ Feature Engineering", use_container_width=True):
    st.session_state.active_tab = "Feature"

if st.sidebar.button("🤖 Model Building", use_container_width=True):
    st.session_state.active_tab = "Model"

if st.sidebar.button("📈 Prediction & Insights", use_container_width=True):
    st.session_state.active_tab = "Prediction"

st.sidebar.divider()
st.sidebar.markdown(
    f"**Current Step:** `{st.session_state.active_tab}`"
)

# ---------------- MAIN CONTENT ----------------
if st.session_state.active_tab == "Upload":
    upload_page()

elif st.session_state.active_tab == "Preprocessing":
    preprocessing_page()

elif st.session_state.active_tab == "EDA":
    eda_page()

elif st.session_state.active_tab == "Feature":
    feature_engineering_page()

elif st.session_state.active_tab == "Model":
    model_page()

elif st.session_state.active_tab == "Prediction":
    prediction_page()
