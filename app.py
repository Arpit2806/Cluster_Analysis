import streamlit as st

st.set_page_config(
    page_title="Customer Profiling Dashboard",
    layout="wide"
)

st.sidebar.title("📊 Customer_Profiling_Dashboard")
st.sidebar.info("🔷 Logo will be added here")

st.title("Welcome to Customer Profiling Dashboard")
st.write("""
Use the sidebar to navigate through different stages of the project:
- Dataset Upload
- Preprocessing
- EDA
- Feature Engineering
- Model Building
- Predictions & Insights
""")
