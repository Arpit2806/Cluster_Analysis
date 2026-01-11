import streamlit as st

st.set_page_config(
    page_title="Customer Profiling Dashboard",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Customer_Profiling_Dashboard")
st.sidebar.info("🔷 Logo will be added here")

st.sidebar.markdown("### Pages")

st.sidebar.page_link("views/upload.py", label="📂 Upload Dataset")
st.sidebar.page_link("views/preprocessing.py", label="🛠️ Preprocessing Stage")
# st.sidebar.page_link("pages/eda.py", label="📊 EDA")
# st.sidebar.page_link("pages/feature_engineering.py", label="⚙️ Feature Engineering")
# st.sidebar.page_link("pages/model.py", label="🤖 Model Building")
# st.sidebar.page_link("pages/prediction.py", label="📈 Prediction & Insights")

# ---------------- MAIN ----------------
st.title("Customer Profiling Dashboard")
st.write("""
Welcome to the Customer Profiling Dashboard.

Use the sidebar to navigate through different stages:
1. Upload Dataset  
2. Preprocessing  
3. EDA  
4. Feature Engineering  
5. Model Building  
6. Prediction & Insights  
""")
