import streamlit as st

st.set_page_config(
    page_title="AcmeTel Churn Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏢 AcmeTel Churn Prediction Dashboard")
st.markdown("""
Welcome to the **AcmeTel Customer Churn Prediction Dashboard**.

Use the navigation on the left to access:

| Page | Description |
|------|-------------|
| 📊 **Monitoring** | Real-time prediction monitoring & analytics |
| 🔍 **Single Predict** | Predict churn for a single customer |
| 📁 **Batch Predict** | Upload CSV and predict for multiple customers |

---
""")

col1, col2, col3 = st.columns(3)
col1.info("📊 **Monitoring**\nView churn trends, risk distribution, and prediction history")
col2.info("🔍 **Single Predict**\nInput subscriber details and get instant churn prediction")
col3.info("📁 **Batch Predict**\nUpload CSV file and download results with risk levels")

st.divider()
st.caption("AcmeTel Churn Prediction API · XGBoost Model · ROC AUC ~0.86")
