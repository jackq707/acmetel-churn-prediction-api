import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import httpx
import pandas as pd
import io

st.set_page_config(page_title="Batch Predict — AcmeTel", page_icon="📁", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.2rem!important;padding-bottom:0.2rem!important}
[data-testid="stSidebar"]{background:#0f172a!important}
[data-testid="stSidebar"] *{color:#e2e8f0!important;font-size:14px!important}
.sh{font-size:12px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;padding-bottom:3px;border-bottom:2px solid #e2e8f0}
.stat-box{border-radius:10px;padding:14px;text-align:center;font-weight:700}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
  <div style="background:linear-gradient(135deg,#1e3a5f,#2563a8);border-radius:10px;padding:8px 12px;font-size:1.6rem;line-height:1">📁</div>
  <div>
    <div style="font-size:1.8rem;font-weight:900;color:#0f172a;line-height:1.1">Batch Predict</div>
    <div style="font-size:13px;color:#94a3b8;margin-top:2px">Upload CSV and predict churn for multiple customers at once</div>
  </div>
</div>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "https://jackq707-acmetel-churn-api.hf.space")

REQUIRED_COLS = [
    "gender","SeniorCitizen","Partner","Dependents","tenure",
    "PhoneService","MultipleLines","InternetService","OnlineSecurity",
    "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
    "StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
    "MonthlyCharges","TotalCharges_clean"
]

# ── Template download ─────────────────────────────────────────────────────────
col_dl, col_up, col_info = st.columns([1.2, 1.5, 1.3])

with col_dl:
    st.markdown('<div class="sh">📥 Step 1 — Download Template</div>', unsafe_allow_html=True)
    template_data = {
        "gender":["Female","Male"],
        "SeniorCitizen":[0,0],
        "Partner":["No","Yes"],
        "Dependents":["No","Yes"],
        "tenure":[5,48],
        "PhoneService":["Yes","Yes"],
        "MultipleLines":["No","Yes"],
        "InternetService":["Fiber optic","DSL"],
        "OnlineSecurity":["No","Yes"],
        "OnlineBackup":["No","Yes"],
        "DeviceProtection":["No","Yes"],
        "TechSupport":["No","Yes"],
        "StreamingTV":["No","No"],
        "StreamingMovies":["No","No"],
        "Contract":["Month-to-month","Two year"],
        "PaperlessBilling":["Yes","No"],
        "PaymentMethod":["Electronic check","Bank transfer (automatic)"],
        "MonthlyCharges":[80.0,55.0],
        "TotalCharges_clean":[400.0,2640.0]
    }
    df_template = pd.DataFrame(template_data)
    csv_template = df_template.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download CSV Template",
        data=csv_template,
        file_name="acmetel_batch_template.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.caption("Template contains 2 sample rows. Add your subscriber data.")

with col_up:
    st.markdown('<div class="sh">📤 Step 2 — Upload CSV</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV file", type=["csv"], label_visibility="collapsed")

with col_info:
    st.markdown('<div class="sh">📋 File Info</div>', unsafe_allow_html=True)
    if uploaded:
        st.success(f"✅ **{uploaded.name}**")
        st.caption(f"Size: {uploaded.size/1024:.1f} KB")
    else:
        st.info("No file uploaded yet")

st.markdown('<hr style="border:1px solid #e2e8f0;margin:6px 0">', unsafe_allow_html=True)

if uploaded:
    df_input = pd.read_csv(uploaded)
    st.markdown(f'<div class="sh" style="margin-top:6px">👀 Preview ({len(df_input)} rows)</div>', unsafe_allow_html=True)
    st.dataframe(df_input.head(2), use_container_width=True, hide_index=True, height=100)

    # Validate columns
    missing_cols = [c for c in REQUIRED_COLS if c not in df_input.columns]
    if missing_cols:
        st.error(f"❌ Missing columns: {missing_cols}")
        st.stop()

    st.success(f"✅ {len(df_input)} rows ready — all {len(REQUIRED_COLS)} required columns found")

    if st.button("▶️ Run Batch Prediction", use_container_width=True, type="primary"):
        records = df_input[REQUIRED_COLS].to_dict(orient="records")
        payload = {"items": [{"data": r} for r in records]}

        with st.spinner(f"Predicting {len(records)} customers..."):
            try:
                resp    = httpx.post(f"{API_URL}/predict_batch", json=payload, timeout=60)
                results = resp.json()["results"]

                # Build result DataFrame
                df_result = df_input.copy()
                df_result["churn_probability"] = [r["churn_probability"] for r in results]
                df_result["churn_flag"]        = [r["churn_flag"]        for r in results]
                df_result["risk_level"]        = df_result["churn_probability"].apply(
                    lambda p: "HIGH" if p>=0.70 else "MEDIUM" if p>=0.40 else "SAFE"
                )

                # Summary stats
                st.markdown('<div class="sh" style="margin-top:12px">📊 Summary</div>', unsafe_allow_html=True)
                total   = len(df_result)
                churned = int(df_result["churn_flag"].sum())
                safe    = total - churned
                high    = len(df_result[df_result["risk_level"]=="HIGH"])
                medium  = len(df_result[df_result["risk_level"]=="MEDIUM"])
                low     = len(df_result[df_result["risk_level"]=="SAFE"])
                avg_p   = df_result["churn_probability"].mean()

                c1,c2,c3,c4,c5,c6 = st.columns(6)
                c1.markdown(f'<div class="stat-box" style="background:#dbeafe;color:#1e40af">📋 Total<br><span style="font-size:1.2rem">{total}</span></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="stat-box" style="background:#fee2e2;color:#991b1b">🔴 Churn<br><span style="font-size:1.2rem">{churned}</span></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="stat-box" style="background:#dcfce7;color:#14532d">🟢 Safe<br><span style="font-size:1.2rem">{safe}</span></div>', unsafe_allow_html=True)
                c4.markdown(f'<div class="stat-box" style="background:#fee2e2;color:#991b1b">⚡ High<br><span style="font-size:1.2rem">{high}</span></div>', unsafe_allow_html=True)
                c5.markdown(f'<div class="stat-box" style="background:#fef9c3;color:#854d0e">🟡 Med<br><span style="font-size:1.2rem">{medium}</span></div>', unsafe_allow_html=True)
                c6.markdown(f'<div class="stat-box" style="background:#f1f5f9;color:#334155">📈 Avg<br><span style="font-size:1.2rem">{avg_p:.3f}</span></div>', unsafe_allow_html=True)

                # Results table
                st.markdown('<div class="sh" style="margin-top:14px">📋 Prediction Results</div>', unsafe_allow_html=True)

                show_cols = ["gender","tenure","Contract","InternetService",
                             "MonthlyCharges","churn_probability","churn_flag","risk_level"]
                available = [c for c in show_cols if c in df_result.columns]
                df_show   = df_result[available].copy()
                df_show["churn_probability"] = df_show["churn_probability"].map("{:.4f}".format)

                st.markdown("**Filter by Risk Level:**")
                rf1, rf2, rf3 = st.columns(3)
                show_high   = rf1.checkbox("🔴 HIGH",   value=True)
                show_medium = rf2.checkbox("🟡 MEDIUM", value=True)
                show_safe   = rf3.checkbox("🟢 SAFE",   value=True)
                risk_filter = []
                if show_high:   risk_filter.append("HIGH")
                if show_medium: risk_filter.append("MEDIUM")
                if show_safe:   risk_filter.append("SAFE")
                df_filtered = df_show[df_result["risk_level"].isin(risk_filter)]

                def color_risk(val):
                    return {
                        "HIGH"  :"background-color:#fee2e2;color:#991b1b;font-weight:700",
                        "MEDIUM":"background-color:#fef9c3;color:#854d0e;font-weight:700",
                        "SAFE"  :"background-color:#dcfce7;color:#166534;font-weight:700"
                    }.get(val, "")

                st.dataframe(
                    df_filtered.style.applymap(color_risk, subset=["risk_level"]),
                    use_container_width=True, hide_index=True, height=140
                )

                # Download buttons
                st.markdown('<div class="sh" style="margin-top:8px">📥 Download Results</div>', unsafe_allow_html=True)
                d1, d2 = st.columns(2)

                csv_out = df_result.to_csv(index=False).encode("utf-8")
                d1.download_button("📥 Download CSV", data=csv_out,
                    file_name="churn_predictions.csv", mime="text/csv",
                    use_container_width=True)

                # Excel
                excel_buf = io.BytesIO()
                with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                    df_result.to_excel(writer, index=False, sheet_name="Predictions")
                d2.download_button("📊 Download Excel", data=excel_buf.getvalue(),
                    file_name="churn_predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True)

            except Exception as e:
                st.error(f"API Error: {e}")
else:
    st.markdown("""
    <div style="background:#f8fafc;border:2px dashed #cbd5e1;border-radius:12px;padding:40px;text-align:center">
        <div style="font-size:3rem">📁</div>
        <div style="font-size:15px;color:#64748b;margin-top:8px;font-weight:600">Upload CSV file to start batch prediction</div>
        <div style="font-size:13px;color:#94a3b8;margin-top:4px">Download the template above to get the correct format</div>
    </div>
    """, unsafe_allow_html=True)
