import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import httpx
import pandas as pd
import io

st.set_page_config(page_title="Batch Predict — AcmeTel", page_icon="📁", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:3rem!important;padding-bottom:0.2rem!important}
[data-testid="stSidebar"]{background:#0f172a!important}
[data-testid="stSidebar"] *{color:#e2e8f0!important;font-size:14px!important}
.sh{font-size:12px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;padding-bottom:3px;border-bottom:2px solid #e2e8f0}
.stat-box{border-radius:8px;padding:8px 4px;text-align:center;font-weight:700}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
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

c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    st.markdown('<div class="sh">📥 Step 1 — Template</div>', unsafe_allow_html=True)
    template_data = {
        "gender":["Female","Male"],"SeniorCitizen":[0,0],
        "Partner":["No","Yes"],"Dependents":["No","Yes"],
        "tenure":[5,48],"PhoneService":["Yes","Yes"],
        "MultipleLines":["No","Yes"],"InternetService":["Fiber optic","DSL"],
        "OnlineSecurity":["No","Yes"],"OnlineBackup":["No","Yes"],
        "DeviceProtection":["No","Yes"],"TechSupport":["No","Yes"],
        "StreamingTV":["No","No"],"StreamingMovies":["No","No"],
        "Contract":["Month-to-month","Two year"],
        "PaperlessBilling":["Yes","No"],
        "PaymentMethod":["Electronic check","Bank transfer (automatic)"],
        "MonthlyCharges":[80.0,55.0],"TotalCharges_clean":[400.0,2640.0]
    }
    csv_template = pd.DataFrame(template_data).to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV Template", data=csv_template,
        file_name="acmetel_batch_template.csv", mime="text/csv", use_container_width=True)
    st.caption("2 sample rows. Add your subscriber data.")

with c2:
    st.markdown('<div class="sh">📤 Step 2 — Upload CSV</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload", type=["csv"], label_visibility="collapsed")

with c3:
    st.markdown('<div class="sh">✅ Step 3 — Validation</div>', unsafe_allow_html=True)
    if uploaded:
        df_check = pd.read_csv(uploaded)
        missing_cols = [c for c in REQUIRED_COLS if c not in df_check.columns]
        if missing_cols:
            st.error(f"❌ Missing columns: {len(missing_cols)}")
            for mc in missing_cols[:3]:
                st.caption(f"• {mc}")
        else:
            st.success(f"✅ {len(df_check)} rows ready")
            st.caption(f"All {len(REQUIRED_COLS)} columns found")
        else:
        st.markdown('<div style="background:#f8fafc;border:1px dashed #cbd5e1;border-radius:8px;padding:14px;text-align:center;color:#94a3b8;font-size:13px">Upload a CSV file to validate</div>', unsafe_allow_html=True)

st.markdown('<hr style="border:1px solid #e2e8f0;margin:6px 0">', unsafe_allow_html=True)

if uploaded:
    uploaded.seek(0)
    df_input = pd.read_csv(uploaded)
    missing_cols = [c for c in REQUIRED_COLS if c not in df_input.columns]
    if missing_cols:
        st.error(f"❌ Missing columns: {missing_cols}")
        st.stop()

    st.markdown(f'<div class="sh">👀 Preview ({len(df_input)} rows)</div>', unsafe_allow_html=True)
    st.dataframe(df_input.head(3), use_container_width=True, hide_index=True, height=130)

    if st.button("▶️ Run Batch Prediction", use_container_width=True, type="primary"):
        records = df_input[REQUIRED_COLS].to_dict(orient="records")
        payload = {"items": [{"data": r} for r in records]}
        with st.spinner(f"Predicting {len(records)} customers..."):
            try:
                resp    = httpx.post(f"{API_URL}/predict_batch", json=payload, timeout=60)
                results = resp.json()["results"]
                df_result = df_input.copy()
                df_result["churn_probability"] = [r["churn_probability"] for r in results]
                df_result["churn_flag"]        = [r["churn_flag"]        for r in results]
                df_result["risk_level"]        = df_result["churn_probability"].apply(
                    lambda p: "HIGH" if p>=0.70 else "MEDIUM" if p>=0.40 else "SAFE")

                col_sum, col_res = st.columns([1, 2])

                with col_sum:
                    st.markdown('<div class="sh" style="margin-top:4px">📊 Summary</div>', unsafe_allow_html=True)
                    total=len(df_result); churned=int(df_result["churn_flag"].sum())
                    safe=total-churned; high=len(df_result[df_result["risk_level"]=="HIGH"])
                    medium=len(df_result[df_result["risk_level"]=="MEDIUM"]); avg_p=df_result["churn_probability"].mean()

                    r1,r2=st.columns(2)
                    r1.markdown(f'<div class="stat-box" style="background:#dbeafe;color:#1e40af">📋 Total<br><span style="font-size:1.3rem">{total}</span></div>',unsafe_allow_html=True)
                    r2.markdown(f'<div class="stat-box" style="background:#fee2e2;color:#991b1b">🔴 Churn<br><span style="font-size:1.3rem">{churned}</span></div>',unsafe_allow_html=True)
                    r3,r4=st.columns(2)
                    r3.markdown(f'<div class="stat-box" style="background:#dcfce7;color:#14532d">🟢 Safe<br><span style="font-size:1.3rem">{safe}</span></div>',unsafe_allow_html=True)
                    r4.markdown(f'<div class="stat-box" style="background:#fee2e2;color:#991b1b">⚡ High<br><span style="font-size:1.3rem">{high}</span></div>',unsafe_allow_html=True)
                    r5,r6=st.columns(2)
                    r5.markdown(f'<div class="stat-box" style="background:#fef9c3;color:#854d0e">🟡 Med<br><span style="font-size:1.3rem">{medium}</span></div>',unsafe_allow_html=True)
                    r6.markdown(f'<div class="stat-box" style="background:#f1f5f9;color:#334155">📈 Avg<br><span style="font-size:1.3rem">{avg_p:.3f}</span></div>',unsafe_allow_html=True)

                    st.markdown('<div class="sh" style="margin-top:8px">📥 Download</div>', unsafe_allow_html=True)
                    st.download_button("📥 CSV", data=df_result.to_csv(index=False).encode("utf-8"),
                        file_name="churn_predictions.csv", mime="text/csv", use_container_width=True)
                    excel_buf=io.BytesIO()
                    with pd.ExcelWriter(excel_buf, engine="openpyxl") as w:
                        df_result.to_excel(w, index=False, sheet_name="Predictions")
                    st.download_button("📊 Excel", data=excel_buf.getvalue(),
                        file_name="churn_predictions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True)

                with col_res:
                    st.markdown('<div class="sh" style="margin-top:4px">📋 Prediction Results</div>', unsafe_allow_html=True)
                    f1,f2,f3,f4=st.columns([1.2,1,1,1])
                    f1.markdown("**Filter:**")
                    sh=f2.checkbox("🔴 HIGH",value=True)
                    sm=f3.checkbox("🟡 MED",value=True)
                    ss=f4.checkbox("🟢 SAFE",value=True)
                    risk_filter=[*( ["HIGH"] if sh else []),*( ["MEDIUM"] if sm else []),*( ["SAFE"] if ss else [])]

                    show_cols=["gender","tenure","Contract","InternetService","MonthlyCharges","churn_probability","churn_flag","risk_level"]
                    avail=[c for c in show_cols if c in df_result.columns]
                    dfs=df_result[avail].copy()
                    dfs["churn_probability"]=dfs["churn_probability"].map("{:.4f}".format)
                    dff=dfs[df_result["risk_level"].isin(risk_filter)]

                    def cr(val):
                        return {"HIGH":"background-color:#fee2e2;color:#991b1b;font-weight:700",
                                "MEDIUM":"background-color:#fef9c3;color:#854d0e;font-weight:700",
                                "SAFE":"background-color:#dcfce7;color:#166534;font-weight:700"}.get(val,"")

                    st.dataframe(dff.style.applymap(cr,subset=["risk_level"]),
                        use_container_width=True, hide_index=True, height=300)

            except Exception as e:
                st.error(f"API Error: {e}")
else:
    st.markdown("""
    <div style="background:#f8fafc;border:2px dashed #cbd5e1;border-radius:12px;padding:30px;text-align:center">
        <div style="font-size:2.5rem">📁</div>
        <div style="font-size:15px;color:#64748b;margin-top:6px;font-weight:600">Upload a CSV file to start batch prediction</div>
        <div style="font-size:13px;color:#94a3b8;margin-top:3px">Download the template above to get the correct format</div>
    </div>""", unsafe_allow_html=True)
