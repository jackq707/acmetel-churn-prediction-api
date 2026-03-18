import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import httpx
import pandas as pd
import io
import base64

st.set_page_config(page_title="Batch Predict", page_icon="📁", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:3rem!important;padding-bottom:0.2rem!important}
[data-testid="stSidebar"]{background:#0f172a!important}
[data-testid="stSidebar"] *{color:#e2e8f0!important;font-size:14px!important}
.sh{font-size:12px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;padding-bottom:3px;border-bottom:2px solid #e2e8f0}
.sb{border-radius:8px;padding:8px 4px;text-align:center;font-weight:700}
[data-testid="stFileUploaderDropzone"] button {font-size:14px!important}
[data-testid="stFileUploaderDropzoneInstructions"] div[data-testid="stMarkdownContainer"] p {font-size:14px!important;font-weight:600!important;color:#262730!important}
[data-testid="stFileUploaderDropzoneInstructions"] small {font-size:12px!important;color:#808495!important}

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
    st.markdown('<div class="sh">Step 1 — Template</div>', unsafe_allow_html=True)
    tpl = pd.DataFrame({
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
    })
    csv_data = tpl.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv_data).decode()
    st.markdown(f"""
    <div style="background:#f0f2f6;border-radius:8px;padding:16px 20px;display:flex;align-items:center;justify-content:space-between;min-height:72px">
        <div style="display:flex;align-items:center;gap:12px">
            <span style="font-size:1.6rem">📥</span>
            <div>
                <div style="font-weight:600;color:#262730;font-size:14px">Download CSV Template</div>
                <div style="font-size:12px;color:#808495">2 sample rows • CSV format</div>
            </div>
        </div>
        <a href="data:text/csv;base64,{b64}" download="acmetel_batch_template.csv"
           style="background:white;color:#262730;border:1px solid #d0d0d0;border-radius:6px;padding:6px 18px;font-size:14px;text-decoration:none;white-space:nowrap;font-family:sans-serif">
            Download
        </a>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown('<div class="sh">Step 2 — Upload CSV</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

with c3:
    st.markdown('<div class="sh">Step 3 — Validation</div>', unsafe_allow_html=True)
    if uploaded:
        uploaded.seek(0)
        df_chk = pd.read_csv(uploaded)
        miss = [c for c in REQUIRED_COLS if c not in df_chk.columns]
        if miss:
            st.error(f"Missing {len(miss)} columns")
        else:
            st.success(f"✅ {len(df_chk)} rows ready")
            st.caption(f"All {len(REQUIRED_COLS)} columns found")
    else:
        st.markdown("""
        <div style="background:#f0f2f6;border-radius:8px;padding:16px 20px;min-height:72px;display:flex;align-items:center;gap:12px">
            <span style="font-size:1.6rem">✅</span>
            <div>
                <div style="font-weight:600;color:#262730;font-size:14px">Upload a CSV file to validate</div>
                <div style="font-size:12px;color:#808495">All 19 required columns will be checked</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr style="border:1px solid #e2e8f0;margin:6px 0">', unsafe_allow_html=True)

if uploaded:
    uploaded.seek(0)
    df_input = pd.read_csv(uploaded)
    miss2 = [c for c in REQUIRED_COLS if c not in df_input.columns]
    if miss2:
        st.error(f"Missing columns: {miss2}")
        st.stop()

    st.markdown(f'<div class="sh">Preview ({len(df_input)} rows)</div>', unsafe_allow_html=True)
    st.dataframe(df_input.head(3), use_container_width=True, hide_index=True, height=130)

    if st.button("Run Batch Prediction", use_container_width=True, type="primary"):
        records = df_input[REQUIRED_COLS].to_dict(orient="records")
        payload = {"items": [{"data": r} for r in records]}
        with st.spinner(f"Predicting {len(records)} customers..."):
            try:
                resp = httpx.post(f"{API_URL}/predict_batch", json=payload, timeout=60, headers={"X-API-Key": os.getenv("API_KEY", "acmetel-dev-key-2026")})
                res  = resp.json()["results"]
                df_r = df_input.copy()
                df_r["churn_probability"] = [r["churn_probability"] for r in res]
                df_r["churn_flag"]        = [r["churn_flag"] for r in res]
                df_r["risk_level"]        = df_r["churn_probability"].apply(
                    lambda p: "HIGH" if p >= 0.70 else "MEDIUM" if p >= 0.40 else "SAFE"
                )

                cs, cr = st.columns([1, 2])
                with cs:
                    st.markdown('<div class="sh" style="margin-top:4px">Summary</div>', unsafe_allow_html=True)
                    tot  = len(df_r)
                    chu  = int(df_r["churn_flag"].sum())
                    saf  = tot - chu
                    hi   = len(df_r[df_r["risk_level"] == "HIGH"])
                    med  = len(df_r[df_r["risk_level"] == "MEDIUM"])
                    avgp = df_r["churn_probability"].mean()

                    a1, a2 = st.columns(2)
                    a1.markdown(f'<div class="sb" style="background:#dbeafe;color:#1e40af">Total<br><b>{tot}</b></div>', unsafe_allow_html=True)
                    a2.markdown(f'<div class="sb" style="background:#fee2e2;color:#991b1b">Churn<br><b>{chu}</b></div>', unsafe_allow_html=True)
                    b1, b2 = st.columns(2)
                    b1.markdown(f'<div class="sb" style="background:#dcfce7;color:#14532d">Safe<br><b>{saf}</b></div>', unsafe_allow_html=True)
                    b2.markdown(f'<div class="sb" style="background:#fee2e2;color:#991b1b">High Risk<br><b>{hi}</b></div>', unsafe_allow_html=True)
                    c1b, c2b = st.columns(2)
                    c1b.markdown(f'<div class="sb" style="background:#fef9c3;color:#854d0e">Medium<br><b>{med}</b></div>', unsafe_allow_html=True)
                    c2b.markdown(f'<div class="sb" style="background:#f1f5f9;color:#334155">Avg Prob<br><b>{avgp:.3f}</b></div>', unsafe_allow_html=True)

                    st.markdown('<div class="sh" style="margin-top:8px">Download</div>', unsafe_allow_html=True)
                    st.download_button("📥 CSV", data=df_r.to_csv(index=False).encode("utf-8"),
                        file_name="churn_predictions.csv", mime="text/csv", use_container_width=True)
                    xbuf = io.BytesIO()
                    with pd.ExcelWriter(xbuf, engine="openpyxl") as w:
                        df_r.to_excel(w, index=False, sheet_name="Predictions")
                    st.download_button("📊 Excel", data=xbuf.getvalue(),
                        file_name="churn_predictions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True)

                with cr:
                    st.markdown('<div class="sh" style="margin-top:4px">Prediction Results</div>', unsafe_allow_html=True)
                    scols = ["gender","tenure","Contract","InternetService",
                             "MonthlyCharges","churn_probability","churn_flag","risk_level"]
                    av  = [c for c in scols if c in df_r.columns]
                    dfs = df_r[av].copy()
                    dfs["churn_probability"] = dfs["churn_probability"].map("{:.4f}".format)
                    dff = dfs

                    def cr_style(val):
                        return {
                            "HIGH":   "background-color:#fee2e2;color:#991b1b;font-weight:700",
                            "MEDIUM": "background-color:#fef9c3;color:#854d0e;font-weight:700",
                            "SAFE":   "background-color:#dcfce7;color:#166534;font-weight:700"
                        }.get(val, "")

                    st.dataframe(dff.style.applymap(cr_style, subset=["risk_level"]),
                        use_container_width=True, hide_index=True, height=300)

            except Exception as e:
                st.error(f"API Error: {e}")
else:
    st.markdown("""
    <div style="background:#f8fafc;border:2px dashed #cbd5e1;border-radius:12px;padding:30px;text-align:center">
        <div style="font-size:2.5rem">📁</div>
        <div style="font-size:15px;color:#64748b;margin-top:6px;font-weight:600">Upload a CSV file to start batch prediction</div>
        <div style="font-size:13px;color:#94a3b8;margin-top:3px">Download the template above to get the correct format</div>
    </div>
    """, unsafe_allow_html=True)
