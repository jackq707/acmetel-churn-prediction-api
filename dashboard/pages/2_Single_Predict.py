import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import httpx
import plotly.graph_objects as go

st.set_page_config(page_title="Single Predict — AcmeTel", page_icon="🔍", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:2.5rem!important;padding-bottom:0.5rem!important}
[data-testid="stSidebar"]{background:#0f172a!important}
[data-testid="stSidebar"] *{color:#e2e8f0!important;font-size:14px!important}
.sh{font-size:12px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;padding-bottom:3px;border-bottom:2px solid #1e293b}
.risk-high{background:linear-gradient(135deg,#991b1b,#dc2626);color:white;border-radius:12px;padding:16px;text-align:center}
.risk-medium{background:linear-gradient(135deg,#92400e,#d97706);color:white;border-radius:12px;padding:16px;text-align:center}
.risk-safe{background:linear-gradient(135deg,#14532d,#16a34a);color:white;border-radius:12px;padding:16px;text-align:center}
.factor-bad{background:#fee2e2;border-left:3px solid #dc2626;padding:5px 10px;border-radius:3px;margin:3px 0;font-size:13px}
.factor-good{background:#dcfce7;border-left:3px solid #16a34a;padding:5px 10px;border-radius:3px;margin:3px 0;font-size:13px}
div[data-testid="stSelectbox"] label, div[data-testid="stNumberInput"] label {font-size:13px!important}
div[data-testid="stSelectbox"] > div, div[data-testid="stNumberInput"] > div {font-size:13px!important}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
  <div style="background:linear-gradient(135deg,#1e3a5f,#2563a8);border-radius:10px;padding:8px 12px;font-size:1.6rem;line-height:1">🔍</div>
  <div>
    <div style="font-size:1.8rem;font-weight:900;color:#f1f5f9;line-height:1.1">Single Customer Predict</div>
    <div style="font-size:13px;color:#94a3b8;margin-top:2px">Input subscriber details and get instant churn prediction</div>
  </div>
</div>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "https://jackq707-acmetel-churn-api.hf.space")

col_form, col_result = st.columns([1.3, 1])

with col_form:
    # ── Row 1: Demographics ──────────────────────────────────────────────────
    st.markdown('<div class="sh">👤 Demographics</div>', unsafe_allow_html=True)
    d1,d2,d3,d4,d5 = st.columns(5)
    gender         = d1.selectbox("Gender",         ["Female","Male"],                    label_visibility="visible")
    senior_citizen = d2.selectbox("Senior",         ["No","Yes"],                         label_visibility="visible")
    partner        = d3.selectbox("Partner",        ["No","Yes"],                         label_visibility="visible")
    dependents     = d4.selectbox("Dependents",     ["No","Yes"],                         label_visibility="visible")
    tenure         = d5.number_input("Tenure (mo)", min_value=0, max_value=72, value=5)

    # ── Row 2: Charges ───────────────────────────────────────────────────────
    st.markdown('<div class="sh" style="margin-top:8px">💰 Charges</div>', unsafe_allow_html=True)
    ch1, ch2, ch3 = st.columns(3)
    monthly_charges = ch1.number_input("Monthly ($)",   min_value=0.0,   max_value=200.0,  value=80.0,  step=0.5)
    total_charges   = ch2.number_input("Total ($)",     min_value=0.0,   max_value=10000.0,value=400.0, step=10.0)
    contract        = ch3.selectbox("Contract",         ["Month-to-month","One year","Two year"])

    # ── Row 3: Services ──────────────────────────────────────────────────────
    st.markdown('<div class="sh" style="margin-top:8px">📱 Services</div>', unsafe_allow_html=True)
    s1,s2,s3,s4,s5 = st.columns(5)
    phone_service  = s1.selectbox("Phone",    ["Yes","No"])
    multiple_lines = s2.selectbox("Multi Lines",["No","Yes","No phone service"])
    internet       = s3.selectbox("Internet", ["Fiber optic","DSL","No"])
    online_sec     = s4.selectbox("Security", ["No","Yes","No internet service"])
    online_bak     = s5.selectbox("Backup",   ["No","Yes","No internet service"])

    s6,s7,s8,s9,s10 = st.columns(5)
    device_prot    = s6.selectbox("Device",   ["No","Yes","No internet service"])
    tech_support   = s7.selectbox("Support",  ["No","Yes","No internet service"])
    streaming_tv   = s8.selectbox("Stream TV",["No","Yes","No internet service"])
    streaming_mov  = s9.selectbox("Stream Mv",["No","Yes","No internet service"])
    paperless      = s10.selectbox("Paperless",["Yes","No"])

    # ── Row 4: Payment ───────────────────────────────────────────────────────
    st.markdown('<div class="sh" style="margin-top:8px">💳 Payment</div>', unsafe_allow_html=True)
    payment = st.selectbox("Payment Method", [
        "Electronic check","Mailed check",
        "Bank transfer (automatic)","Credit card (automatic)"
    ])

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Predict Churn", use_container_width=True, type="primary")

with col_result:
    st.markdown('<div class="sh">📊 Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        payload = {"data": {
            "gender": gender, "SeniorCitizen": 1 if senior_citizen=="Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet, "OnlineSecurity": online_sec,
            "OnlineBackup": online_bak, "DeviceProtection": device_prot,
            "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_mov, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges, "TotalCharges_clean": total_charges,
        }}

        with st.spinner("Predicting..."):
            try:
                resp   = httpx.post(f"{API_URL}/predict", json=payload, timeout=30)
                result = resp.json()
                prob   = result["churn_probability"]
                flag   = result["churn_flag"]

                if prob >= 0.70:
                    risk_label,risk_class,risk_color = "🔴 HIGH RISK",  "risk-high",   "#dc2626"
                elif prob >= 0.40:
                    risk_label,risk_class,risk_color = "🟡 MEDIUM RISK","risk-medium", "#d97706"
                else:
                    risk_label,risk_class,risk_color = "🟢 LOW RISK",   "risk-safe",   "#16a34a"

                st.markdown(f"""
                <div class="{risk_class}">
                    <div style="font-size:2.6rem;font-weight:900">{prob:.1%}</div>
                    <div style="font-size:1.1rem;font-weight:700;margin-top:2px">{risk_label}</div>
                    <div style="font-size:12px;opacity:.85">{'Will Churn' if flag==1 else 'Will Stay'}</div>
                </div>""", unsafe_allow_html=True)

                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob*100,
                    number=dict(suffix="%", font=dict(size=26, color=risk_color)),
                    gauge=dict(
                        axis=dict(range=[0,100], ticksuffix="%", tickfont=dict(size=11)),
                        bar=dict(color=risk_color, thickness=0.25),
                        bgcolor="#f1f5f9",
                        steps=[
                            dict(range=[0,40],  color="#dcfce7"),
                            dict(range=[40,70], color="#fef9c3"),
                            dict(range=[70,100],color="#fee2e2"),
                        ],
                        threshold=dict(line=dict(color=risk_color,width=3), thickness=0.8, value=prob*100)
                    )
                ))
                fig_g.update_layout(height=200, margin=dict(t=15,b=5,l=15,r=15), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_g, use_container_width=True)

                st.markdown('<div class="sh" style="margin-top:4px">⚠️ Risk Factors</div>', unsafe_allow_html=True)
                bad,good = [],[]
                if contract=="Month-to-month":      bad.append("Contract Month-to-month (66% churn)")
                elif contract=="Two year":          good.append("Contract Two year (3.3% churn)")
                else:                               good.append("One year contract — low churn risk")
                if internet=="Fiber optic":         bad.append("Fiber Optic — high churn rate (51.7%)")
                elif internet=="No":                good.append("No Internet (sangat rendah)")
                if tenure<=12:                      bad.append(f"Tenure {tenure} bln — new customer, high risk")
                elif tenure>=48:                    good.append(f"Tenure {tenure} mo — loyal customer")
                if monthly_charges>=70:             bad.append(f"Monthly Charges ${monthly_charges:.0f} — high")
                else:                               good.append(f"Monthly Charges ${monthly_charges:.0f} — moderate")
                if online_sec=="No" and internet!="No":   bad.append("No Online Security subscription")
                if tech_support=="No" and internet!="No": bad.append("No Tech Support subscription")
                if payment=="Electronic check":     bad.append("Electronic check — high churn risk")
                for f in bad:  st.markdown(f'<div class="factor-bad">❌ {f}</div>', unsafe_allow_html=True)
                for f in good: st.markdown(f'<div class="factor-good">✅ {f}</div>', unsafe_allow_html=True)

                st.markdown('<div class="sh" style="margin-top:8px">💡 Action</div>', unsafe_allow_html=True)
                if prob>=0.70:   st.error("**Immediate Action** — Contact within 24 hours, offer contract upgrade with discount")
                elif prob>=0.40: st.warning("**Monitor** — Send engagement email, offer add-on services")
                else:            st.success("**Loyal Customer** — Send appreciation message & referral program offer")

            except Exception as e:
                st.error(f"API Error: {e}")
    else:
        st.markdown("""
        <div style="background:#1e293b;border:2px dashed #334155;border-radius:12px;padding:60px 20px;text-align:center;margin-top:10px">
            <div style="font-size:3rem">🔍</div>
            <div style="font-size:15px;color:#64748b;margin-top:8px;font-weight:600">Fill in the form and click Predict Churn</div>
        </div>""", unsafe_allow_html=True)
