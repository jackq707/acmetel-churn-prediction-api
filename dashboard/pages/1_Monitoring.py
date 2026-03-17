import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from utils.supabase_client import fetch_all_predictions

st.set_page_config(page_title="Monitoring", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:2.5rem!important;padding-bottom:0.5rem!important}
[data-testid="metric-container"]{background:linear-gradient(135deg,#1e3a5f,#2563a8);border-radius:12px;padding:16px 20px!important;box-shadow:0 4px 12px rgba(37,99,168,.25)}
[data-testid="metric-container"] label{color:#93c5fd!important;font-size:14px!important;font-weight:700!important;text-transform:uppercase;letter-spacing:.05em}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#fff!important;font-size:2.4rem!important;font-weight:800!important}
[data-testid="metric-container"] [data-testid="stMetricDelta"]{color:#86efac!important;font-size:14px!important}
[data-testid="stSidebar"]{background:#0f172a!important}
[data-testid="stSidebar"] *{color:#e2e8f0!important;font-size:14px!important}
[data-testid="stSidebar"] .stButton button{background:#2563a8!important;color:white!important;border:none!important;border-radius:8px!important}
.sh{font-size:14px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;padding-bottom:4px;border-bottom:2px solid #e2e8f0}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:10px">
  <div style="background:linear-gradient(135deg,#1e3a5f,#2563a8);border-radius:12px;padding:10px 14px;font-size:1.8rem;line-height:1">📊</div>
  <div>
    <div style="font-size:2rem;font-weight:900;color:#0f172a;line-height:1.1">Monitoring Dashboard</div>
    <div style="font-size:14px;color:#94a3b8;margin-top:3px">Real-time churn prediction monitoring — AcmeTel Churn Prediction API</div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading..."):
    raw = fetch_all_predictions()

if not raw:
    st.warning("No prediction data yet.")
    st.stop()

df = pd.DataFrame(raw)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date

with st.sidebar:
    st.markdown("### 🔍 Filters")
    st.markdown("---")
    date_range = st.date_input("📅 Date Range", value=[df["date"].min(), df["date"].max()], min_value=df["date"].min(), max_value=df["date"].max())
    risk_filter = st.multiselect("⚡ Risk Level", options=["HIGH","MEDIUM","SAFE"], default=["HIGH","MEDIUM","SAFE"])
    source_filter = st.multiselect("📡 Source", options=df["source"].unique().tolist(), default=df["source"].unique().tolist())
    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    st.markdown(f"<div style='font-size:12px;color:#94a3b8;text-align:center'>Updated: {datetime.now().strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)

if len(date_range) == 2:
    df = df[(df["date"] >= date_range[0]) & (df["date"] <= date_range[1])]
if risk_filter:
    df = df[df["risk_level"].isin(risk_filter)]
if source_filter:
    df = df[df["source"].isin(source_filter)]

if df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

total = len(df)
churners = int(df["churn_flag"].sum())
safe = total - churners
churn_rate = df["churn_flag"].mean() * 100
avg_prob = df["churn_probability"].mean()
high_risk = len(df[df["risk_level"] == "HIGH"])

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("📋 Total Predictions", f"{total:,}")
c2.metric("🔴 Predicted Churn", f"{churners:,}", f"{churn_rate:.1f}%")
c3.metric("🟢 Predicted Safe", f"{safe:,}", f"{100-churn_rate:.1f}%")
c4.metric("⚡ High Risk", f"{high_risk:,}")
c5.metric("📈 Avg Probability", f"{avg_prob:.3f}")

st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)

TH = dict(plot_bgcolor="#f8fafc", paper_bgcolor="white", font=dict(family="Inter,sans-serif", size=13, color="#334155"), margin=dict(t=20,b=35,l=45,r=20))

col1,col2,col3 = st.columns([2.2,1.4,2.0])

with col1:
    st.markdown('<div class="sh">📈 Churn Rate Trend</div>', unsafe_allow_html=True)
    df["date_str"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    daily = df.groupby("date_str").agg(total=("churn_flag","count"), churned=("churn_flag","sum")).reset_index()
    daily["pct"] = (daily["churned"]/daily["total"]*100).round(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date_str"], y=daily["pct"], mode="lines+markers", line=dict(color="#e74c3c",width=2.5), marker=dict(size=8,color="#e74c3c",line=dict(width=2,color="white")), fill="tozeroy", fillcolor="rgba(231,76,60,0.08)"))
    fig.add_hline(y=churn_rate, line_dash="dash", line_color="#94a3b8", line_width=1.5, annotation_text=f"Avg {churn_rate:.1f}%", annotation_font_size=12, annotation_font_color="#64748b")

    fig.update_layout(height=300, showlegend=False, yaxis=dict(ticksuffix="%",gridcolor="#e2e8f0",zeroline=False,tickfont=dict(size=13)), xaxis=dict(gridcolor="#e2e8f0",tickfont=dict(size=13),tickangle=0,tickformat="%b %d"), margin=dict(t=20,b=35,l=45,r=20), plot_bgcolor="#f8fafc", paper_bgcolor="white", font=dict(family="Inter,sans-serif",size=13,color="#334155"))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<div class="sh">🎯 Risk Distribution</div>', unsafe_allow_html=True)
    rc = df["risk_level"].value_counts().reset_index()
    rc.columns = ["risk_level","count"]
    cm = {"HIGH":"#e74c3c","MEDIUM":"#f39c12","SAFE":"#2ecc71"}
    fig2 = go.Figure(go.Pie(labels=rc["risk_level"], values=rc["count"], hole=0.58, marker=dict(colors=[cm.get(r,"#aaa") for r in rc["risk_level"]], line=dict(color="white",width=2.5)), textfont=dict(size=13), textinfo="percent+label"))
    fig2.update_layout(height=250, showlegend=False, margin=dict(t=20,b=10,l=0,r=0), paper_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    st.markdown('<div class="sh">📋 Churn by Segment</div>', unsafe_allow_html=True)
    frames = []
    if "contract" in df.columns and df["contract"].notna().any():
        cdf = df.groupby("contract")["churn_flag"].mean().reset_index()
        cdf.columns = ["label","rate"]
        cdf["pct"] = (cdf["rate"]*100).round(1)
        frames.append(cdf)
    if "internet_service" in df.columns and df["internet_service"].notna().any():
        idf = df.groupby("internet_service")["churn_flag"].mean().reset_index()
        idf.columns = ["label","rate"]
        idf["pct"] = (idf["rate"]*100).round(1)
        frames.append(idf)
    if frames:
        combined = pd.concat(frames).sort_values("pct", ascending=True)
        bc = ["#e74c3c" if p>=70 else "#f39c12" if p>=30 else "#2ecc71" for p in combined["pct"]]
        fig3 = go.Figure(go.Bar(y=combined["label"], x=combined["pct"], orientation="h", marker=dict(color=bc, line=dict(color="white",width=1)), text=[f"{p:.1f}%" for p in combined["pct"]], textposition="outside", textfont=dict(size=13)))
        fig3.update_layout(height=300, showlegend=False, xaxis=dict(ticksuffix="%",gridcolor="#e2e8f0",range=[0,105],tickfont=dict(size=13)), yaxis=dict(tickfont=dict(size=13)), **TH)
        st.plotly_chart(fig3, use_container_width=True)

st.markdown('<hr style="border:1px solid #e2e8f0;margin:4px 0">', unsafe_allow_html=True)

col_l, col_r = st.columns([2.2, 3.6])

with col_l:
    st.markdown('<div class="sh">📅 Tenure Distribution</div>', unsafe_allow_html=True)
    if "tenure" in df.columns and df["tenure"].notna().any():
        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(x=df[df["churn_flag"]==1]["tenure"], name="Churn", marker_color="#e74c3c", opacity=0.78, nbinsx=20))
        fig4.add_trace(go.Histogram(x=df[df["churn_flag"]==0]["tenure"], name="Safe",  marker_color="#2ecc71", opacity=0.78, nbinsx=20))
        fig4.update_layout(barmode="overlay", height=300, legend=dict(orientation="h",y=0.98,x=0.02,font=dict(size=13),bgcolor="rgba(255,255,255,0.8)",bordercolor="rgba(0,0,0,0)"), xaxis=dict(title="Tenure (months)",gridcolor="#e2e8f0",tickfont=dict(size=13)), yaxis=dict(gridcolor="#e2e8f0",tickfont=dict(size=13)), **TH)
        st.plotly_chart(fig4, use_container_width=True)

with col_r:
    st.markdown('<div class="sh">🕐 Recent Predictions</div>', unsafe_allow_html=True)
    dcols = ["timestamp","source","gender","tenure","contract","internet_service","monthly_charges","churn_probability","churn_flag","risk_level"]
    avail = [c for c in dcols if c in df.columns]
    dfd = df[avail].head(8).copy()
    dfd["timestamp"] = dfd["timestamp"].dt.strftime("%m-%d %H:%M")
    if "churn_probability" in dfd.columns:
        dfd["churn_probability"] = dfd["churn_probability"].map("{:.4f}".format)
    def cr(val):
        return {"HIGH":"background-color:#fee2e2;color:#991b1b;font-weight:700","MEDIUM":"background-color:#fef9c3;color:#854d0e;font-weight:700","SAFE":"background-color:#dcfce7;color:#166534;font-weight:700"}.get(val,"")
    st.dataframe(dfd.style.applymap(cr, subset=["risk_level"]), use_container_width=True, hide_index=True, height=280)
