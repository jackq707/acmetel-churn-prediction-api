import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.supabase_client import fetch_all_predictions

st.set_page_config(page_title="Analytics", page_icon="🔬", layout="wide")

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
  <div style="background:linear-gradient(135deg,#1e3a5f,#2563a8);border-radius:12px;padding:10px 14px;font-size:1.8rem;line-height:1">🔬</div>
  <div>
    <div style="font-size:2rem;font-weight:900;color:#0f172a;line-height:1.1">Churn Analytics</div>
    <div style="font-size:14px;color:#94a3b8;margin-top:3px">Deep-dive analysis of churn drivers — AcmeTel Churn Prediction API</div>
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

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filters")
    st.markdown("---")
    date_range = st.date_input("📅 Date Range", value=[df["date"].min(), df["date"].max()], min_value=df["date"].min(), max_value=df["date"].max())
    source_filter = st.multiselect("📡 Source", options=df["source"].unique().tolist(), default=df["source"].unique().tolist())
    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

if len(date_range) == 2:
    df = df[(df["date"] >= date_range[0]) & (df["date"] <= date_range[1])]
if source_filter:
    df = df[df["source"].isin(source_filter)]

if df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────
total      = len(df)
churners   = int(df["churn_flag"].sum())
churn_rate = df["churn_flag"].mean() * 100
avg_prob   = df["churn_probability"].mean()
high_risk  = len(df[df["risk_level"] == "HIGH"])
senior_churn_rate = df[df["senior_citizen"]==1]["churn_flag"].mean() * 100 if df["senior_citizen"].notna().any() else 0

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("📋 Total Predictions", f"{total:,}")
c2.metric("🔴 Churn Rate",        f"{churn_rate:.1f}%")
c3.metric("📈 Avg Probability",   f"{avg_prob:.3f}")
c4.metric("⚡ High Risk",         f"{high_risk:,}")
c5.metric("👴 Senior Churn Rate", f"{senior_churn_rate:.1f}%")

st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)

TH = dict(plot_bgcolor="#f8fafc", paper_bgcolor="white",
          font=dict(family="Inter,sans-serif", size=13, color="#334155"),
          margin=dict(t=20,b=35,l=45,r=20))

# ── Row 1: Contract, Internet Service, Source ─────────────────────────────────
col1, col2, col3 = st.columns([1.6, 1.6, 1.2])

with col1:
    st.markdown('<div class="sh">📄 Churn Rate by Contract</div>', unsafe_allow_html=True)
    cdf = df.groupby("contract")["churn_flag"].agg(["mean","count"]).reset_index()
    cdf.columns = ["contract","rate","count"]
    cdf["pct"] = (cdf["rate"]*100).round(1)
    cdf = cdf.sort_values("pct", ascending=True)
    colors = ["#e74c3c" if p>=50 else "#f39c12" if p>=25 else "#2ecc71" for p in cdf["pct"]]
    fig1 = go.Figure(go.Bar(
        y=cdf["contract"], x=cdf["pct"], orientation="h",
        marker=dict(color=colors, line=dict(color="white", width=1)),
        text=[f"{p:.1f}%  (n={n})" for p,n in zip(cdf["pct"],cdf["count"])],
        textposition="outside", textfont=dict(size=12)
    ))
    fig1.update_layout(height=260, showlegend=False,
                       xaxis=dict(ticksuffix="%", gridcolor="#e2e8f0", range=[0,110], tickfont=dict(size=12)),
                       yaxis=dict(tickfont=dict(size=12)), **TH)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown('<div class="sh">🌐 Churn Rate by Internet Service</div>', unsafe_allow_html=True)
    idf = df.groupby("internet_service")["churn_flag"].agg(["mean","count"]).reset_index()
    idf.columns = ["internet_service","rate","count"]
    idf["pct"] = (idf["rate"]*100).round(1)
    idf = idf.sort_values("pct", ascending=True)
    colors2 = ["#e74c3c" if p>=50 else "#f39c12" if p>=25 else "#2ecc71" for p in idf["pct"]]
    fig2 = go.Figure(go.Bar(
        y=idf["internet_service"], x=idf["pct"], orientation="h",
        marker=dict(color=colors2, line=dict(color="white", width=1)),
        text=[f"{p:.1f}%  (n={n})" for p,n in zip(idf["pct"],idf["count"])],
        textposition="outside", textfont=dict(size=12)
    ))
    fig2.update_layout(height=260, showlegend=False,
                       xaxis=dict(ticksuffix="%", gridcolor="#e2e8f0", range=[0,110], tickfont=dict(size=12)),
                       yaxis=dict(tickfont=dict(size=12)), **TH)
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    st.markdown('<div class="sh">📡 Predictions by Source</div>', unsafe_allow_html=True)
    sdf = df["source"].value_counts().reset_index()
    sdf.columns = ["source","count"]
    fig3 = go.Figure(go.Pie(
        labels=sdf["source"], values=sdf["count"], hole=0.55,
        marker=dict(colors=["#2563a8","#7c3aed","#0891b2"], line=dict(color="white", width=2.5)),
        textfont=dict(size=13), textinfo="percent+label"
    ))
    fig3.update_layout(height=260, showlegend=False, margin=dict(t=20,b=10,l=0,r=0), paper_bgcolor="white")
    st.plotly_chart(fig3, use_container_width=True)

st.markdown('<hr style="border:1px solid #e2e8f0;margin:4px 0">', unsafe_allow_html=True)

# ── Row 2: Gender+Senior, Monthly Charges dist, Prob histogram ────────────────
col4, col5, col6 = st.columns([1.2, 1.8, 1.8])

with col4:
    st.markdown('<div class="sh">👤 Churn by Demographics</div>', unsafe_allow_html=True)
    rows = []
    for g in df["gender"].dropna().unique():
        sub = df[df["gender"]==g]
        rows.append({"label": g, "pct": sub["churn_flag"].mean()*100, "n": len(sub)})
    for val, lbl in [(1,"Senior"), (0,"Non-Senior")]:
        sub = df[df["senior_citizen"]==val]
        if len(sub) > 0:
            rows.append({"label": lbl, "pct": sub["churn_flag"].mean()*100, "n": len(sub)})
    ddf = pd.DataFrame(rows).sort_values("pct", ascending=True)
    colors3 = ["#e74c3c" if p>=50 else "#f39c12" if p>=25 else "#2ecc71" for p in ddf["pct"]]
    fig4 = go.Figure(go.Bar(
        y=ddf["label"], x=ddf["pct"], orientation="h",
        marker=dict(color=colors3, line=dict(color="white", width=1)),
        text=[f"{p:.1f}%" for p in ddf["pct"]],
        textposition="outside", textfont=dict(size=12)
    ))
    fig4.update_layout(height=280, showlegend=False,
                       xaxis=dict(ticksuffix="%", gridcolor="#e2e8f0", range=[0,110], tickfont=dict(size=12)),
                       yaxis=dict(tickfont=dict(size=12)), **TH)
    st.plotly_chart(fig4, use_container_width=True)

with col5:
    st.markdown('<div class="sh">💰 Monthly Charges Distribution</div>', unsafe_allow_html=True)
    fig5 = go.Figure()
    fig5.add_trace(go.Histogram(
        x=df[df["churn_flag"]==1]["monthly_charges"], name="Churn",
        marker_color="#e74c3c", opacity=0.75, nbinsx=25
    ))
    fig5.add_trace(go.Histogram(
        x=df[df["churn_flag"]==0]["monthly_charges"], name="Safe",
        marker_color="#2ecc71", opacity=0.75, nbinsx=25
    ))
    fig5.update_layout(
        barmode="overlay", height=280,
        legend=dict(orientation="h", y=0.98, x=0.02, font=dict(size=12),
                    bgcolor="rgba(255,255,255,0.8)"),
        xaxis=dict(title="Monthly Charges ($)", gridcolor="#e2e8f0", tickfont=dict(size=12)),
        yaxis=dict(gridcolor="#e2e8f0", tickfont=dict(size=12)), **TH
    )
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    st.markdown('<div class="sh">📊 Churn Probability Histogram</div>', unsafe_allow_html=True)
    fig6 = go.Figure()
    fig6.add_trace(go.Histogram(
        x=df[df["churn_flag"]==1]["churn_probability"], name="Churn",
        marker_color="#e74c3c", opacity=0.75, nbinsx=20, xbins=dict(start=0, end=1, size=0.05)
    ))
    fig6.add_trace(go.Histogram(
        x=df[df["churn_flag"]==0]["churn_probability"], name="Safe",
        marker_color="#2563a8", opacity=0.75, nbinsx=20, xbins=dict(start=0, end=1, size=0.05)
    ))
    fig6.add_vline(x=0.5, line_dash="dash", line_color="#94a3b8", line_width=1.5,
                   annotation_text="Threshold 0.5", annotation_font_size=11, annotation_font_color="#64748b")
    fig6.update_layout(
        barmode="overlay", height=280,
        legend=dict(orientation="h", y=0.98, x=0.02, font=dict(size=12),
                    bgcolor="rgba(255,255,255,0.8)"),
        xaxis=dict(title="Churn Probability", gridcolor="#e2e8f0", tickfont=dict(size=12)),
        yaxis=dict(gridcolor="#e2e8f0", tickfont=dict(size=12)), **TH
    )
    st.plotly_chart(fig6, use_container_width=True)
