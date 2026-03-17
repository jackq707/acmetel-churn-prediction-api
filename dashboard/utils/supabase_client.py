import os
import streamlit as st
from supabase import create_client

@st.cache_resource
def get_supabase():
    """Get cached Supabase client — reads from st.secrets or env vars."""
    url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
    key = st.secrets.get("SUPABASE_PUBLISHABLE_KEY", os.getenv("SUPABASE_PUBLISHABLE_KEY"))
    if not url or not key:
        st.error("Supabase credentials not configured.")
        st.stop()
    return create_client(url, key)

def fetch_all_predictions():
    """Fetch all predictions from Supabase."""
    client = get_supabase()
    result = client.table("predictions").select("*").order("timestamp", desc=True).execute()
    return result.data

def fetch_summary():
    """Fetch summary stats from API /logs/summary endpoint."""
    import httpx, os
    api_url = st.secrets.get("API_URL", os.getenv("API_URL", "https://jackq707-acmetel-churn-api.hf.space"))
    try:
        resp = httpx.get(f"{api_url}/logs/summary", timeout=10)
        return resp.json()
    except Exception as e:
        return None
