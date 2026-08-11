import streamlit as st
import pandas as pd
import joblib
import requests
import plotly.graph_objects as go
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="AgriPrice Intelligence", page_icon="🌾", layout="wide")

# ============================================================
# CUSTOM STYLING — "Harvest Field" theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --forest: #1B4332;
    --forest-light: #2D6A4F;
    --amber: #E9A23B;
    --amber-light: #F4C978;
    --clay: #B3541E;
    --cream: #F7F5EF;
    --charcoal: #1A2E22;
    --sage: #E8EFE6;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--charcoal);
}

h1, h2, h3 {
    font-family: 'Fraunces', serif !important;
    color: var(--forest) !important;
}

.stApp {
    background: linear-gradient(180deg, #FAFAF7 0%, #F2F0E8 100%);
}

/* Hero banner */
.hero {
    background: linear-gradient(120deg, var(--forest) 0%, var(--forest-light) 100%);
    padding: 2.2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute;
    right: -40px; top: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(233,162,59,0.35) 0%, transparent 70%);
}
.hero h1 {
    color: #FAFAF7 !important;
    font-size: 2.4rem;
    margin-bottom: 0.2rem;
    font-weight: 600 !important;
}
.hero p {
    color: var(--amber-light);
    font-size: 1.05rem;
    margin: 0;
    letter-spacing: 0.3px;
}

/* Metric / info cards */
.metric-card {
    background: white;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 2px 12px rgba(27,67,50,0.07);
    border-left: 5px solid var(--amber);
    height: 100%;
}
.metric-card .label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #7A8B7F;
    font-weight: 600;
}
.metric-card .value {
    font-family: 'Fraunces', serif;
    font-size: 1.7rem;
    color: var(--forest);
    font-weight: 600;
    margin-top: 0.2rem;
}
.metric-card .sub {
    font-size: 0.82rem;
    color: #6B6B6B;
    margin-top: 0.3rem;
}

.risk-badge {
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.3px;
}
.risk-low { background: #E1F0E5; color: #1B4332; }
.risk-medium { background: #FCEACB; color: #8A5A0C; }
.risk-high { background: #F6DCD0; color: var(--clay); }

.section-divider {
    height: 1px;
    background: linear-gradient(90deg, var(--amber) 0%, transparent 100%);
    margin: 1.8rem 0 1.2rem 0;
}

.market-row {
    background: white;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 6px rgba(27,67,50,0.06);
}
.market-row.top {
    border-left: 4px solid var(--amber);
    background: var(--sage);
}

.stButton > button {
    background: var(--forest) !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.8rem !important;
    font-weight: 600 !important;
    border: none !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover {
    background: var(--forest-light) !important;
}

.footer-note {
    text-align: center;
    color: #8A8A8A;
    font-size: 0.82rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #E5E5E5;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD ASSETS
# ============================================================
@st.cache_resource
def load_model_assets():
    model = joblib.load('models/price_prediction_model.pkl')
    le_commodity = joblib.load('models/le_commodity.pkl')
    le_market = joblib.load('models/le_market.pkl')
    le_district = joblib.load('models/le_district.pkl')
    le_season = joblib.load('models/le_season.pkl')
    return model, le_commodity, le_market, le_district, le_season

@st.cache_data
def load_historical_data():
    df = pd.read_csv('data/processed/agriprice_features.csv')
    df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])
    return df

@st.cache_data
def load_volatility():
    return pd.read_csv('data/processed/crop_volatility_final.csv')

model, le_commodity, le_market, le_district, le_season = load_model_assets()
df = load_historical_data()
volatility_df = load_volatility()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_KEY
BASE_URL = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
HEADERS = {'User-Agent': 'Mozilla/5.0'}

def fetch_live_price(state, district, commodity, limit=10):
    params = {
        'api-key': API_KEY, 'format': 'json', 'limit': limit,
        'filters[State]': state, 'filters[District]': district, 'filters[Commodity]': commodity
    }
    try:
        response = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=20)
        if response.status_code != 200:
            return None
        records = response.json().get('records', [])
        if not records:
            return None
        temp_df = pd.DataFrame(records)
        temp_df['Arrival_Date_parsed'] = pd.to_datetime(temp_df['Arrival_Date'], format='%d/%m/%Y', errors='coerce')
        temp_df = temp_df.sort_values('Arrival_Date_parsed', ascending=False)
        latest = temp_df.iloc[0]
        days_old = (datetime.now() - latest['Arrival_Date_parsed']).days
        return {
            'Market': latest['Market'], 'Date': latest['Arrival_Date'],
            'Modal_Price': latest['Modal_Price'], 'Days_Old': days_old,
            'Is_Recent': days_old <= 90
        }
    except Exception:
        return None

def recommend_markets(crop, top_n=5):
    crop_df = df[df['Commodity'] == crop].copy()
    if crop_df.empty:
        return None
    latest_per_market = crop_df.sort_values('Arrival_Date').groupby('Market').tail(1).copy()
    latest_per_market['Commodity_enc'] = le_commodity.transform(latest_per_market['Commodity'])
    latest_per_market['Market_enc'] = le_market.transform(latest_per_market['Market'])
    latest_per_market['District_enc'] = le_district.transform(latest_per_market['District'])
    latest_per_market['Season_enc'] = le_season.transform(latest_per_market['Season'])
    feature_cols = ['Commodity_enc', 'Market_enc', 'District_enc', 'Season_enc',
                     'Month', 'DayOfWeek', 'lag_price_1', 'rolling_avg_7']
    X = latest_per_market[feature_cols]
    latest_per_market['Predicted_Price'] = model.predict(X)
    result = latest_per_market[['Market', 'District', 'Modal_Price', 'Predicted_Price']].sort_values(
        'Predicted_Price', ascending=False).head(top_n)
    return result.rename(columns={'Modal_Price': 'Last_Known_Price'})

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🌾 AgriPrice Intelligence</h1>
    <p>Smart market analytics for farmers — know where to sell, and when.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INPUT ROW
# ============================================================
crops = sorted(df['Commodity'].unique())
districts = sorted(df['District'].unique())

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    selected_crop = st.selectbox("🌱 Select your crop", crops)
with col2:
    selected_district = st.selectbox("📍 Select your district", districts)
with col3:
    st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
    go_btn = st.button("Get Recommendation →", type="primary", use_container_width=True)

if go_btn:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    state_lookup = df[df['District'] == selected_district]['State'].iloc[0]
    crop_df_all = df[df['Commodity'] == selected_crop].copy()

    with st.spinner("Fetching latest market data..."):
        live = fetch_live_price(state_lookup, selected_district, selected_crop)

    risk_row = volatility_df[volatility_df['Commodity'] == selected_crop]
    risk_level = risk_row.iloc[0]['Risk_Level'] if not risk_row.empty else "Unknown"
    risk_class = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}.get(risk_level, "risk-medium")

    recs = recommend_markets(selected_crop, top_n=5)
    top_price = recs.iloc[0]['Predicted_Price'] if recs is not None else None

    # ---- Metric cards ----
    m1, m2, m3 = st.columns(3)

    with m1:
        if live and live['Is_Recent']:
            price_display = f"₹{live['Modal_Price']}"
            sub = f"at {live['Market']} · {live['Date']}"
        elif live:
            price_display = f"₹{live['Modal_Price']}"
            if live['Days_Old'] > 365:
                sub = "⚠️ No recent data available for this market"
            else:
                sub = f"⚠️ {live['Days_Old']} days old — may be outdated"
        else:
            price_display = "—"
            sub = "No live data found"
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Latest Reported Price</div>
            <div class="value">{price_display}<span style="font-size:0.9rem; color:#7A8B7F;">/quintal</span></div>
            <div class="sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        pred_display = f"₹{top_price:.0f}" if top_price else "—"
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Best Predicted Price</div>
            <div class="value">{pred_display}<span style="font-size:0.9rem; color:#7A8B7F;">/quintal</span></div>
            <div class="sub">at top recommended market</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Price Risk Level</div>
            <div class="value"><span class="risk-badge {risk_class}">{risk_level}</span></div>
            <div class="sub">based on historical volatility</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ---- Market recommendations ----
    st.markdown("### 🏪 Best Markets to Sell In")
    if recs is not None:
        for i, row in recs.reset_index(drop=True).iterrows():
            cls = "market-row top" if i == 0 else "market-row"
            badge = "⭐ TOP PICK" if i == 0 else ""
            st.markdown(f"""
            <div class="{cls}">
                <div><b>{row['Market']}</b> &nbsp;·&nbsp; {row['District']} &nbsp; <span style="color:var(--amber); font-weight:600;">{badge}</span></div>
                <div>Last: ₹{row['Last_Known_Price']:.0f} &nbsp;→&nbsp; <b style="color:var(--forest);">Predicted: ₹{row['Predicted_Price']:.0f}</b></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Not enough data to recommend markets for this crop.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ---- Price trend chart ----
    st.markdown("### 📈 Recent Price Trend")
    trend_data = crop_df_all.sort_values('Arrival_Date').groupby(
        crop_df_all['Arrival_Date'].dt.to_period('M')
    )['Modal_Price'].mean().reset_index()
    trend_data['Arrival_Date'] = trend_data['Arrival_Date'].astype(str)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_data['Arrival_Date'], y=trend_data['Modal_Price'],
        mode='lines+markers', line=dict(color='#1B4332', width=3),
        marker=dict(color='#E9A23B', size=8),
        fill='tozeroy', fillcolor='rgba(233,162,59,0.08)'
    ))
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Inter', color='#1A2E22'),
        xaxis_title=None, yaxis_title="Avg Modal Price (₹/quintal)",
        margin=dict(l=10, r=10, t=10, b=10), height=320,
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#F0F0F0')
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("⚠️ Predictions are model-based estimates using historical patterns and recent price trends. Actual prices may vary due to weather, policy, and other unpredictable factors.")

st.markdown("""
<div class="footer-note">
    Data source: data.gov.in (Agmarknet) &nbsp;·&nbsp; Built for Project Presentation Showcase 2026
</div>
""", unsafe_allow_html=True)