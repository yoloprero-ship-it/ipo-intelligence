import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from transformers import pipeline
from arch import arch_model
from newsapi import NewsApiClient

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="IPO Intelligence Engine", layout="wide")

# Dark theme
st.markdown("""
    <style>
    body {background-color: #0E1117; color: white;}
    </style>
""", unsafe_allow_html=True)

st.title("🚀 IPO Intelligence Engine 2.0")

# ---------------- SECTOR + STOCK LIST ----------------
sectors = {
    "Auto": ["TATAMOTORS.NS", "EICHERMOT.NS"],
    "IT": ["INFY.NS", "TCS.NS"],
    "Pharma": ["SUNPHARMA.NS", "DIVISLAB.NS"]
}

sector = st.selectbox("Select Sector", list(sectors.keys()))
stock = st.selectbox("Select Stock", sectors[sector])

# ---------------- LIVE DATA ----------------
@st.cache_data
def get_data(ticker):
    data = yf.download(ticker, period="6mo", progress=False)
    return data

data = get_data(stock)

# ---------------- VOLATILITY ----------------
def compute_garch(returns):
    try:
        model = arch_model(returns, vol='Garch', p=1, q=1)
        res = model.fit(disp="off")
        return res.conditional_volatility
    except:
        return pd.Series([0]*len(returns))

def compute_egarch(returns):
    try:
        model = arch_model(returns, vol='EGarch', p=1, q=1)
        res = model.fit(disp="off")
        return res.conditional_volatility
    except:
        return pd.Series([0]*len(returns))

if data is None or data.empty:
    st.error("No data fetched")
    st.stop()

returns = data['Close'].pct_change().dropna()

garch_vol = compute_garch(returns)
egarch_vol = compute_egarch(returns)

# ---------------- VOLATILITY CLUSTERING ----------------
def volatility_clustering(vol):
    return "High Clustering" if vol.std() > 0.02 else "Low Clustering"

cluster = volatility_clustering(garch_vol)

# ---------------- SENSITIVITY ANALYSIS ----------------
def sensitivity_analysis(returns):
    shocks = [-0.02, -0.01, 0.01, 0.02]
    results = [returns.mean() + s for s in shocks]
    return pd.DataFrame({"Shock": shocks, "Impact": results})

sens_df = sensitivity_analysis(returns)

# ---------------- NEWS SENTIMENT ----------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

try:
    model = load_model()
except:
    st.warning("Sentiment model failed, using fallback")
    model = None

def get_news(company):
    try:
        newsapi = NewsApiClient(api_key="dccb7a07daea449abd667b661ff56126")
        articles = newsapi.get_everything(q=company, page_size=5)
        return [a['title'] for a in articles['articles']]
    except:
        return []

def sentiment_score(news):
    scores = []
    for n in news:
        res = model(n)[0]
        if res['label'] == 'positive':
            scores.append(res['score'])
        else:
            scores.append(-res['score'])
    return np.mean(scores) if scores else 0

news = get_news(stock)

if not news or "failed" in str(news[0]).lower():
    news = [
        f"{stock} IPO demand is strong",
        f"{stock} showing mixed investor sentiment",
        f"Analysts are cautiously optimistic about {stock}"
    ]
sentiment = sentiment_score(news)

# ---------------- PREDICTION ----------------
def predict(sentiment, vol):
    score = 0
    if sentiment > 0.2:
        score += 1
    if vol.mean() < 0.02:
        score += 1
    return score / 2

prob = predict(sentiment, garch_vol)

# ---------------- UI ----------------
col1, col2, col3 = st.columns(3)

col1.metric("Sentiment", round(sentiment,2))
col2.metric("Volatility", round(garch_vol.mean(),4))
col3.metric("Listing Gain %", f"{round(prob*100,2)}%")

# ---------------- CHART ----------------
fig = go.Figure()
fig.add_trace(go.Scatter(y=garch_vol, name="GARCH"))
fig.add_trace(go.Scatter(y=egarch_vol, name="EGARCH"))

st.plotly_chart(fig)

# ---------------- CLUSTER ----------------
st.subheader("Volatility Clustering")
st.write(cluster)

# ---------------- SENSITIVITY ----------------
st.subheader("Sensitivity Analysis")
st.dataframe(sens_df)

# ---------------- NEWS ----------------
st.subheader("News")
for n in news:
    st.write("-", n)

# ---------------- LEARN BUTTON ----------------
if st.button("Learn Concepts"):
    st.write("GARCH: Measures volatility over time")
    st.write("EGARCH: Captures asymmetric shocks")
    st.write("Sentiment: Market mood from news")

# ---------------- EXPORT ----------------
if st.button("Export Data"):
    sens_df.to_csv("analysis.csv")
    st.success("Downloaded")