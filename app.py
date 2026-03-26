import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="IPO Intelligence Engine", layout="wide")

st.title("🚀 IPO Intelligence Engine")

# ---------------- LOAD CSV ----------------
@st.cache_data
def load_data():
    return pd.read_csv("stocks_pro.csv")

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Filters")

sector = st.sidebar.selectbox("Select Sector", df["Sector"].unique())
filtered = df[df["Sector"] == sector]

company = st.sidebar.selectbox("Select Company", filtered["Company"])
ticker = filtered[filtered["Company"] == company]["Ticker"].values[0]

# ---------------- FETCH DATA ----------------
@st.cache_data
def get_data(ticker):
    try:
        data = yf.download(ticker, period="6mo", progress=False)

        if data is None or data.empty:
            data = yf.download("^NSEI", period="6mo", progress=False)

        return data
    except:
        return None

data = get_data(ticker)

if data is None or data.empty:
    st.error("No data available")
    st.stop()

# ---------------- VOLATILITY (ONLY SELECTED STOCK) ----------------
def calculate_volatility(data):
    returns = data['Close'].pct_change().dropna()
    if len(returns) == 0:
        return 0
    return float(returns.std())

volatility = calculate_volatility(data)

# ---------------- RETURNS ----------------
def calculate_returns(data):
    if len(data) < 2:
        return 0
    return float((data['Close'][-1] - data['Close'][0]) / data['Close'][0])

returns = calculate_returns(data)

# ---------------- METRICS ----------------
col1, col2 = st.columns(2)

col1.metric("📊 Volatility", round(volatility, 4))
col2.metric("📈 6M Return", f"{round(returns*100,2)}%")

# ---------------- CHART ----------------
st.subheader("📈 Price Chart")

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Price"))

st.plotly_chart(fig)

# ---------------- SIMPLE PREDICTION ----------------
def predict(vol, ret):
    score = 0
    if ret > 0:
        score += 1
    if vol < 0.03:
        score += 1
    return score / 2

prob = predict(volatility, returns)

st.subheader("🎯 Listing Gain Probability")
st.progress(int(prob * 100))
st.write(f"{round(prob * 100,2)}%")

# ---------------- SECTOR HEATMAP ----------------
st.subheader("🌡️ Sector Performance")

@st.cache_data
def sector_performance(df):
    perf = []
    for t in df["Ticker"].head(15):  # limit for speed
        try:
            d = yf.download(t, period="1mo", progress=False)
            r = (d['Close'][-1] - d['Close'][0]) / d['Close'][0]
            perf.append(r)
        except:
            perf.append(0)
    df2 = df.head(15).copy()
    df2["Returns"] = perf
    return df2.groupby("Sector")["Returns"].mean()

sector_perf = sector_performance(df)

st.bar_chart(sector_perf)

# ---------------- LEARN ----------------
if st.button("📚 Learn Concepts"):
    st.write("Volatility = risk level of stock")
    st.write("Returns = price change over time")
    st.write("Prediction = simple logic based")

# ---------------- EXPORT ----------------
if st.button("📤 Export Data"):
    df.to_csv("analysis.csv", index=False)
    st.success("Downloaded successfully")
