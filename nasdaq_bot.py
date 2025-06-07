# 📈 NASDAQ ANALYSIS BOT – con previsione e classificazione rischio
# Requisiti: yfinance, pandas, ta, scikit-learn, streamlit, plotly

import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.linear_model import LinearRegression
import streamlit as st
import plotly.graph_objs as go
import datetime

# -------------------------------
# FUNZIONI PRINCIPALI
# -------------------------------

def get_nasdaq_tickers():
    # Puoi sostituire con un CSV di tickers reali NASDAQ completi
    table = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")[4]
    return table['Ticker'].tolist()

def get_data(ticker):
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False)
        if len(df) < 20:
            return None
        df.dropna(inplace=True)
        return df
    except:
        return None

def add_indicators(df):
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['MACD_signal'] = ta.trend.MACD(df['Close']).macd_signal()
    df['Volatility'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    return df

def predict_growth(df):
    df = df.tail(10).copy()
    if len(df) < 5:
        return 0
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Close'].values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0][0]
    pct_growth = (slope / y.mean()) * 5  # stima crescita in % nei prossimi 5 giorni
    return round(pct_growth * 100, 2)

def classify_risk(df):
    vol = df['Volatility'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    if vol > 10 or rsi < 25 or rsi > 75:
        return "Super-Rischioso"
    elif vol > 5 or rsi < 35 or rsi > 65:
        return "Rischioso"
    else:
        return "Stabile"

def analyze_ticker(ticker):
    df = get_data(ticker)
    if df is None:
        return None
    df = add_indicators(df)
    if df.isna().any().any():
        return None
    prediction = predict_growth(df)
    risk = classify_risk(df)
    signal = "BUY" if df['RSI'].iloc[-1] < 35 and df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1] else "HOLD"
    return {
        'Ticker': ticker,
        'Signal': signal,
        'Prediction (%)': prediction,
        'Risk': risk,
        'Price': round(df['Close'].iloc[-1], 2)
    }

# -------------------------------
# STREAMLIT DASHBOARD
# -------------------------------

st.set_page_config(page_title="NASDAQ Investment Bot", layout="wide")
st.title("📈 NASDAQ Investment Bot – Rischio + Previsioni")
st.write("Analisi automatica dei titoli NASDAQ, classificazione rischio e previsioni di crescita.")

nasdaq_tickers = get_nasdaq_tickers()
results = []

progress = st.progress(0)
for i, ticker in enumerate(nasdaq_tickers):
    res = analyze_ticker(ticker)
    if res:
        results.append(res)
    progress.progress((i + 1) / len(nasdaq_tickers))

if results:
    df_res = pd.DataFrame(results)
    risk_level = st.selectbox("Filtra per rischio:", ["Tutti"] + sorted(df_res['Risk'].unique()))
    if risk_level != "Tutti":
        df_res = df_res[df_res['Risk'] == risk_level]

    st.dataframe(df_res.sort_values(by='Prediction (%)', ascending=False))

    selected = st.selectbox("📊 Vedi dettagli per titolo:", df_res['Ticker'].tolist())
    df_chart = get_data(selected)
    df_chart = add_indicators(df_chart)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Close'], name='Close'))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA50'], name='SMA50'))
    fig.update_layout(title=f"Andamento e SMA - {selected}", xaxis_title="Data", yaxis_title="Prezzo")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Nessun titolo analizzato con successo. Riprova più tardi.")
