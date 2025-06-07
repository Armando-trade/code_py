import yfinance as yf
import pandas as pd
import numpy as np
import talib
from sklearn.linear_model import LinearRegression
import streamlit as st
import plotly.graph_objs as go

def get_nasdaq_tickers():
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")[4]
        return table['Ticker'].tolist()
    except Exception as e:
        st.error(f"Errore caricamento tickers NASDAQ: {e}")
        return []

def get_data(ticker):
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False)
        df.dropna(inplace=True)
        if df.shape[0] < 20:
            return None
        return df
    except Exception as e:
        st.warning(f"Errore download dati {ticker}: {e}")
        return None

def add_indicators(df):
    if df is None or df.empty:
        return None

    # Controllo che le colonne esistano
    if not all(col in df.columns for col in ['Close', 'High', 'Low']):
        return None

    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values

    try:
        rsi = talib.RSI(close, timeperiod=14)
        df['RSI'] = rsi
    except Exception as e:
        st.warning(f"Errore calcolo RSI: {e}")
        df['RSI'] = np.nan

    try:
        sma50 = talib.SMA(close, timeperiod=50)
        df['SMA50'] = sma50
    except Exception as e:
        st.warning(f"Errore calcolo SMA50: {e}")
        df['SMA50'] = np.nan

    try:
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_signal'] = macd_signal
    except Exception as e:
        st.warning(f"Errore calcolo MACD: {e}")
        df['MACD'] = np.nan
        df['MACD_signal'] = np.nan

    try:
        atr = talib.ATR(high, low, close, timeperiod=14)
        df['Volatility'] = atr
    except Exception as e:
        st.warning(f"Errore calcolo ATR: {e}")
        df['Volatility'] = np.nan

    return df

def predict_growth(df):
    if df is None or len(df) < 5:
        return 0
    df_tail = df.tail(10).dropna(subset=['Close'])
    if df_tail.empty or len(df_tail) < 5:
        return 0
    X = np.arange(len(df_tail)).reshape(-1,1)
    y = df_tail['Close'].values.reshape(-1,1)
    model = LinearRegression().fit(X,y)
    slope = model.coef_[0][0]
    pct_growth = (slope / y.mean()) * 5
    return round(pct_growth * 100, 2)

def classify_risk(df):
    try:
        vol = df['Volatility'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
    except Exception:
        return "Dati insufficienti"
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
    if df is None or df[['RSI','MACD','MACD_signal','Volatility']].isna().any().any():
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

# STREAMLIT DASHBOARD
st.set_page_config(page_title="NASDAQ Investment Bot", layout="wide")
st.title("📈 NASDAQ Investment Bot – Rischio + Previsioni")

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

