import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.linear_model import LinearRegression
import streamlit as st
import plotly.graph_objs as go
import datetime
import logging
import os
import requests

# --- Setup logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configurazione Telegram per notifiche ---
TELEGRAM_BOT_TOKEN = st.secrets.get("telegram_bot_token", "")
TELEGRAM_CHAT_ID = st.secrets.get("telegram_chat_id", "")

def send_telegram_message(message):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        try:
            response = requests.post(url, data=data)
            if response.status_code == 200:
                logger.info("Messaggio Telegram inviato.")
            else:
                logger.warning(f"Telegram API errore: {response.text}")
        except Exception as e:
            logger.error(f"Errore invio Telegram: {e}")

# --- Funzioni analisi ---

def get_nasdaq_tickers():
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")[4]
        tickers = table['Ticker'].tolist()
        logger.info(f"Ticker scaricati: {len(tickers)}")
        return tickers
    except Exception as e:
        st.error(f"Errore caricamento tickers NASDAQ: {e}")
        return []

def get_data(ticker):
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False)
        df.dropna(inplace=True)
        if len(df) < 50:
            logger.warning(f"Dati insufficienti per {ticker}")
            return None
        return df
    except Exception as e:
        logger.warning(f"Errore download dati {ticker}: {e}")
        return None

def add_indicators(df):
    close = df['Close']
    df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df['SMA50'] = close.rolling(window=50).mean()
    macd = ta.trend.MACD(close)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], close)
    df['ATR'] = atr.average_true_range()
    return df

def predict_growth(df):
    df_tail = df.tail(10).dropna(subset=['Close'])
    if len(df_tail) < 5:
        return 0.0
    X = np.arange(len(df_tail)).reshape(-1, 1)
    y = df_tail['Close'].values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0][0]
    avg_price = y.mean()
    if avg_price == 0:
        return 0.0
    pct_growth = (slope / avg_price) * 5 * 100  # previsione % 5 giorni
    return round(pct_growth, 2)

def classify_risk(df):
    try:
        latest = df.iloc[-1]
        atr = latest['ATR']
        rsi = latest['RSI']
    except Exception:
        return "Dati insufficienti"
    if atr > 5 or rsi < 25 or rsi > 75:
        return "Alto"
    elif atr > 3 or rsi < 35 or rsi > 65:
        return "Medio"
    else:
        return "Basso"

def generate_signal(df):
    latest = df.iloc[-1]
    if latest['RSI'] < 30 and latest['Close'] > latest['SMA50'] and latest['MACD'] > latest['MACD_signal']:
        return "BUY"
    elif latest['RSI'] > 70 or latest['Close'] < latest['SMA50']:
        return "SELL"
    else:
        return "HOLD"

def analyze_ticker(ticker):
    df = get_data(ticker)
    if df is None:
        return None
    df = add_indicators(df)
    if df[['RSI','SMA50','MACD','MACD_signal','ATR']].isna().any().any():
        return None
    prediction = predict_growth(df)
    risk = classify_risk(df)
    signal = generate_signal(df)
    price = round(df['Close'].iloc[-1], 2)
    return {
        'Ticker': ticker,
        'Price': price,
        'Signal': signal,
        'Prediction (%)': prediction,
        'Risk': risk,
        'RSI': round(df['RSI'].iloc[-1], 2),
        'ATR': round(df['ATR'].iloc[-1], 2)
    }

# --- Funzione per creare report CSV ---
def save_report_csv(df, filename):
    df.to_csv(filename, index=False)
    logger.info(f"Report salvato: {filename}")

# --- Funzione per inviare alert Telegram su BUY/SELL ---
def notify_signals(df):
    buy_tickers = df[df['Signal'] == 'BUY']['Ticker'].tolist()
    sell_tickers = df[df['Signal'] == 'SELL']['Ticker'].tolist()
    if buy_tickers:
        send_telegram_message(f"Segnali BUY: {', '.join(buy_tickers)}")
    if sell_tickers:
        send_telegram_message(f"Segnali SELL: {', '.join(sell_tickers)}")

# --- Streamlit UI ---

st.set_page_config(page_title="Professional NASDAQ Investment Bot FULL", layout="wide")
st.title("📈 Professional NASDAQ Investment Bot FULL")

st.sidebar.header("Configurazione")
telegram_enabled = st.sidebar.checkbox("Abilita notifiche Telegram (configura secrets)", value=False)
report_enabled = st.sidebar.checkbox("Genera report CSV giornaliero", value=True)

nasdaq_tickers = get_nasdaq_tickers()
results = []

progress = st.progress(0)
status_text = st.empty()

for i, ticker in enumerate(nasdaq_tickers):
    status_text.text(f"Analizzando {ticker} ({i+1}/{len(nasdaq_tickers)})...")
    res = analyze_ticker(ticker)
    if res:
        results.append(res)
    progress.progress((i+1)/len(nasdaq_tickers))

status_text.text("Analisi completata!")

if results:
    df_res = pd.DataFrame(results)
    risk_filter = st.selectbox("Filtra per rischio:", options=["Tutti", "Basso", "Medio", "Alto"])
    signal_filter = st.selectbox("Filtra per segnale:", options=["Tutti", "BUY", "HOLD", "SELL"])

    df_filtered = df_res
    if risk_filter != "Tutti":
        df_filtered = df_filtered[df_filtered['Risk'] == risk_filter]
    if signal_filter != "Tutti":
        df_filtered = df_filtered[df_filtered['Signal'] == signal_filter]

    st.subheader("Tabella titoli")
    st.dataframe(df_filtered.sort_values(by='Prediction (%)', ascending=False))

    selected_ticker = st.selectbox("Seleziona titolo per dettagli:", df_filtered['Ticker'].tolist())

    df_chart = get_data(selected_ticker)
    df_chart = add_indicators(df_chart)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA50'], mode='lines', name='SMA50'))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MACD'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MACD_signal'], mode='lines', name='MACD Signal'))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], mode='lines', name='RSI', yaxis="y2"))

    fig.update_layout(
        title=f"Grafici dettagliati per {selected_ticker}",
        xaxis_title="Data",
        yaxis_title="Prezzo / Valore",
        yaxis2=dict(title="RSI", overlaying='y', side='right')
    )
    st.plotly_chart(fig, use_container_width=True)

    if telegram_enabled:
        notify_signals(df_res)

    if report_enabled:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        filename = f"nasdaq_report_{today}.csv"
        save_report_csv(df_res, filename)
        with open(filename, "rb") as f:
            st.download_button(
                label="Scarica report CSV",
                data=f,
                file_name=filename,
                mime="text/csv"
            )
else:
    st.warning("Nessun titolo analizzato con successo. Riprova più tardi.")


