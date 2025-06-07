import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.linear_model import LinearRegression
import streamlit as st
import plotly.graph_objs as go
import datetime
import logging
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NASDAQ_BOT")

# --- Telegram config ---
TELEGRAM_BOT_TOKEN = st.secrets.get("telegram_bot_token", "")
TELEGRAM_CHAT_ID = st.secrets.get("telegram_chat_id", "")

# --- Email config ---
EMAIL_ENABLED = st.secrets.get("email_enabled", False)
EMAIL_SMTP_SERVER = st.secrets.get("email_smtp_server", "")
EMAIL_SMTP_PORT = int(st.secrets.get("email_smtp_port", 587))
EMAIL_ADDRESS = st.secrets.get("email_address", "")
EMAIL_PASSWORD = st.secrets.get("email_password", "")
EMAIL_RECEIVER = st.secrets.get("email_receiver", "")

def send_telegram_message(message):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        try:
            response = requests.post(url, data=data)
            if response.status_code == 200:
                logger.info("Telegram message sent.")
            else:
                logger.warning(f"Telegram error: {response.text}")
        except Exception as e:
            logger.error(f"Telegram sending failed: {e}")

def send_email(subject, body):
    if not EMAIL_ENABLED:
        logger.info("Email alert disabled.")
        return
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        logger.info("Email sent.")
    except Exception as e:
        logger.error(f"Email sending failed: {e}")

@st.cache_data(ttl=600)
def get_nasdaq_tickers():
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")[4]
        tickers = table['Ticker'].tolist()
        logger.info(f"Tickers fetched: {len(tickers)}")
        return tickers
    except Exception as e:
        st.error(f"Could not load NASDAQ tickers: {e}")
        return []

@st.cache_data(ttl=600)
def get_data(ticker, period="3mo", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        df.dropna(inplace=True)
        if len(df) < 50:
            logger.warning(f"Not enough data for {ticker}")
            return None
        return df
    except Exception as e:
        logger.warning(f"Download error {ticker}: {e}")
        return None

def add_indicators(df):
    if not isinstance(df, pd.DataFrame):
        logger.error(f"add_indicators: input not a DataFrame but {type(df)}")
        return df
    required_cols = ['Close', 'High', 'Low']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"add_indicators: missing required columns {required_cols} in df.columns {df.columns}")
        return df

    df = df.copy()
    try:
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill').fillna(method='bfill')
        high = pd.to_numeric(df['High'], errors='coerce').fillna(method='ffill').fillna(method='bfill')
        low = pd.to_numeric(df['Low'], errors='coerce').fillna(method='ffill').fillna(method='bfill')
    except Exception as e:
        logger.error(f"Error converting columns to numeric: {e}")
        return df

    close = pd.Series(close.values, index=df.index)
    high = pd.Series(high.values, index=df.index)
    low = pd.Series(low.values, index=df.index)

    try:
        rsi_indicator = ta.momentum.RSIIndicator(close=close, window=14, fillna=True)
        df['RSI'] = rsi_indicator.rsi()
    except Exception as e:
        df['RSI'] = np.nan
        logger.warning(f"RSI calculation error: {e}")

    df['SMA50'] = close.rolling(window=50, min_periods=1).mean()

    try:
        macd_indicator = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9, fillna=True)
        df['MACD'] = macd_indicator.macd()
        df['MACD_signal'] = macd_indicator.macd_signal()
    except Exception as e:
        df['MACD'] = df['MACD_signal'] = np.nan
        logger.warning(f"MACD calculation error: {e}")

    try:
        atr_indicator = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14, fillna=True)
        df['ATR'] = atr_indicator.average_true_range()
    except Exception as e:
        df['ATR'] = np.nan
        logger.warning(f"ATR calculation error: {e}")

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
    pct_growth = (slope / avg_price) * 5 * 100
    return round(pct_growth, 2)

def classify_risk(df):
    try:
        latest = df.iloc[-1]
        atr = latest['ATR']
        rsi = latest['RSI']
    except Exception:
        return "Insufficient data"
    if atr > 5 or rsi < 25 or rsi > 75:
        return "High"
    elif atr > 3 or rsi < 35 or rsi > 65:
        return "Medium"
    else:
        return "Low"

def generate_signal(df):
    latest = df.iloc[-1]
    if latest['RSI'] < 30 and latest['Close'] > latest['SMA50'] and latest['MACD'] > latest['MACD_signal']:
        return "BUY"
    elif latest['RSI'] > 70 or latest['Close'] < latest['SMA50']:
        return "SELL"
    else:
        return "HOLD"

def analyze_ticker(ticker, period, interval):
    df = get_data(ticker, period=period, interval=interval)
    if df is None or df.empty:
        return None
    if 'Close' not in df.columns:
        logger.warning(f"{ticker} data missing 'Close' column")
        return None
    
    df = add_indicators(df)

    prediction = predict_growth(df)
    risk = classify_risk(df)
    signal = generate_signal(df)

    return {
        'Ticker': ticker,
        'Prediction (%)': prediction,
        'Risk': risk,
        'Signal': signal,
    }

def save_report_csv(df, filename):
    df.to_csv(filename, index=False)
    logger.info(f"Report saved: {filename}")

def append_to_history(df, folder="history"):
    os.makedirs(folder, exist_ok=True)
    month_str = datetime.datetime.now().strftime("%Y-%m")
    path = os.path.join(folder, f"signals_{month_str}.csv")
    if os.path.exists(path):
        existing_df = pd.read_csv(path)
        combined_df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates(subset=['Ticker', 'Date'])
    else:
        combined_df = df
    combined_df.to_csv(path, index=False)
    logger.info(f"History updated: {path}")

def notify_signals(df, telegram_enabled=False, email_enabled=False):
    buy_tickers = df[df['Signal'] == 'BUY']['Ticker'].tolist()
    sell_tickers = df[df['Signal'] == 'SELL']['Ticker'].tolist()
    msg_parts = []
    if buy_tickers:
        msg_parts.append(f"🟢 BUY signals: {', '.join(buy_tickers)}")
    if sell_tickers:
        msg_parts.append(f"🔴 SELL signals: {', '.join(sell_tickers)}")
    message = "\n".join(msg_parts)
    if message:
        if telegram_enabled:
            send_telegram_message(message)
        if email_enabled and EMAIL_ENABLED:
            send_email("NASDAQ Bot Alerts", message)

# --- Streamlit App ---

st.set_page_config(page_title="Professional NASDAQ Investment Bot FULL", layout="wide")
st.title("📈 Professional NASDAQ Investment Bot FULL")

with st.sidebar:
    st.header("Configurazione")
    telegram_enabled = st.checkbox("Abilita notifiche Telegram", value=False)
    email_enabled = st.checkbox("Abilita notifiche Email", value=False)
    report_enabled = st.checkbox("Genera report CSV giornaliero", value=True)
    period = st.selectbox("Seleziona periodo dati:", options=["1mo","3mo","6mo","1y","2y"], index=1)
    interval = st.selectbox("Seleziona intervallo dati:", options=["1d","1wk","1mo"], index=0)
    auto_refresh = st.checkbox("Auto-refresh ogni 15 minuti", value=False)

nasdaq_tickers = get_nasdaq_tickers()

results = []
progress = st.progress(0)
status_text = st.empty()

today_str = datetime.datetime.now().strftime("%Y-%m-%d")

with st.spinner("Analizzando tickers..."):
    for i, ticker in enumerate(nasdaq_tickers):
        status_text.text(f"Analizzando {ticker} ({i+1}/{len(nasdaq_tickers)})...")
        res = analyze_ticker(ticker, period, interval)
        if res:
            res['Date'] = today_str
            results.append(res)
        progress.progress((i+1)/len(nasdaq_tickers))

status_text.text("Analisi completata!")

if results:
    df_res = pd.DataFrame(results)

    # Salva storico mensile
    append_to_history(df_res)

    # Riepilogo
    st.markdown("### 📊 Riepilogo Segnali")
    buy_count = (df_res['Signal'] == 'BUY').sum()
    sell_count = (df_res['Signal'] == 'SELL').sum()
    hold_count = (df_res['Signal'] == 'HOLD').sum()
    st.write(f"🟢 BUY: {buy_count} | 🔴 SELL: {sell_count} | ⚪ HOLD: {hold_count}")

    # Filtri
    risk_filter = st.selectbox("Filtra per rischio:", options=["Tutti", "Low", "Medium", "High"])
    signal_filter = st.selectbox("Filtra per segnale:", options=["Tutti", "BUY", "HOLD", "SELL"])

    df_filtered = df_res
    if risk_filter != "Tutti":
        df_filtered = df_filtered[df_filtered['Risk'] == risk_filter]
    if signal_filter != "Tutti":
        df_filtered = df_filtered[df_filtered['Signal'] == signal_filter]

    st.subheader("Tabella titoli")
    st.dataframe(df_filtered.sort_values(by='Prediction (%)', ascending=False), use_container_width=True)

    selected_ticker = st.selectbox("Seleziona titolo per dettagli:", df_filtered['Ticker'].tolist())

    df_chart = get_data(selected_ticker, period=period, interval=interval)
    if df_chart is not None and not df_chart.empty:
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
            yaxis=dict(domain=[0, 0.7]),
            yaxis2=dict(domain=[0.7, 1], overlaying='y', side='right'),
            legend=dict(orientation="h")
        )

        st.plotly_chart(fig, use_container_width=True)

    if report_enabled:
        filename = f"nasdaq_report_{today_str}.csv"
        save_report_csv(df_res, filename)
        st.success(f"Report salvato: {filename}")

    notify_signals(df_res, telegram_enabled=telegram_enabled, email_enabled=email_enabled)

else:
    st.warning("Nessun dato disponibile o errore nell'analisi.")

if auto_refresh:
    st.experimental_rerun()
