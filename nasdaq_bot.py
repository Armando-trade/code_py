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
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials missing.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data, timeout=10)
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
    if not all([EMAIL_SMTP_SERVER, EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_RECEIVER]):
        logger.warning("Email credentials missing.")
        return
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info("Email sent.")
    except Exception as e:
        logger.error(f"Email sending failed: {e}")

@st.cache_data(ttl=600, show_spinner=False)
def get_nasdaq_tickers():
    # Usa un source più stabile o fallback in caso di errore
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")[4]
        tickers = table['Ticker'].tolist()
        logger.info(f"Tickers fetched: {len(tickers)}")
        return tickers
    except Exception as e:
        logger.error(f"Could not load NASDAQ tickers: {e}")
        return []

@st.cache_data(ttl=600, show_spinner=False)
def get_data(ticker, period="3mo", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            logger.warning(f"No data for {ticker}")
            return None
        # Verifica colonne necessarie
        required_cols = ['Close', 'High', 'Low']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing columns in data for {ticker}")
            return None
        df = df.dropna(subset=required_cols)
        if len(df) < 50:
            logger.warning(f"Not enough data for {ticker}")
            return None
        return df
    except Exception as e:
        logger.warning(f"Download error {ticker}: {e}")
        return None

def add_indicators(df):
    df = df.copy()
    # Assicurati tipi corretti e pulizia dati
    for col in ['Close', 'High', 'Low']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    try:
        rsi = ta.momentum.RSIIndicator(close=df['Close'], window=14, fillna=True).rsi()
        df['RSI'] = rsi
    except Exception as e:
        df['RSI'] = np.nan
        logger.warning(f"RSI error: {e}")

    df['SMA50'] = df['Close'].rolling(window=50, min_periods=1).mean()

    try:
        macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
    except Exception as e:
        df['MACD'] = df['MACD_signal'] = np.nan
        logger.warning(f"MACD error: {e}")

    try:
        atr = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14, fillna=True)
        df['ATR'] = atr.average_true_range()
    except Exception as e:
        df['ATR'] = np.nan
        logger.warning(f"ATR error: {e}")

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
    pct_growth = (slope / avg_price) * 5 * 100  # proiezione su 5 giorni approx
    return round(pct_growth, 2)

def classify_risk(df):
    try:
        latest = df.iloc[-1]
        atr = latest['ATR']
        rsi = latest['RSI']
        if pd.isna(atr) or pd.isna(rsi):
            return "Insufficient data"
    except Exception:
        return "Insufficient data"

    if atr > 5 or rsi < 25 or rsi > 75:
        return "High"
    elif atr > 3 or rsi < 35 or rsi > 65:
        return "Medium"
    else:
        return "Low"

def generate_signal(df):
    try:
        latest = df.iloc[-1]
        if latest['RSI'] < 30 and latest['Close'] > latest['SMA50'] and latest['MACD'] > latest['MACD_signal']:
            return "BUY"
        elif latest['RSI'] > 70 or latest['Close'] < latest['SMA50']:
            return "SELL"
        else:
            return "HOLD"
    except Exception as e:
        logger.warning(f"Signal generation error: {e}")
        return "HOLD"

def analyze_ticker(ticker, period, interval):
    df = get_data(ticker, period=period, interval=interval)
    if df is None or df.empty:
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
    # Invia notifiche solo se ci sono segnali BUY/SELL nuovi
    buy_tickers = df[df['Signal'] == 'BUY']['Ticker'].tolist()
    sell_tickers = df[df['Signal'] == 'SELL']['Ticker'].tolist()
    if not buy_tickers and not sell_tickers:
        logger.info("No BUY/SELL signals to notify.")
        return

    msg_parts = []
    if buy_tickers:
        msg_parts.append(f"🟢 BUY signals: {', '.join(buy_tickers)}")
    if sell_tickers:
        msg_parts.append(f"🔴 SELL signals: {', '.join(sell_tickers)}")
    message = "\n".join(msg_parts)
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

    if not df_filtered.empty:
        selected_ticker = st.selectbox("Seleziona titolo per dettagli:", df_filtered['Ticker'].tolist())

        df_chart = get_data(selected_ticker, period=period, interval=interval)
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
            yaxis_title="Prezzo",
            yaxis2=dict(title="RSI", overlaying="y", side="right", range=[0,100]),
            legend=dict(x=0, y=1),
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)

    if report_enabled:
        report_file = f"nasdaq_report_{today_str}.csv"
        save_report_csv(df_res, report_file)
        st.success(f"Report salvato come {report_file}")

    # Invia notifiche
    notify_signals(df_res, telegram_enabled, email_enabled)

else:
    st.warning("Nessun dato disponibile per i ticker selezionati.")

# Auto-refresh logica (Streamlit non supporta nativamente refresh automatico, workaround possibile con st.experimental_rerun e timer)

