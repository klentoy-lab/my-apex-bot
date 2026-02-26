import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests
from flask import Flask
from threading import Thread
from bs4 import BeautifulSoup
from datetime import datetime
import tensorflow as tf
from scipy.stats import norm
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Bidirectional
from telegram.ext import ApplicationBuilder

# ================= 1. RENDER "WAKE-UP" SERVER =================
# Render's free tier needs a website to look at, or it will turn off.
app = Flask('')

@app.route('/')
def home():
    return "Apex Engine is Online! 🚀 Monitoring the market..."

def run_server():
    # Render automatically tells the bot which port to use via an Environment Variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run_server)
    t.start()

# ================= 2. CONFIG =================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
SYMBOL = "EURUSD=X"
CHECK_DELAY = 60 

# ================= 3. NEWS SENTRY =================
class NewsSentry:
    def __init__(self):
        self.high_impact_events = []

    def update_calendar(self):
        try:
            url = "https://www.forexfactory.com/calendar"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            rows = soup.find_all("tr", class_="calendar__row")
            for row in rows:
                impact = row.find("span", class_="icon--impact")
                if impact and "high" in str(impact).lower():
                    name = row.find("td", class_="calendar__event").text.strip()
                    self.high_impact_events.append(name)
            print(f"📅 Scanned {len(self.high_impact_events)} high-impact events.")
        except:
            print("⚠️ News Scraper Error")

# ================= 4. APEX ENGINE =================
class ApexEngine:
    def __init__(self):
        self.model = self._build_model()
        self.scaler = RobustScaler()
        self.sentry = NewsSentry()

    def _build_model(self):
        model = Sequential([
            Input(shape=(30, 8)),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(64)),
            Dense(32, activation="swish"),
            Dense(1, activation="sigmoid")
        ])
        model.compile(loss="binary_crossentropy", optimizer="adam")
        return model

    async def analyze(self, df):
        df['returns'] = df['Close'].pct_change()
        df['vix'] = df['returns'].rolling(20).std()
        df['ema_f'] = df['Close'].ewm(span=10).mean()
        df['ema_s'] = df['Close'].ewm(span=50).mean()
        
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'ema_f', 'ema_s']
        data = df[features].ffill().dropna().values
        if len(data) < 30: return None
        
        scaled = self.scaler.fit_transform(data)
        X = np.array([scaled[-30:]])
        
        # Monte Carlo Logic: The "Winning" confirmation
        mc_preds = [self.model(X, training=True).numpy()[0][0] for _ in range(30)]
        prob = np.mean(mc_preds)
        std = np.std(mc_preds)

        if std > 0.12: return "STAY FLAT 🛡️", 0, "Uncertain"
        if prob > 0.72: return "STRONG BUY 🔵", int(prob*100), "Stable"
        if prob < 0.28: return "STRONG SELL 🔴", int((1-prob)*100), "Stable"
        return "WAIT ⏳", 50, "Neutral"

# ================= 5. LIVE LOOP =================
async def main_loop(bot_app, engine):
    engine.sentry.update_calendar()
    print("💎 APEX v16.0: DEPLOYED & MONITORING")
    
    while True:
        try:
            df = yf.download(SYMBOL, interval="1m", period="1d", progress=False)
            if not df.empty:
                action, confidence, status = await engine.analyze(df)
                if "STRONG" in action:
                    price = df['Close'].iloc[-1]
                    msg = (f"🎯 **APEX SIGNAL: {SYMBOL}**\n"
                           f"━━━━━━━━━━━━━━━━━━━━\n"
                           f"📍 ACTION: **{action}**\n"
                           f"📊 PROBABILITY: {confidence}%\n"
                           f"💰 ENTRY: {price:.5f}\n"
                           f"━━━━━━━━━━━━━━━━━━━━")
                    await bot_app.bot.send_message(CHAT_ID, msg, parse_mode="Markdown")
        except Exception as e: print(f"Error: {e}")
        await asyncio.sleep(CHECK_DELAY)

def main():
    keep_alive() # Starts the web server to stay awake
    engine = ApexEngine()
    bot_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    asyncio.get_event_loop().create_task(main_loop(bot_app, engine))
    bot_app.run_polling()

if __name__ == "__main__":
    main()