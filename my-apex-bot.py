import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests
from flask import Flask
from threading import Thread
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Bidirectional
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ================= 1. GLOBAL SOVEREIGN STATE =================
class EngineState:
    def __init__(self):
        self.is_running = True
        self.timeframe = "1m" 
        self.last_price = 0.0
        self.total_scans = 0
        self.start_time = datetime.now()

state = EngineState()

# ================= 2. RENDER & ANTI-SLEEP =================
app = Flask(__name__)
RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL") 

@app.route('/')
def home():
    return f"APEX ENGINE: {'RUNNING' if state.is_running else 'PAUSED'} | TIMEFRAME: {state.timeframe} 🚀"

def run_server():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    Thread(target=run_server, daemon=True).start()

async def self_ping():
    while True:
        await asyncio.sleep(840) 
        if RENDER_URL:
            try: requests.get(RENDER_URL)
            except: pass

# ================= 3. THE UNRESTRICTED APEX ENGINE =================
class ApexEngine:
    def __init__(self):
        self.model = self._build_model()
        self.scaler = RobustScaler()

    def _build_model(self):
        model = Sequential([
            Input(shape=(30, 8)),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(64)),
            Dense(32, activation="swish"),
            Dense(1, activation="sigmoid")
        ])
        model.compile(loss="binary_crossentropy", optimizer="adam")
        return model

    async def analyze(self, df):
        df['returns'] = df['Close'].pct_change()
        df['ema_f'] = df['Close'].ewm(span=7).mean()
        df['ema_s'] = df['Close'].ewm(span=25).mean()
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'ema_f', 'ema_s']
        data = df[features].ffill().dropna().values
        if len(data) < 30: return None
        scaled = self.scaler.fit_transform(data)
        X = np.array([scaled[-30:]])
        
        # Bayesian Probability Check
        mc_preds = [self.model(X, training=True).numpy()[0][0] for _ in range(15)]
        prob = np.mean(mc_preds)
        
        # 50% ABOVE RULE: Always signals the dominant direction
        if prob >= 0.50:
            return "BUY 🟢", int(prob * 100)
        else:
            return "SELL 🔴", int((1 - prob) * 100)

# ================= 4. TELEGRAM COMMANDS =================
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.is_running = True
    await update.message.reply_text("⚡ **GOD MODE: ENABLED.** Monitoring EURUSD. ⚡", parse_mode="Markdown")

async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.is_running = False
    await update.message.reply_text("🛑 **SYSTEM HALTED.** 🛑", parse_mode="Markdown")

async def m1_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.timeframe = "1m"
    await update.message.reply_text("⏱ **TIMEFRAME: M1.**", parse_mode="Markdown")

async def m5_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.timeframe = "5m"
    await update.message.reply_text("⌛ **TIMEFRAME: M5.**", parse_mode="Markdown")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uptime = str(datetime.now() - state.start_time).split('.')[0]
    status_msg = (f"🏛 **APEX COMMAND CENTER**\n"
                  f"📡 STATUS: {'🟢 ACTIVE' if state.is_running else '🔴 PAUSED'}\n"
                  f"🕒 TIMEFRAME: {state.timeframe.upper()}\n"
                  f"💰 EURUSD: {state.last_price:.5f}\n"
                  f"⏳ UPTIME: {uptime}")
    await update.message.reply_text(status_msg, parse_mode="Markdown")

# ================= 5. MASTER EXECUTION LOOP =================
async def master_loop(bot_app, engine):
    CHAT_ID = os.environ.get("CHAT_ID")
    while True:
        if state.is_running:
            try:
                df = yf.download("EURUSD=X", interval=state.timeframe, period="1d", progress=False)
                if not df.empty:
                    state.last_price = df['Close'].iloc[-1]
                    state.total_scans += 1
                    result = await engine.analyze(df)
                    if result:
                        action, confidence = result
                        # High Frequency: Always sends the most likely direction
                        msg = (f"🏛 **APEX ANALYSIS: EURUSD**\n"
                               f"━━━━━━━━━━━━━━━━━━━━\n"
                               f"📍 BIAS: **{action}**\n"
                               f"📊 PROBABILITY: {confidence}%\n"
                               f"💰 PRICE: {state.last_price:.5f}\n"
                               f"━━━━━━━━━━━━━━━━━━━━")
                        await bot_app.bot.send_message(CHAT_ID, msg, parse_mode="Markdown")
            except Exception as e: print(f"Loop Error: {e}")
        await asyncio.sleep(60)

def main():
    keep_alive()
    engine = ApexEngine()
    TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("stop", stop_cmd))
    application.add_handler(CommandHandler("status", status_cmd))
    application.add_handler(CommandHandler("m1", m1_cmd))
    application.add_handler(CommandHandler("m5", m5_cmd))
    
    loop = asyncio.get_event_loop()
    loop.create_task(master_loop(application, engine))
    loop.create_task(self_ping())
    application.run_polling()

if __name__ == "__main__":
    main()
