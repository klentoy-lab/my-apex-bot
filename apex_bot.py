import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests
from flask import Flask
from threading import Thread
from datetime import datetime
from scipy.stats import norm
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Bidirectional
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ================= 1. SOVEREIGN STATE & RENDER =================
class EngineState:
    def __init__(self):
        self.is_running = True
        self.timeframe = "1m" 
        self.last_price = 0.0
        self.total_scans = 0
        self.start_time = datetime.now()
        self.check_delay = 60 # Default M1

state = EngineState()
app = Flask(__name__)
RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL") 

@app.route('/')
def home():
    return f"APEX SOVEREIGN: {'ACTIVE' if state.is_running else 'PAUSED'} | TF: {state.timeframe}"

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

# ================= 2. QUANTUM ENGINE (UNRESTRICTED) =================
class ApexQuantum:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Input(shape=(20, 6)),
            Bidirectional(LSTM(64, return_sequences=True)),
            LSTM(32),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        model.compile(loss="mse", optimizer="adam")
        return model

    def calculate_binary_probability(self, df, direction):
        returns = df['close'].pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        t = 1 
        z = (mu * t) / (sigma * np.sqrt(t)) if sigma != 0 else 0
        prob = norm.cdf(z)
        return int(prob * 100) if direction == "BUY" else int((1 - prob) * 100)

    async def analyze(self, df):
        df['ema10'] = df['close'].ewm(span=10).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['delta'] = df['close'].pct_change()
        
        for i in range(2): df[f'feat_{i}'] = 0 
        feat_list = ['close', 'ema10', 'ema50', 'delta', 'feat_0', 'feat_1']
        
        data = df[feat_list].fillna(0).values
        if len(data) < 20: return None
        
        scaled = self.scaler.fit_transform(data)
        X = np.array([scaled[-20:]])
        
        pred = self.model.predict(X, verbose=0)
        target_price = self.scaler.inverse_transform(np.concatenate([pred, np.zeros((1,5))], axis=1))[0][0]
        
        current_price = df['close'].iloc[-1]
        direction = "BUY 🔵" if target_price > current_price else "SELL 🔴"
        prob = self.calculate_binary_probability(df, "BUY" if target_price > current_price else "SELL")
        trend = "📈 UP" if current_price > df['ema50'].iloc[-1] else "📉 DOWN"
        
        return direction, prob, current_price, target_price, trend

# ================= 3. SOVEREIGN COMMANDS =================
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.is_running = True
    await update.message.reply_text("⚡ **GOD MODE: ENABLED.** Engine Initialized. ⚡", parse_mode="Markdown")

async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.is_running = False
    await update.message.reply_text("🛑 **SYSTEM HALTED.** 🛑", parse_mode="Markdown")

async def m1_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.timeframe, state.check_delay = "1m", 60
    await update.message.reply_text("⏱ **TF: M1.** Signals every 60s.", parse_mode="Markdown")

async def m5_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.timeframe, state.check_delay = "5m", 300
    await update.message.reply_text("⌛ **TF: M5.** Signals every 5m.", parse_mode="Markdown")

async def m15_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.timeframe, state.check_delay = "15m", 900
    await update.message.reply_text("🏛 **TF: M15.** Structural trend logic active.", parse_mode="Markdown")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uptime = str(datetime.now() - state.start_time).split('.')[0]
    msg = (f"🏛 **APEX COMMAND CENTER**\n"
           f"📡 STATUS: {'🟢 ACTIVE' if state.is_running else '🔴 PAUSED'}\n"
           f"🕒 TF: {state.timeframe.upper()}\n"
           f"💰 EURUSD: {state.last_price:.5f}\n"
           f"⏳ UPTIME: {uptime}")
    await update.message.reply_text(msg, parse_mode="Markdown")

# ================= 4. MASTER SYNC LOOP =================
async def master_loop(bot_app, engine):
    CHAT_ID = os.environ.get("CHAT_ID", "1936667510")
    while True:
        if state.is_running:
            try:
                df = yf.download("EURUSD=X", interval=state.timeframe, period="1d", progress=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                    df.columns = [c.lower() for c in df.columns]
                    
                    state.last_price = df['close'].iloc[-1]
                    res = await engine.analyze(df)
                    if res:
                        direction, prob, price, target, trend = res
                        meter = "◼️" * (prob // 10) + "◻️" * (10 - (prob // 10))
                        
                        msg = (f"🎯 **QUANTUM SIGNAL: EURUSD**\n"
                               f"━━━━━━━━━━━━━━━━━━━━\n"
                               f"📍 ACTION: **{direction}**\n"
                               f"📊 PROBABILITY: [{meter}] {prob}%\n"
                               f"🕒 TREND: {trend}\n"
                               f"💰 ENTRY: {price:.5f} | 🔮 TARGET: {target:.5f}\n"
                               f"━━━━━━━━━━━━━━━━━━━━\n"
                               f"⚡ TF: {state.timeframe.upper()} | UNRESTRICTED")
                        await bot_app.bot.send_message(CHAT_ID, msg, parse_mode="Markdown")
            except Exception as e: print(f"Loop Error: {e}")
        
        await asyncio.sleep(state.check_delay)

def main():
    keep_alive()
    engine = ApexQuantum()
    TOKEN = os.environ.get("TELEGRAM_TOKEN", "8556975192:AAHwDlJ6okYa46HEsHq_tZgYhR6V9BTXu6A")
    application = ApplicationBuilder().token(TOKEN).build()
    
    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("stop", stop_cmd))
    application.add_handler(CommandHandler("status", status_cmd))
    application.add_handler(CommandHandler("m1", m1_cmd))
    application.add_handler(CommandHandler("m5", m5_cmd))
    application.add_handler(CommandHandler("m15", m15_cmd))
    
    loop = asyncio.get_event_loop()
    loop.create_task(master_loop(application, engine))
    loop.create_task(self_ping())
    application.run_polling()

if __name__ == "__main__":
    main()
