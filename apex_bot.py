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
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from sklearn.linear_model import LogisticRegression
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
import xgboost as xgb

# ====================== 1. ENGINE STATE ======================
class EngineState:
    def __init__(self):
        self.is_running = True
        self.timeframe = "1m"
        self.check_delay = 60
        self.last_price = 0.0
        self.start_time = datetime.now()
state = EngineState()

# ====================== 2. FLASK KEEPALIVE ======================
app = Flask(__name__)
RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL")

@app.route('/')
def home(): 
    return f"🏛 APEX SOVEREIGN STATUS: {'ACTIVE' if state.is_running else 'PAUSED'} | TF: {state.timeframe}"

def run_server(): 
    # Render requires dynamic port binding
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

# ====================== 3. ENHANCED APEX BRAIN ======================
class ApexPrecision:
    def __init__(self):
        self.scaler = RobustScaler()
        self.lstm_model = self._build_lstm()
        self.xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.03, subsample=0.8, eval_metric='logloss')
        self.meta_model = LogisticRegression()
        
    def _build_lstm(self):
        model = Sequential([
            Input(shape=(30, 13)),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(64)),
            Dense(32, activation="swish"),
            Dense(1, activation="sigmoid")
        ])
        model.compile(loss="binary_crossentropy", optimizer="adam")
        return model

    def add_features(self, df):
        # Ensure columns are Capitalized for the math logic
        df.columns = [c.capitalize() for c in df.columns]
        df['Ema10'] = df['Close'].ewm(span=10).mean()
        df['Ema20'] = df['Close'].ewm(span=20).mean()
        df['Ema50'] = df['Close'].ewm(span=50).mean()
        
        # RSI Calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (delta.where(delta < 0, 0).abs()).rolling(window=14).mean()
        rs = gain / loss
        df['Rsi'] = 100 - (100 / (1 + rs))
        
        df['Atr'] = df['High'] - df['Low']
        df['Volatility'] = df['Close'].rolling(14).std()
        df['Returns'] = df['Close'].pct_change()
        df['Bb_h'] = df['Close'].rolling(20).mean() + 2*df['Close'].rolling(20).std()
        df['Bb_l'] = df['Close'].rolling(20).mean() - 2*df['Close'].rolling(20).std()
        df.fillna(0, inplace=True)
        return df

    def get_gaussian_prob(self, df, action):
        returns = df['Close'].pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        z = mu / (sigma if sigma != 0 else 0.0001)
        prob = norm.cdf(z)
        return int(prob*100) if action == "BUY" else int((1-prob)*100)

    async def analyze(self, df):
        df = self.add_features(df)
        features = ['Close','Open','High','Low','Ema10','Ema20','Ema50','Rsi','Atr','Volatility','Returns','Bb_h','Bb_l']
        data = df[features].values
        if len(data) < 30: return None

        scaled = self.scaler.fit_transform(data)
        X_lstm = np.array([scaled[-30:]])
        
        # Bayesian AI Consensus
        preds = [self.lstm_model(X_lstm, training=True).numpy()[0][0] for _ in range(10)]
        lstm_prob = np.mean(preds)
        
        # XGBoost Boosted Logic
        X_xgb = scaled
        y = (df['Close'].shift(-1) > df['Close']).astype(int).values
        try:
            self.xgb_model.fit(X_xgb[:-1], y[:-1])
            xgb_prob = self.xgb_model.predict_proba(X_xgb[-1:].reshape(1,-1))[0][1]
        except:
            xgb_prob = lstm_prob

        final_prob = 0.5*xgb_prob + 0.5*lstm_prob
        direction = "BUY 🔵" if final_prob > 0.5 else "SELL 🔴"
        math_prob = self.get_gaussian_prob(df, "BUY" if final_prob > 0.5 else "SELL")
        
        # Weighted Accuracy Output
        final_confidence = int(final_prob*70 + math_prob*0.3)
        trend = "📈 UP" if df['Close'].iloc[-1] > df['Ema50'].iloc[-1] else "📉 DOWN"
        return direction, final_confidence, df['Close'].iloc[-1], trend

# ====================== 4. TELEGRAM INTERFACE ======================
def get_markup():
    keyboard = [
        [InlineKeyboardButton("🚀 Start", callback_data="start"),
         InlineKeyboardButton("🛑 Stop", callback_data="stop"),
         InlineKeyboardButton("📊 Status", callback_data="status")],
        [InlineKeyboardButton("⏱ M1", callback_data="m1"),
         InlineKeyboardButton("⌛ M5", callback_data="m5"),
         InlineKeyboardButton("📉 Refresh", callback_data="refresh")]
    ]
    return InlineKeyboardMarkup(keyboard)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    
    if data == "start": state.is_running = True
    elif data == "stop": state.is_running = False
    elif data == "status": 
        uptime = str(datetime.now() - state.start_time).split('.')[0]
        await query.message.reply_text(f"🏛 **SYSTEM STATUS**\nEngine: {'🟢 ACTIVE' if state.is_running else '🔴 PAUSED'}\nTF: {state.timeframe.upper()}\nUptime: {uptime}")
        return
    elif data in ["m1", "m5"]: 
        state.timeframe = data
        state.check_delay = 60 if data == "m1" else 300
    
    await query.edit_message_text(f"🕹 **Command Processed**: {data.upper()}\nSystem is now synced.", reply_markup=get_markup())

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.is_running = True
    await update.message.reply_text("💎 **APEX PRECISION v16**\nQuantum-XGBoost Engine Live.", reply_markup=get_markup())

async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.is_running = False
    await update.message.reply_text("🛑 **SYSTEM PAUSED**\nMonitoring Standby.", reply_markup=get_markup())

# ====================== 5. MASTER LOOP ======================
async def master_loop(bot_app, engine):
    CHAT_ID = os.environ.get("CHAT_ID", "1936667510")
    while True:
        if state.is_running:
            try:
                # Fetching data for EURUSD
                df = yf.download("EURUSD=X", interval=state.timeframe, period="1d", progress=False)
                if not df.empty:
                    # Logic is handled inside engine.analyze
                    engine_out = await engine.analyze(df)
                    if engine_out:
                        direction, conf, price, trend = engine_out
                        meter = "🟦"*(conf//10)+"⬜"*(10-(conf//10))
                        msg = (f"🎯 **SIGNAL: EURUSD**\n"
                               f"━━━━━━━━━━━━━━━━━━━━\n"
                               f"📍 ACTION: **{direction}**\n"
                               f"📊 PROBABILITY: {conf}%\n"
                               f"💡 METER: [{meter}]\n"
                               f"🕒 TREND: {trend}\n"
                               f"💰 PRICE: {price:.5f}\n"
                               f"━━━━━━━━━━━━━━━━━━━━")
                        await bot_app.bot.send_message(CHAT_ID, msg, parse_mode="Markdown", reply_markup=get_markup())
            except Exception as e: 
                print("Loop Error:", e)
        
        await asyncio.sleep(state.check_delay)

# ====================== 6. MAIN ======================
def main():
    keep_alive()
    engine = ApexPrecision()
    TOKEN = os.environ.get("TELEGRAM_TOKEN", "8211041373:AAG7DnleQ-0UpS1zL83aB6E2In6YennVf-c")
    app_bot = ApplicationBuilder().token(TOKEN).build()
    
    app_bot.add_handler(CommandHandler("start", start_cmd))
    app_bot.add_handler(CommandHandler("stop", stop_cmd))
    app_bot.add_handler(CallbackQueryHandler(button_handler))
    
    loop = asyncio.get_event_loop()
    loop.create_task(master_loop(app_bot, engine))
    loop.create_task(self_ping())
    
    print("💎 APEX SOVEREIGN ENGINE DEPLOYED")
    app_bot.run_polling()

if __name__ == "__main__":
    main()

