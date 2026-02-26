import os
import asyncio
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from scipy.stats import norm
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from flask import Flask
from threading import Thread

# ====================== 1. CONFIG & SYSTEM STATE ======================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
TD_KEY = os.environ.get("TD_API_KEY", "YOUR_TWELVE_DATA_KEY")
INSTANCE_ID = os.environ.get("RENDER_INSTANCE_ID", "local-dev")[:6]

class EngineState:
    def __init__(self):
        self.is_running = True
        self.timeframe = "1min"
        self.check_delay = 60    
        self.start_time = datetime.now()
        
        # POCKET OPTIONS TRACKER
        self.wins = 0
        self.losses = 0
        self.streak = 0
        self.target = 10
        self.last_signal = None  # "BUY" or "SELL"
        self.last_price = 0.0
state = EngineState()

# ====================== 2. TWELVE DATA ENGINE ======================
def fetch_twelve_data(symbol="EUR/USD", interval="1min"):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={TD_KEY}&outputsize=50"
    try:
        r = requests.get(url)
        data = r.json()
        if data.get("status") == "error":
            return pd.DataFrame()
        df = pd.DataFrame(data.get("values"))
        if df.empty: return pd.DataFrame()
        df.columns = [c.capitalize() for c in df.columns]
        df = df.astype({"Close": float, "Open": float, "High": float, "Low": float})
        return df.iloc[::-1] 
    except:
        return pd.DataFrame()

# ====================== 3. THE QUANTUM BRAIN (ACCURACY UPGRADE) ======================
class QuantumBrain:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = Sequential([
            Input(shape=(20, 6)),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        self.model.compile(loss="mse", optimizer="adam")

    def calculate_binary_probability(self, df, direction):
        returns = df['Close'].pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        t = 1 
        z = (mu * t) / (sigma * np.sqrt(t)) if sigma != 0 else 0
        prob = norm.cdf(z)
        return int(prob * 100) if direction == "BUY" else int((1 - prob) * 100)

    def add_features(self, df):
        df['Ema10'] = df['Close'].ewm(span=10).mean()
        df['Ema50'] = df['Close'].ewm(span=50).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (delta.where(delta < 0, 0).abs()).rolling(14).mean()
        df['Rsi'] = 100 - (100 / (1 + (gain/(loss+1e-9))))
        df['Atr'] = df['High'] - df['Low']
        df['Returns'] = df['Close'].pct_change()
        return df.fillna(0)

    async def analyze(self, df):
        df = self.add_features(df)
        features = ['Close', 'Ema10', 'Ema50', 'Rsi', 'Atr', 'Returns']
        data = df[features].values
        scaled = self.scaler.fit_transform(data)
        
        if len(scaled) < 21: return None
        
        X = np.array([scaled[i-20:i] for i in range(20, len(scaled))])
        Y = scaled[20:, 0]
        
        # Adaptive Fast-Training (The accuracy secret)
        self.model.fit(X, Y, epochs=1, verbose=0)
        
        pred = self.model.predict(X[-1].reshape(1, 20, 6), verbose=0)
        target_price = self.scaler.inverse_transform(np.concatenate([pred, np.zeros((1, 5))], axis=1))[0][0]
        
        current_price = df['Close'].iloc[-1]
        raw_dir = "BUY" if target_price > current_price else "SELL"
        prob = self.calculate_binary_probability(df, raw_dir)
        trend = "📈 UP" if current_price > df['Ema50'].iloc[-1] else "📉 DOWN"
        
        return raw_dir, prob, current_price, target_price, trend

# ====================== 4. LUXURY UI & BUTTONS ======================
def get_markup():
    keyboard = [
        [InlineKeyboardButton("🚀 START", callback_data="start"),
         InlineKeyboardButton("🛑 STOP", callback_data="stop"),
         InlineKeyboardButton("📊 STATUS", callback_data="status")],
        [InlineKeyboardButton("⏱ M1", callback_data="1min"),
         InlineKeyboardButton("⌛ M5", callback_data="5min"),
         InlineKeyboardButton("🔄 RESET", callback_data="reset")]
    ]
    return InlineKeyboardMarkup(keyboard)

async def luxury_loading(query):
    frames = [
        "⚜️ QUANTUM CORE BOOTING ⚜️\n`[████▒▒▒▒▒▒] 40%`",
        "⚜️ NEURAL ALIGNMENT ⚜️\n`[███████▒▒▒] 75%`",
        "⚜️ GAUSSIAN SYNCED ⚜️\n`[██████████] 100%`"
    ]
    for frame in frames:
        try:
            await query.edit_message_text(frame, parse_mode="Markdown")
            await asyncio.sleep(0.3)
        except: pass

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    await luxury_loading(query)
    
    if data == "start": state.is_running = True
    elif data == "stop": state.is_running = False
    elif data == "reset":
        state.wins, state.losses, state.streak = 0, 0, 0
        state.last_signal = None
        await query.edit_message_text("⚜️ **SESSION RESET** ⚜️\nReady for new trades.", reply_markup=get_markup(), parse_mode="Markdown")
        return
    elif data == "status":
        uptime = str(datetime.now() - state.start_time).split('.')[0]
        total = state.wins + state.losses
        wr = int((state.wins / total * 100)) if total > 0 else 0
        msg = (f"🏛 APEX QUANTUM STATUS 🏛\n━━━━━━━━━━━━━━━━━━━━\n"
               f"⚡ ENGINE: {'🟢 ACTIVE' if state.is_running else '🔴 PAUSED'}\n"
               f"⏱ TF: {state.timeframe} | ⏳ **UPTIME:** {uptime}\n"
               f"━━━━━━━━━━━━━━━━━━━━\n"
               f"🏆 WIN RATE: {wr}% ({state.wins}W - {state.losses}L)\n"
               f"🎯 TARGET: {state.wins}/{state.target} WINS | 🔥 **STREAK:** {state.streak}\n"
               f"🆔 `{INSTANCE_ID}`")
        await query.edit_message_text(msg, parse_mode="Markdown", reply_markup=get_markup())
        return
    elif data in ["1min", "5min"]:
        state.timeframe = data
        state.check_delay = 60 if data == "1min" else 300

    await query.edit_message_text(f"💎 **SYSTEM UPDATED** 💎\nMode: `{data.upper()}`", parse_mode="Markdown", reply_markup=get_markup())

# ====================== 5. MASTER LOOP (SOVEREIGN FEED) ======================
async def master_loop(bot_app, engine):
    CHAT_ID = os.environ.get("CHAT_ID", "1936667510")
    while True:
        if state.is_running:
            df = fetch_twelve_data(interval=state.timeframe)
            if not df.empty:
                current_close = df['Close'].iloc[-1]
                
                # Evaluation for Pocket Options
                if state.last_signal is not None:
                    won = (state.last_signal == "BUY" and current_close > state.last_price) or \
                          (state.last_signal == "SELL" and current_close < state.last_price)
                    if won:
                        state.wins += 1; state.streak += 1
                    else:
                        state.losses += 1; state.streak = 0
                    if state.wins >= state.target:
                        await bot_app.bot.send_message(CHAT_ID, "🥂 **DAILY TARGET REACHED!** 🥂")

                # Generate Signal
                res = await engine.analyze(df)
                if res:
                    raw_dir, prob, price, target, trend = res
                    state.last_signal, state.last_price = raw_dir, price
                    meter = "🟦" * (prob // 10) + "⬜" * (10 - (prob // 10))
                    
                    msg = (f"⚜️ QUANTUM TRADE SIGNAL ⚜️\n━━━━━━━━━━━━━━━━━━━━\n"
                           f"📍 ACTION: **{raw_dir} {'🔵' if raw_dir=='BUY' else '🔴'}**\n"
                           f"📊 BINARY PROB: [{meter}] {prob}%\n"
                           f"🕒 TREND: {trend}\n"
                           f"💰 ENTRY: {price:.5f} | 🔮 **TARGET:** {target:.5f}\n"
                           f"━━━━━━━━━━━━━━━━━━━━\n"
                           f"🏆 ACCURACY: {int((state.wins/(state.wins+state.losses+1e-9))*100)}% | 🔥 **STREAK:** {state.streak}\n"
                           f"🎯 TARGET: {state.wins}/{state.target} WINS\n"
                           f"⚡ TF: {state.timeframe.upper()} | 🆔 `{INSTANCE_ID}`")
                    await bot_app.bot.send_message(CHAT_ID, msg, parse_mode="Markdown", reply_markup=get_markup())
                    
        await asyncio.sleep(max(1, state.check_delay - 5))

# Flask Service
server = Flask(__name__)
@server.route('/')
def home(): return f"QUANTUM_ENGINE: {INSTANCE_ID} | ACTIVE"

def main():
    Thread(target=lambda: server.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000))), daemon=True).start()
    TOKEN = os.environ.get("TELEGRAM_TOKEN", "8556975192:AAHwDlJ6okYa46HEsHq_tZgYhR6V9BTXu6A")
    app_bot = ApplicationBuilder().token(TOKEN).build()
    app_bot.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text("⚜️ **QUANTUM ONLINE** ⚜️", reply_markup=get_markup())))
    app_bot.add_handler(CallbackQueryHandler(button_handler))
    
    engine = QuantumBrain()
    asyncio.get_event_loop().create_task(master_loop(app_bot, engine))
    app_bot.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
