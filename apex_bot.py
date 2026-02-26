import os
import asyncio
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from scipy.stats import norm
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
import xgboost as xgb
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from flask import Flask
from threading import Thread

# ====================== 1. CONFIG & SYSTEM STATE ======================
TD_KEY = os.environ.get("TD_API_KEY", "YOUR_TWELVE_DATA_KEY")
INSTANCE_ID = os.environ.get("RENDER_INSTANCE_ID", "local-dev")[:6]

class EngineState:
    def __init__(self):
        self.is_running = True
        self.timeframe = "1min"
        self.check_delay = 60    
        self.start_time = datetime.now()
        
        # --- POCKET OPTIONS WIN TRACKER ---
        self.wins = 0
        self.losses = 0
        self.streak = 0
        self.target = 10  
        self.last_signal = None  
        self.last_price = 0.0    
state = EngineState()

# ====================== 2. TWELVE DATA ENGINE ======================
def fetch_twelve_data(symbol="EUR/USD", interval="1min"):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={TD_KEY}&outputsize=50"
    try:
        r = requests.get(url)
        data = r.json()
        if data.get("status") == "error":
            print(f"⚠️ TD ERROR: {data.get('message')}")
            return pd.DataFrame()
        df = pd.DataFrame(data.get("values"))
        if df.empty: return pd.DataFrame()
        df.columns = [c.capitalize() for c in df.columns]
        df = df.astype({"Close": float, "Open": float, "High": float, "Low": float})
        return df.iloc[::-1] 
    except Exception as e:
        print(f"API Error: {e}")
        return pd.DataFrame()

# ====================== 3. THE APEX BRAIN (UNCHANGED) ======================
class ApexPrecision:
    def __init__(self):
        self.scaler = RobustScaler()
        self.lstm_model = self._build_lstm()
        self.xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric='logloss')
        
    def _build_lstm(self):
        model = Sequential([
            Input(shape=(30, 13)),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.1),
            Bidirectional(LSTM(32)),
            Dense(16, activation="swish"),
            Dense(1, activation="sigmoid")
        ])
        model.compile(loss="binary_crossentropy", optimizer="adam")
        return model

    def add_features(self, df):
        df['Ema10'] = df['Close'].ewm(span=10).mean()
        df['Ema50'] = df['Close'].ewm(span=50).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (delta.where(delta < 0, 0).abs()).rolling(14).mean()
        df['Rsi'] = 100 - (100 / (1 + (gain/(loss+1e-9))))
        df['Atr'] = df['High'] - df['Low']
        df['Volatility'] = df['Close'].rolling(14).std()
        df['Returns'] = df['Close'].pct_change()
        df['Bb_h'] = df['Close'].rolling(20).mean() + 2*df['Close'].rolling(20).std()
        df['Bb_l'] = df['Close'].rolling(20).mean() - 2*df['Close'].rolling(20).std()
        for i in range(2): df[f'F_{i}'] = 0 
        df.fillna(0, inplace=True)
        return df

    async def analyze(self, df):
        df = self.add_features(df)
        features = ['Close','Open','High','Low','Ema10','Ema50','Rsi','Atr','Volatility','Returns','Bb_h','Bb_l','F_0']
        data = df[features].values
        if len(data) < 30: return None
        scaled = self.scaler.fit_transform(data)
        X_lstm = np.array([scaled[-30:]])
        
        lstm_p = float(self.lstm_model(X_lstm).numpy()[0][0])
        try:
            y = (df['Close'].shift(-1) > df['Close']).astype(int).values
            self.xgb_model.fit(scaled[:-1], y[:-1])
            xgb_p = self.xgb_model.predict_proba(scaled[-1:].reshape(1,-1))[0][1]
        except: xgb_p = lstm_p

        final_prob = 0.5 * xgb_p + 0.5 * lstm_p
        raw_direction = "BUY" if final_prob > 0.5 else "SELL"
        direction = "BUY 🔵" if final_prob > 0.5 else "SELL 🔴"
        
        z = df['Returns'].mean() / (df['Returns'].std() + 1e-9)
        math_prob = int(norm.cdf(z)*100) if final_prob > 0.5 else int((1-norm.cdf(z))*100)
        
        return direction, raw_direction, int(final_prob*70 + math_prob*0.3), df['Close'].iloc[-1]

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
    """Creates a high-end animated terminal effect in Telegram"""
    frames = [
        "⚜️ SYPHONING MARKET DATA ⚜️\n`[████▒▒▒▒▒▒] 40%`",
        "⚜️ ALIGNING NEURAL WEIGHTS ⚜️\n`[███████▒▒▒] 75%`",
        "⚜️ SYSTEM OPTIMIZED ⚜️\n`[██████████] 100%`"
    ]
    for frame in frames:
        try:
            await query.edit_message_text(frame, parse_mode="Markdown")
            await asyncio.sleep(0.4)
        except:
            pass # Ignore if Telegram API throws a minor timing exception

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    
    # Trigger the luxury animation first
    await luxury_loading(query)
    
    if data == "start": state.is_running = True
    elif data == "stop": state.is_running = False
    elif data == "reset":
        state.wins, state.losses, state.streak = 0, 0, 0
        state.last_signal = None
        state.last_price = 0.0
        await query.edit_message_text("⚜️ **SESSION RESET** ⚜️\nScoreboard cleared. Awaiting new signals...", reply_markup=get_markup(), parse_mode="Markdown")
        return
    elif data == "status":
        uptime = str(datetime.now() - state.start_time).split('.')[0]
        total = state.wins + state.losses
        wr = int((state.wins / total * 100)) if total > 0 else 0
        msg = (f"🏛 APEX SOVEREIGN STATUS 🏛\n"
               f"━━━━━━━━━━━━━━━━━━━━\n"
               f"⚡ ENGINE: {'🟢 ACTIVE' if state.is_running else '🔴 PAUSED'}\n"
               f"⏱  TIMEFRAME: {state.timeframe}\n"
               f"⏳ UPTIME: {uptime}\n"
               f"━━━━━━━━━━━━━━━━━━━━\n"
               f"🏆 WIN RATE {wr}% ({state.wins}W - {state.losses}L)\n"
               f"🎯 TARGET: {state.wins}/{state.target} WINS\n"
               f"🔥 STREAK: {state.streak}\n"
               f"🆔 `{INSTANCE_ID}`")
        await query.edit_message_text(msg, parse_mode="Markdown", reply_markup=get_markup())
        return
    elif data in ["1min", "5min"]:
        state.timeframe = data
        state.check_delay = 60 if data == "1min" else 300

    await query.edit_message_text(f"💎 **SYSTEM UPDATED** 💎\nCommand: `{data.upper()}` Applied.\nInstance: `{INSTANCE_ID}`", parse_mode="Markdown", reply_markup=get_markup())

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.is_running = True
    await update.message.reply_text("⚜️ **APEX SOVEREIGN: ONLINE** ⚜️\nQuantum Engine Initialized.", parse_mode="Markdown", reply_markup=get_markup())

async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.wins, state.losses, state.streak = 0, 0, 0
    state.last_signal = None
    state.last_price = 0.0
    await update.message.reply_text("⚜️ **SESSION RESET** ⚜️\nAll trackers cleared.", parse_mode="Markdown", reply_markup=get_markup())

# ====================== 5. MASTER LOOP ======================
async def master_loop(bot_app, engine):
    CHAT_ID = os.environ.get("CHAT_ID", "1936667510")
    while True:
        if state.is_running:
            df = fetch_twelve_data(interval=state.timeframe)
            if not df.empty:
                current_close = df['Close'].iloc[-1]
                
                if state.last_signal is not None:
                    won = False
                    if state.last_signal == "BUY" and current_close > state.last_price: won = True
                    elif state.last_signal == "SELL" and current_close < state.last_price: won = True
                    
                    if won:
                        state.wins += 1
                        state.streak += 1
                    else:
                        state.losses += 1
                        state.streak = 0
                        
                    if state.wins == state.target:
                        target_msg = f"🥂 **DAILY TARGET REACHED!** 🥂\n({state.target} Wins secured). Step away with profits."
                        await bot_app.bot.send_message(CHAT_ID, target_msg, parse_mode="Markdown")

                res = await engine.analyze(df)
                if res:
                    direction_ui, raw_direction, conf, price = res
                    
                    state.last_signal = raw_direction
                    state.last_price = price
                    
                    total_trades = state.wins + state.losses
                    win_rate = int((state.wins / total_trades * 100)) if total_trades > 0 else 0
                    meter = "🟦"*(conf//10)+"⬜"*(10-(conf//10))
                    
                    msg = (f"⚜️ APEX PO SIGNAL ⚜️\n"
                           f"━━━━━━━━━━━━━━━━━━━━\n"
                           f"📍 ACTION: **{direction_ui}**\n"
                           f"📊 PROBABILITY: {conf}%\n"
                           f"💡 METER: [{meter}]\n"
                           f"💰 ENTRY: {price:.5f}\n"
                           f"━━━━━━━━━━━━━━━━━━━━\n"
                           f"🏆 SESSION: {win_rate}% | 🔥 **STREAK:** {state.streak}\n"
                           f"🎯 TARGET: {state.wins}/{state.target} WINS\n"
                           f"⚡ TF: {state.timeframe.upper()} | 🆔 `{INSTANCE_ID}`")
                    await bot_app.bot.send_message(CHAT_ID, msg, parse_mode="Markdown", reply_markup=get_markup())
                    
        await asyncio.sleep(state.check_delay)

# Flask Service
server = Flask(__name__)
@server.route('/')
def home(): return f"TD_ENGINE: {INSTANCE_ID} | {'RUNNING' if state.is_running else 'PAUSED'}"

def main():
    Thread(target=lambda: server.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000))), daemon=True).start()
    TOKEN = os.environ.get("TELEGRAM_TOKEN", "8556975192:AAHwDlJ6okYa46HEsHq_tZgYhR6V9BTXu6A")
    app_bot = ApplicationBuilder().token(TOKEN).build()
    
    app_bot.add_handler(CommandHandler("start", start_cmd))
    app_bot.add_handler(CommandHandler("reset", reset_cmd))
    app_bot.add_handler(CallbackQueryHandler(button_handler))
    
    engine = ApexPrecision()
    loop = asyncio.get_event_loop()
    loop.create_task(master_loop(app_bot, engine))
    
    print(f"[{INSTANCE_ID}] Polling Luxury PO Engine...")
    app_bot.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
