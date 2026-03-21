import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from tqdm import tqdm
import os

def run_master_backtest():
    # 1. LOAD DATA 
    data_file = 'BTCUSDT_recent_10000_15m.csv'
    if not os.path.exists(data_file):
        print(f"❌ File {data_file} not found!")
        return

    df = pd.read_csv(data_file)
    df.columns = [col.lower() for col in df.columns]
    
    # Load XGBoost Brain
    model = xgb.XGBClassifier()
    model.load_model('xgb_model.json')

    # 2. FEATURE ENGINEERING (Must match train_xgboost.py exactly)
    print("🛠️ Calculating Indicators...")
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss)))
    
    df['high_low'] = df['high'] - df['low']
    df['atr'] = df['high_low'].rolling(14).mean()
    
    df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['trend_score'] = (df['ema_8'] - df['ema_21']) / df['ema_21']
    
    df = df.dropna().reset_index(drop=True)

    # 3. AI PREDICTIONS (Vectorized for speed)
    print("🧠 Generating AI Predictions...")
    features = ['rsi', 'trend_score', 'atr']
    probs = model.predict_proba(df[features])
    df['ai_signal'] = np.argmax(probs, axis=1) # 0=Down, 1=Up
    df['ai_conf'] = np.max(probs, axis=1)

    # 4. SIMULATION: ML + LLM GUARD
    balance = 1000.0
    position = 0.0
    history = []
    returns = []
    
    # LLM Guard Simulation
    np.random.seed(42)
    bad_news_events = np.random.choice([True, False], size=len(df), p=[0.03, 0.97])

    print("📈 Running Master Hybrid Simulation...")
    # FIXED: Single loop for efficiency
    for i in tqdm(range(len(df))):
        price = df['close'].iloc[i]
        conf = df['ai_conf'].iloc[i]
        sig = df['ai_signal'].iloc[i]
        ema_8 = df['ema_8'].iloc[i]
        is_bad_news = bad_news_events[i]
        
        prev_val = balance if position == 0 else position * price

        # LOGIC: Only enter if AI is > 30% confident AND no bad news
        if position == 0 and conf > 0.3 and sig == 1:
            if not is_bad_news:
                position = balance / price
                balance = 0
                
        # EXIT: Price drops below the short-term EMA trend line
        elif position > 0 and price < ema_8:
            balance = position * price
            position = 0
        
        current_val = balance if position == 0 else position * price
        history.append(current_val)
        ret = (current_val - prev_val) / prev_val if prev_val != 0 else 0
        returns.append(ret)

    # 5. FINAL METRICS
    hold_roi = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
    strat_roi = ((history[-1] - 1000) / 1000) * 100
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(35040) if np.std(returns) != 0 else 0

    print(f"\n📊 Hybrid Strategy Results:")
    print(f"Strategy ROI: {strat_roi:.2f}% | Hold ROI: {hold_roi:.2f}% | Sharpe: {sharpe:.2f}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(history, label=f'Hybrid AI Strategy (ROI: {strat_roi:.2f}%)', color='blue')
    plt.plot((df['close'] / df['close'].iloc[0]) * 1000, label=f'Buy & Hold (ROI: {hold_roi:.2f}%)', color='gray', alpha=0.5)
    plt.title("Master Backtest: ML + LLM Guard Performance")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_master_backtest()