import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from tqdm import tqdm
import os

def run_real_data_backtest(data_file, model_file='xgb_universal_model.json', conf_threshold=0.51):
    if not os.path.exists(data_file):
        print(f"❌ File {data_file} not found!")
        return

    print(f"\n🚀 Starting Real-Data Backtest for {data_file}")
    df = pd.read_csv(data_file)
    df.columns = [col.lower() for col in df.columns]
    
    # Load the Universal Brain
    model = xgb.XGBClassifier()
    model.load_model(model_file)

    # 1. FEATURE ENGINEERING 
    print("🛠️ Calculating Indicators...")
    df['close'] = pd.to_numeric(df['close'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss)))
    
    # Calculate ATR as a Percentage (The Universal Fix)
    df['high_low'] = df['high'] - df['low']
    df['atr'] = df['high_low'].rolling(14).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100 
    
    # Exponential Moving Averages
    df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['trend_score'] = (df['ema_8'] - df['ema_21']) / df['ema_21']
    
    df = df.dropna().reset_index(drop=True)

    # 2. AI PREDICTIONS
    print("🧠 Generating AI Predictions...")
    features = ['rsi', 'trend_score', 'atr_pct']
    probs = model.predict_proba(df[features])
    df['ai_signal'] = np.argmax(probs, axis=1) # 0=Down, 1=Up
    df['ai_conf'] = np.max(probs, axis=1)

    # 3. PURE PRICE SIMULATION
    balance = 1000.0
    position = 0.0
    history = []
    returns = []
    
    print("📈 Running Simulation (No fake data)...")
    for i in tqdm(range(len(df))):
        price = df['close'].iloc[i]
        conf = df['ai_conf'].iloc[i]
        sig = df['ai_signal'].iloc[i]
        ema_8 = df['ema_8'].iloc[i]
        ema_21 = df['ema_21'].iloc[i]
        
        prev_val = balance if position == 0 else position * price

        # ENTRY: Enter if AI is confident of an UP trend
        if position == 0 and conf > conf_threshold and sig == 1:
            position = balance / price
            balance = 0
                
        # EXIT UPGRADE: Sell only if the actual trend reverses (EMA 8 crosses below EMA 21)
        elif position > 0 and ema_8 < ema_21:
            balance = position * price
            position = 0
        
        current_val = balance if position == 0 else position * price
        history.append(current_val)
        ret = (current_val - prev_val) / prev_val if prev_val != 0 else 0
        returns.append(ret)

    # 4. FINAL METRICS
    hold_roi = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
    strat_roi = ((history[-1] - 1000) / 1000) * 100
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(35040) if np.std(returns) != 0 else 0

    print(f"\n📊 Results for {data_file}:")
    print(f"Strategy ROI: {strat_roi:.2f}% | Hold ROI: {hold_roi:.2f}% | Sharpe: {sharpe:.2f}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(history, label=f'Hybrid AI Strategy (ROI: {strat_roi:.2f}%)', color='blue')
    plt.plot((df['close'] / df['close'].iloc[0]) * 1000, label=f'Buy & Hold (ROI: {hold_roi:.2f}%)', color='gray', alpha=0.5)
    plt.title(f"Real-Data Backtest: {data_file}")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Test BOTH assets with the optimized 0.51 threshold
    run_real_data_backtest('BTCUSDT_2023_2024_June_15m.csv', model_file='xgb_universal_model.json', conf_threshold=0.51)
    
    run_real_data_backtest('ETHUSDT_2023_2024_June_15m.csv', model_file='xgb_universal_model.json', conf_threshold=0.51)