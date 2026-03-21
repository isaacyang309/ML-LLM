import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

def calculate_indicators(df):
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss)))
    
    # Trend Strength (ADX-lite)
    df['high_low'] = df['high'] - df['low']
    df['atr'] = df['high_low'].rolling(14).mean()
    
    # Exponential Moving Averages (The "Trend Filter")
    df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['trend_score'] = (df['ema_8'] - df['ema_21']) / df['ema_21']
    
    return df

def train_trend_model():
    df = pd.read_csv("BTCUSDT_5000_training_DATA.csv")
    df = calculate_indicators(df)
    
    # TARGET: Look further ahead (2 hours / 8 candles) to capture trends
    df['future_ret'] = df['close'].shift(-8) / df['close'] - 1
    
    # Binary Labeling: 1 if Up, 0 if Down
    df['target'] = (df['future_ret'] > 0.001).astype(int) # Small threshold to filter flat noise
    
    df = df.dropna()
    features = ['rsi', 'trend_score', 'atr']
    X, y = df[features], df['target']

    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.01,
        scale_pos_weight=1.2, # Slightly bias towards LONGs to stay in the trend
        subsample=0.8,        # <--- Add this
        colsample_bytree=0.8, # <--- Add this
        gamma=1,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Trend Brain Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    model.save_model('xgb_model.json')

if __name__ == "__main__":
    train_trend_model()