import requests
import pandas as pd
import time

def download_specific_range(symbol="BTCUSDT", interval="15m"):
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    
    # Precise timestamps in milliseconds (Jan 1, 2023 to June 30, 2024)
    start_time = 1672531200000  
    end_time = 1719705600000    
    
    current_start = start_time
    print(f"📡 Downloading {symbol} {interval} data from Jan 2023 to June 2024...")

    while current_start < end_time:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': 1000,
            'startTime': current_start,
            'endTime': end_time
        }
        
        response = requests.get(url, params=params)
        data = response.json()

        if not data or len(data) == 0:
            break

        all_klines.extend(data)
        current_start = data[-1][0] + 1
        
        print(f"   Collected {len(all_klines)} candles...")
        time.sleep(0.1) # Rate limit protection

    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 
        'taker_quote_vol', 'ignore'
    ])

    # Convert timestamp to readable format like your snippet
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    file_name = f"{symbol}_2023_2024_June_15m.csv"
    df.to_csv(file_name, index=False)
    print(f"✅ SUCCESS! Saved to {file_name}\n")

if __name__ == "__main__":
    # Download both assets for the multi-pair bot
    download_specific_range("BTCUSDT")
    download_specific_range("ETHUSDT")