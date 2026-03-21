import requests
import pandas as pd
import time
import os

def download_deep_history(symbol="BTCUSDT", interval="15m", total_limit=5000):
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    
    # Calculate start time: Exactly 2 years ago from right now
    # (2 years * 365 days * 24 hours * 60 mins * 60 secs * 1000 ms)
    start_time_ms = int(time.time() )
    current_start = start_time_ms

    print(f"📡 Downloading {total_limit} candles")
    print(f"⏳ This will take approximately {total_limit//1000} API calls.")

    while len(all_klines) < total_limit:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': 1000,
            'startTime': current_start
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()

            if not data or len(data) == 0:
                print("⚠️ No more data found for this range.")
                break

            all_klines.extend(data)
            
            # Update start time to the timestamp of the last candle + 1ms
            current_start = data[-1][0] + 1
            
            print(f"   Progress: {len(all_klines)}/{total_limit} sticks collected...")
            
            # Small delay to avoid Binance rate limits (IP Ban protection)
            time.sleep(0.1) 
            
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            break

    # Save exactly what was requested
    df = pd.DataFrame(all_klines[:total_limit], columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 
        'taker_quote_vol', 'ignore'
    ])

    file_name = f"{symbol}_40000_DEEP_DATA.csv"
    df.to_csv(file_name, index=False)
    print(f"\n✅ SUCCESS! File saved as: {file_name}")

if __name__ == "__main__":
    download_deep_history()