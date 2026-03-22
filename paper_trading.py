#!/usr/bin/env python3
"""
Paper Trading Mode - Upgraded for AI (XGBoost + FinBERT)
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import your NEW AI-powered Bot
from src.grid_trading_bot import GridTradingBot

def run_paper_trading():
    print("="*80)
    print("AI PAPER TRADING MODE: ACTIVATED")
    print("="*80)
    print("Strategy: Adaptive Grid")
    print("Brain 1: XGBoost Classifier")
    print("Brain 2: FinBERT Sentiment (Live News)")
    print("="*80 + "\n")

    # Initialize your new AI Bot
    # pair="BTC/USD"
    # enable_trading=False (This ensures it stays in Paper Mode)
    bot = GridTradingBot(pair="BTC/USD", enable_trading=False)

    # Configure your grid settings
    bot.set_initial_lots("0.01,0.02,0.03,0.04,0.09")
    
    interval = 60 # Run every 60 seconds
    
    try:
        while True:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting AI Analysis Cycle...")
            
            # This triggers the fetch_market_data -> get_ml_predictions -> 
            # get_crypto_sentiment -> grid_strategy.analyze loop
            success = bot.run_cycle()
            
            if not success:
                print("Cycle failed, checking connection...")
            
            print(f"Waiting {interval}s for next cycle...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nStop signal received. AI Paper Trading stopped.")

if __name__ == "__main__":
    from datetime import datetime
    run_paper_trading()