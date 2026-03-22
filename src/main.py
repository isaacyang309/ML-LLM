#!/usr/bin/env python3
import sys
import os
import time
import threading
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from typing import Dict
from enum import Enum

# Set up dynamic paths to ensure modules are found correctly
# main.py lives at: quant-trading-bot-master/src/main.py
# So: parents[0] = src/           (for trading_logger, strategy, etc.)
#     parents[1] = project root   (for config/ folder and xgb_model.json)
file_path = Path(__file__).resolve()
src_dir = str(file_path.parents[0])       # .../quant-trading-bot-master/src
project_root = str(file_path.parents[1])  # .../quant-trading-bot-master
for p in [src_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Imports — no 'src.' prefix needed since src_dir is now on sys.path
# config/ is at the project root level, so 'config.config' works once project_root is on sys.path
from trading_logger import TradingLogger
from roostoo_client import RoostooClient
from horus_client import HorusClient
from sentiment_analyzer import SentimentAnalyzer
from strategy import GridStrategy, GridTradeConfig, GridDirection
from config.config import Config

# Helper Enum for compatibility
class Action(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class TradingBot:
    def __init__(self, enable_dashboard=False):
        # Logger MUST be initialized first so it's available in all subsequent except blocks
        self.logger = TradingLogger()
        self.config = Config()
        self.roostoo = RoostooClient()
        self.horus = HorusClient()
        
        # --- [STEP 2] INITIALIZE HYBRID AI BRAIN ---
        self.sentiment_analyzer = SentimentAnalyzer()
        self.ml_model = None
        # xgb_model.json lives in the ML/ folder at project root
        model_path = os.path.join(project_root, "ML", "xgb_model.json")
        if os.path.exists(model_path):
            try:
                self.ml_model = xgb.Booster()
                self.ml_model.load_model(model_path)
                self.logger.logger.info(f"XGBoost Trend Brain Loaded from: {model_path}")
            except Exception as e:
                self.logger.logger.error(f"Failed to load ML Brain: {e}")
                self.ml_model = None
        else:
            self.logger.logger.warning(
                f"ML Brain model not found at '{model_path}'. "
                "Bot will run without XGBoost signals. "
                "To enable it, train a model using ML/train_xgboost.py first."
            )
            
        # Create Strategy instances for each pair
        self.strategies = {}
        for pair in self.config.SUPPORTED_PAIRS:
            strategy_config = GridTradeConfig.get_config_for_asset(pair)
            self.strategies[pair] = GridStrategy(config=strategy_config)
            self.strategies[pair].min_trade_interval_seconds = 3600  # 1 hour cooldown
            self.strategies[pair].last_trade_time = datetime.now()
        
        self.running = True
        self.MIN_BTC_AMOUNT = 0.00001
        self.MIN_TRADE_VALUE = 1.0
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.consecutive_losses = 0
        self.peak_portfolio_value = 50000.0
        self.initial_trades_executed = {pair: False for pair in self.config.SUPPORTED_PAIRS}
        
        # RECOVERY: Sync state with actual positions
        self._recover_position_state()

    def _recover_position_state(self):
        """Recover strategy position state from actual balance and trade logs."""
        try:
            balance = self.roostoo.get_account_balance()
            if 'error' in balance: return
            
            for pair in self.config.SUPPORTED_PAIRS:
                base_currency = pair.split('/')[0]
                base_holdings = balance.get(base_currency, {}).get('free', 0)
                
                if base_holdings > 0.0001:
                    self.strategies[pair].add_position(0.0, base_holdings)
                    self.strategies[pair].current_direction = GridDirection.UP
        except Exception as e:
            self.logger.logger.error(f"Recovery Error: {e}")

    def get_pair_precision(self, pair: str) -> dict:
        return {'BTC/USD': {'amount': 5, 'price': 2}, 'ETH/USD': {'amount': 4, 'price': 2}}.get(pair, {'amount': 4, 'price': 2})

    def execute_trade(self, decision, balance_data: Dict, pair: str):
        try:
            symbol = pair
            base_currency = symbol.split('/')[0]
            quote_currency = symbol.split('/')[1]
            precision = self.get_pair_precision(pair)
            
            if decision.action == Action.BUY:
                available_cash = balance_data.get(quote_currency, {}).get('free', 0)
                quantity = (available_cash * self.config.MAX_POSITION_SIZE) / decision.price
                quantity = round(quantity, precision['amount'])
                
                if quantity * decision.price >= self.MIN_TRADE_VALUE:
                    result = self.roostoo.place_order(coin=base_currency, side='BUY', quantity=quantity)
                    if 'error' not in result:
                        self.logger.log_trade({'action': 'BUY', 'symbol': symbol, 'quantity': quantity, 'price': decision.price, 'reason': decision.reason})
                        self.strategies[pair].add_position(decision.price, quantity)

            elif decision.action == Action.SELL:
                available_coin = balance_data.get(base_currency, {}).get('free', 0)
                quantity = round(available_coin, precision['amount'])
                if quantity > 0:
                    result = self.roostoo.place_order(coin=base_currency, side='SELL', quantity=quantity)
                    if 'error' not in result:
                        self.logger.log_trade({'action': 'SELL', 'symbol': symbol, 'quantity': quantity, 'price': decision.price, 'reason': decision.reason})
        except Exception as e:
            self.logger.logger.error(f"Trade Execution Failed: {e}")

    def run(self):
        self.logger.logger.info("Starting Fused AI Trading Bot Cycle...")
        iteration = 0
        while self.running:
            try:
                iteration += 1
                balance_data = self.roostoo.get_account_balance()
                if 'error' in balance_data: continue

                current_prices = {}
                for pair in self.config.SUPPORTED_PAIRS:
                    base_currency = pair.split('/')[0]
                    
                    # 1. Fetch Price History
                    end_time = int(time.time())
                    start_time = end_time - (15 * 60 * 100)
                    klines = self.horus.get_price_history(symbol=base_currency, interval='15m', start=start_time, end=end_time)
                    
                    if not klines: continue

                    # --- [STEP 3] DEFINE DF AND RUN AI BRAINS ---
                    df = pd.DataFrame(klines)
                    df = df.apply(pd.to_numeric)
                    current_price = df.iloc[-1]['price']
                    current_prices[base_currency] = current_price

                    # A. Sentiment Brain (Brain 1)
                    sentiment_score = self.sentiment_analyzer.get_crypto_sentiment()

                    # B. XGBoost Brain (Brain 2)
                    ml_signal, ml_confidence = 0, 0.0
                    if self.ml_model is not None and len(df) > 25:
                        # Feature Engineering
                        df['rsi'] = 100 - (100 / (1 + (df['price'].diff().where(lambda x: x>0, 0).rolling(14).mean() / -df['price'].diff().where(lambda x: x<0, 0).rolling(14).mean())))
                        df['trend_score'] = (df['price'].ewm(span=8).mean() - df['price'].ewm(span=21).mean()) / df['price'].ewm(span=21).mean()
                        df['atr'] = (df['high'] - df['low']).rolling(14).mean()
                        
                        feat = df.iloc[-1][['rsi', 'trend_score', 'atr']].to_frame().T
                        probs = self.ml_model.predict(xgb.DMatrix(feat))[0]
                        ml_signal = 1 if probs > 0.5 else -1
                        ml_confidence = float(probs if ml_signal == 1 else 1 - probs)

                    # 2. Strategic Decision
                    decision_dict = self.strategies[pair].analyze(
                        klines_data=df, 
                        current_price=current_price,
                        ml_signal=ml_signal,
                        ml_confidence=ml_confidence,
                        sentiment_score=sentiment_score
                    )

                    # 3. Apply Veto Logic for Entry
                    if decision_dict['action'] == 'OPEN_POSITION':
                        if sentiment_score < 0.0:
                            decision_dict.update({'action': 'HOLD', 'reason': 'VETO: Bearish Sentiment'})
                        elif ml_confidence < 0.60:
                            decision_dict.update({'action': 'HOLD', 'reason': f'VETO: Low Confidence ({ml_confidence:.2f})'})

                    # 4. Map and Execute
                    class TempDecision: pass
                    decision = TempDecision()
                    decision.action = Action.BUY if decision_dict['action'] == 'OPEN_POSITION' else (Action.SELL if decision_dict['action'] in ['CLOSE_ALL', 'REVERSE_POSITION'] else Action.HOLD)
                    decision.price = current_price
                    decision.reason = decision_dict['reason']
                    
                    self.logger.logger.info(f"[{pair}] Decision: {decision.action.value} | Reason: {decision.reason}")
                    
                    if decision.action != Action.HOLD:
                        self.execute_trade(decision, balance_data, pair)

                time.sleep(self.config.TRADE_INTERVAL)
            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                self.logger.logger.error(f"Loop Error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = TradingBot(enable_dashboard=False)
    bot.run()