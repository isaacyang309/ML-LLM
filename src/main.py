#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Fix: Dynamic pathing to find the project root
# This gets the directory where main.py is (src) and then goes up one level
file_path = Path(__file__).resolve()
project_root = str(file_path.parents[1])

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import threading
from datetime import datetime
from typing import Dict

# --- AI & STRATEGY IMPORTS ---
import pandas as pd
import numpy as np
import xgboost as xgb
from trading_logger import TradingLogger
from roostoo_client import RoostooClient
from horus_client import HorusClient
from sentiment_analyzer import SentimentAnalyzer
from strategy import GridStrategy, GridTradeConfig, GridDirection
from config.config import Config

# Helper Enum to maintain compatibility with Cw's logger
from enum import Enum

class Action(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradingBot:

    def __init__(self, enable_dashboard=False):
        # --- MUST initialize logger, config, and clients first ---
        self.config = Config()
        self.logger = TradingLogger()
        self.roostoo = RoostooClient()
        self.horus = HorusClient()

        # --- LOAD SENTIMENT ANALYZER ---
        self.sentiment_analyzer = SentimentAnalyzer()

        # --- LOAD ML BRAIN (model lives in ML/ subfolder) ---
        try:
            self.ml_model = xgb.Booster()
            model_path = os.path.join(project_root, "ML", "xgb_model.json")
            self.ml_model.load_model(model_path)
            self.logger.logger.info(f"XGBoost Trend Brain Loaded from {model_path}!")
        except Exception as e:
            self.logger.logger.error(f"Failed to load ML Brain: {e}")
            self.ml_model = None

        # Create a Fused Grid Strategy instance for each trading pair
        self.strategies = {}
        for pair in self.config.SUPPORTED_PAIRS:
            strategy_config = GridTradeConfig.get_config_for_asset(pair)
            self.strategies[pair] = GridStrategy(config=strategy_config)
            # Add compatibility attributes for Cw's cooldown logic
            self.strategies[pair].min_trade_interval_seconds = 300
            self.strategies[pair].last_trade_time = datetime.now()

        self.running = True
        self.enable_dashboard = enable_dashboard
        self.dashboard_thread = None

        # Trading constants
        self.MIN_BTC_AMOUNT = 0.00001
        self.MIN_TRADE_VALUE = 1.0

        # Performance tracking (shared across all pairs)
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.consecutive_losses = 0
        self.peak_portfolio_value = 50000.0

        # Track which pairs have executed initial trades
        self.initial_trades_executed = {pair: False for pair in self.config.SUPPORTED_PAIRS}

        # RECOVERY: Sync strategy state with actual positions from logs
        self._recover_position_state()

        if enable_dashboard:
            self.start_dashboard()

    def start_dashboard(self):
        """Start dashboard (local testing only)"""
        try:
            from dashboard import start_dashboard
            self.dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
            self.dashboard_thread.start()
            self.logger.logger.info("Dashboard started: http://localhost:8050")
        except Exception as e:
            self.logger.logger.error(f"Failed to start dashboard: {e}")

    def _recover_position_state(self):
        """Recover strategy position state from actual balance and trade logs."""
        try:
            self.logger.logger.info("=" * 60)
            self.logger.logger.info("CHECKING FOR EXISTING POSITIONS TO RECOVER")
            self.logger.logger.info("=" * 60)

            balance = self.roostoo.get_account_balance()
            if 'error' in balance:
                self.logger.logger.warning(f"Could not fetch balance for recovery: {balance.get('error')}")
                return

            for pair in self.config.SUPPORTED_PAIRS:
                base_currency = pair.split('/')[0]
                base_holdings = balance.get(base_currency, {}).get('free', 0)

                self.logger.logger.info(f"[{pair}] Current Balance: {base_holdings:.8f} {base_currency}")

                if base_holdings < 0.0001:
                    self.logger.logger.info(f"[{pair}] No significant position detected - starting fresh")
                    continue

                self.logger.logger.warning(f"[{pair}] [WARNING] EXISTING POSITION DETECTED: {base_holdings:.8f} {base_currency}")

                import json
                trade_log_path = self.logger.logs_dir / 'trade_history.json'

                if not trade_log_path.exists():
                    self.logger.logger.error(f"[{pair}] [ERROR] Position exists but no trade_history.json found!")
                    continue

                with open(trade_log_path, 'r') as f:
                    trades = json.load(f)

                if not trades:
                    continue

                pair_trades = [t for t in trades if t.get('symbol') == pair]
                if not pair_trades:
                    continue

                buy_trades = [t for t in pair_trades if t.get('action') == 'BUY']
                sell_trades = [t for t in pair_trades if t.get('action') == 'SELL']

                if not buy_trades:
                    continue

                total_bought_qty = sum(t.get('quantity', 0) for t in buy_trades)
                total_bought_value = sum(t.get('total', 0) for t in buy_trades)

                if total_bought_qty > 0:
                    avg_entry_price = total_bought_value / total_bought_qty
                else:
                    avg_entry_price = buy_trades[-1].get('price', 0)

                last_buy = buy_trades[-1]
                last_trade = pair_trades[-1]

                # Restore position in GridStrategy (Adapted for Fused version)
                self.strategies[pair].add_position(avg_entry_price, base_holdings)
                self.strategies[pair].current_direction = GridDirection.UP

                self.logger.logger.warning(f"[{pair}] [SUCCESS] POSITION RESTORED - Entry: ${avg_entry_price:.2f}, Qty: {base_holdings:.8f} {base_currency}")

                try:
                    last_trade_time = datetime.fromisoformat(last_trade.get('timestamp'))
                    self.strategies[pair].last_trade_time = last_trade_time
                    time_since_last_trade = (datetime.now() - last_trade_time).total_seconds()

                    if time_since_last_trade < self.strategies[pair].min_trade_interval_seconds:
                        remaining = self.strategies[pair].min_trade_interval_seconds - time_since_last_trade
                        self.logger.logger.warning(f"[{pair}] [COOLDOWN] Cooldown active: {remaining:.0f}s remaining")
                except Exception as e:
                    self.logger.logger.error(f"[{pair}] Failed to restore cooldown state: {e}")

            self.logger.logger.info("=" * 60)
            self.logger.logger.info("[SUCCESS] STRATEGY STATE RECOVERY COMPLETE FOR ALL PAIRS")
            self.logger.logger.info("=" * 60)

        except Exception as e:
            self.logger.logger.error(f"[ERROR] Failed to recover position state: {e}")
            import traceback
            traceback.print_exc()

    def get_pair_precision(self, pair: str) -> dict:
        precision_map = {
            'BTC/USD': {'amount': 5, 'price': 2, 'min_order': 1.0},
            'ETH/USD': {'amount': 4, 'price': 2, 'min_order': 1.0},
        }
        return precision_map.get(pair, {'amount': 5, 'price': 2, 'min_order': 1.0})

    def get_portfolio_value(self, balance_data: Dict, current_prices: Dict) -> float:
        try:
            usd_balance = balance_data.get('USD', {})
            cash = float(usd_balance.get('free', 0))
            total_value = cash

            for coin, balance in balance_data.items():
                if coin == 'USD' or not isinstance(balance, dict):
                    continue
                try:
                    free_amount = float(balance.get('free', 0))
                    coin_price = float(current_prices.get(coin, 0))
                    total_value += free_amount * coin_price
                except:
                    continue
            return total_value
        except Exception as e:
            self.logger.logger.error(f"Failed to calculate portfolio value: {e}")
            raise

    def crypto_risk_checks(self, current_price: float, balance_data: Dict, pair: str = None) -> bool:
        try:
            from datetime import date
            today = date.today()

            if self.last_trade_date != today:
                self.daily_trade_count = 0
                self.last_trade_date = today

            if self.daily_trade_count >= self.config.DAILY_TRADE_LIMIT:
                return False

            if pair:
                base_currency = pair.split('/')[0]
                coin_balance = balance_data.get(base_currency, {}).get('free', 0)

                current_prices = {}
                for supported_pair in self.config.SUPPORTED_PAIRS:
                    coin = supported_pair.split('/')[0]
                    if coin == base_currency:
                        current_prices[coin] = current_price
                    else:
                        current_prices[coin] = 0

                total_value = self.get_portfolio_value(balance_data, current_prices)

                if total_value > 0:
                    coin_percentage = ((coin_balance * current_price) / total_value) * 100
                    if coin_percentage > 85:
                        self.logger.logger.warning(f"[{pair}] [WARNING] Portfolio too concentrated ({coin_percentage:.1f}%)")
                        return False
            return True
        except Exception as e:
            return False

    def monitor_performance(self, current_portfolio_value: float):
        try:
            if current_portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = current_portfolio_value

            if self.peak_portfolio_value > 0:
                drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
                if drawdown >= self.config.DRAWDOWN_ALERT:
                    self.logger.logger.warning(f"[WARNING] DRAWDOWN ALERT: {drawdown*100:.1f}% from peak")

            if self.consecutive_losses >= self.config.CONSECUTIVE_LOSS_ALERT:
                self.logger.logger.warning(f"[WARNING] CONSECUTIVE LOSSES: {self.consecutive_losses} trades in a row")
        except Exception as e:
            pass

    def execute_trade(self, decision, balance_data: Dict, pair: str):
        try:
            symbol = pair
            base_currency = symbol.split('/')[0]
            quote_currency = symbol.split('/')[1]
            precision = self.get_pair_precision(pair)

            if decision.action == Action.BUY:
                if not self.crypto_risk_checks(decision.price, balance_data, pair):
                    return

                available_cash = balance_data.get(quote_currency, {}).get('free', 0)
                max_trade_value = available_cash * self.config.MAX_POSITION_SIZE
                quantity = max_trade_value / decision.price

                if quantity * decision.price < self.MIN_TRADE_VALUE:
                    return

                quantity = round(quantity, precision['amount'])

                result = self.roostoo.place_order(coin=base_currency, side='BUY', quantity=quantity)

                if 'error' not in result:
                    trade_data = {
                        'action': 'BUY', 'symbol': symbol, 'quantity': quantity,
                        'price': decision.price, 'total': quantity * decision.price, 'reason': decision.reason
                    }
                    self.logger.log_trade(trade_data)
                    self.strategies[pair].add_position(decision.price, quantity)
                    self.daily_trade_count += 1

            elif decision.action == Action.SELL:
                if getattr(decision, 'quantity', 0) > 0:
                    quantity = decision.quantity
                else:
                    available_coin = balance_data.get(base_currency, {}).get('free', 0)
                    quantity = available_coin * self.config.MAX_POSITION_SIZE

                if quantity * decision.price < self.MIN_TRADE_VALUE:
                    return

                quantity = round(quantity, precision['amount'])

                result = self.roostoo.place_order(coin=base_currency, side='SELL', quantity=quantity)

                if 'error' not in result:
                    trade_data = {
                        'action': 'SELL', 'symbol': symbol, 'quantity': quantity,
                        'price': decision.price, 'total': quantity * decision.price, 'reason': decision.reason
                    }
                    self.logger.log_trade(trade_data)
                    self.daily_trade_count += 1

                    if 'stop loss' in decision.reason.lower():
                        self.consecutive_losses += 1
                    else:
                        self.consecutive_losses = 0

        except Exception as e:
            self.logger.logger.error(f"[{pair}] Failed to execute trade: {e}")

    def execute_initial_trade(self, current_price: float, balance_data: Dict, pair: str):
        try:
            symbol = pair
            base_currency = symbol.split('/')[0]
            quote_currency = symbol.split('/')[1]
            precision = self.get_pair_precision(pair)

            trade_value = 1.14
            quantity = trade_value / current_price
            quantity = round(quantity, precision['amount'])

            available_cash = balance_data.get(quote_currency, {}).get('free', 0)
            required_cash = quantity * current_price

            if available_cash < required_cash:
                return False

            result = self.roostoo.place_order(coin=base_currency, side='BUY', quantity=quantity)

            if 'error' not in result:
                self.strategies[pair].add_position(current_price, quantity)
                trade_data = {
                    'action': 'BUY', 'symbol': symbol, 'quantity': quantity,
                    'price': current_price, 'total': quantity * current_price, 'reason': 'INITIAL TRADE: Competition requirement'
                }
                self.logger.log_trade(trade_data)
                return True
            return False
        except Exception as e:
            return False

    def run(self):
        self.logger.logger.info("Starting Fused Multi-Pair AI Trading Bot...")
        iteration = 0
        while self.running:
            try:
                iteration += 1

                # Fetch account balance once per iteration
                balance_data = self.roostoo.get_account_balance()
                if 'error' in balance_data:
                    self.logger.logger.error(f"Failed to fetch balance: {balance_data.get('error')}")
                    time.sleep(60)
                    continue

                # Fetch current prices for all pairs
                current_prices = {}
                for p in self.config.SUPPORTED_PAIRS:
                    coin = p.split('/')[0]
                    current_prices[coin] = self.horus.get_current_price(coin)

                for pair in self.config.SUPPORTED_PAIRS:
                    try:
                        # Fetch klines and build DataFrame
                        coin = pair.split('/')[0]
                        current_price = current_prices.get(coin, 0)
                        if current_price == 0:
                            self.logger.logger.warning(f"[{pair}] Could not fetch price, skipping.")
                            continue

                        df = self.horus.get_klines(symbol=coin, interval='15m', limit=100)
                        if df.empty:
                            self.logger.logger.warning(f"[{pair}] Empty klines data, skipping.")
                            continue

                        # --- [STEP 3] AI SIGNAL GENERATION & VETO ---

                        # 1. Get Live Sentiment (Brain 1)
                        sentiment_score = self.sentiment_analyzer.get_crypto_sentiment()

                        # 2. Get XGBoost Prediction (Brain 2)
                        ml_signal, ml_confidence = 0, 0.0
                        if self.ml_model is not None and len(df) > 25:
                            # Feature Engineering (Same as your training script)
                            delta = df['close'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                            df['rsi'] = 100 - (100 / (1 + (gain / loss)))
                            df['high_low'] = df['high'] - df['low']
                            df['atr'] = df['high_low'].rolling(14).mean()
                            df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
                            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
                            df['trend_score'] = (df['ema_8'] - df['ema_21']) / df['ema_21']

                            # Prepare data for XGBoost Booster
                            # Cast to float32 to avoid object dtype error from NaN-affected rolling columns
                            current_features = df.iloc[-1][['rsi', 'trend_score', 'atr']].to_frame().T
                            current_features = current_features.astype('float32')
                            dmatrix = xgb.DMatrix(current_features)

                            # Predict probability
                            probs = self.ml_model.predict(dmatrix)[0]
                            ml_signal = 1 if probs > 0.5 else 0
                            ml_confidence = float(probs if ml_signal == 1 else 1 - probs)

                        # 3. Apply the Veto Logic
                        # We send these to your GridStrategy.analyze method
                        decision_dict = self.strategies[pair].analyze(
                            klines_data=df,
                            current_price=current_price,
                            ml_signal=(1 if ml_signal == 1 else -1),
                            ml_confidence=ml_confidence,
                            sentiment_score=sentiment_score
                        )

                        # 4. Filter the Decision based on AI Consensus
                        # If technicals say BUY, but Sentiment is Bearish or ML Confidence is too low, we HOLD.
                        if decision_dict['action'] == 'OPEN_POSITION':
                            if sentiment_score < 0.0:
                                decision_dict['action'] = 'HOLD'
                                decision_dict['reason'] = "VETO: Bearish News Sentiment"
                            elif ml_confidence < 0.6:  # 60% confidence threshold
                                decision_dict['action'] = 'HOLD'
                                decision_dict['reason'] = f"VETO: Low ML Confidence ({ml_confidence:.2f})"

                        # Map back to Cw's expected object format
                        class TempDecision: pass
                        decision = TempDecision()
                        decision.action = Action.BUY if decision_dict['action'] == 'OPEN_POSITION' else (Action.SELL if decision_dict['action'] in ['CLOSE_ALL', 'REVERSE_POSITION'] else Action.HOLD)
                        decision.confidence = ml_confidence
                        decision.reason = decision_dict['reason']
                        decision.price = current_price
                        decision.quantity = decision_dict.get('quantity', 0)

                        self.logger.logger.info(f"[{pair}] AI Decision: {decision.action.value} | Conf: {decision.confidence:.2f} | Reason: {decision.reason}")

                        if not self.initial_trades_executed[pair]:
                            if self.execute_initial_trade(current_price, balance_data, pair):
                                self.initial_trades_executed[pair] = True

                        if decision.action != Action.HOLD:
                            self.execute_trade(decision, balance_data, pair)

                    except Exception as e:
                        self.logger.logger.error(f"[{pair}] Error processing pair: {e}")
                        continue

                portfolio_value = self.get_portfolio_value(balance_data, current_prices)
                self.monitor_performance(portfolio_value)

                self.logger.logger.info(f"Iteration {iteration} complete. Waiting {self.config.TRADE_INTERVAL}s...")
                time.sleep(self.config.TRADE_INTERVAL)

            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                time.sleep(60)


if __name__ == "__main__":
    bot = TradingBot(enable_dashboard=False)
    bot.run()