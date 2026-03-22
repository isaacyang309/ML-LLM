#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Dynamic pathing to find the project root
# This gets the directory where main.py is (src) and then goes up one level
file_path = Path(__file__).resolve()
project_root = str(file_path.parents[1])

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import threading
import queue
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

from enum import Enum

class Action(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class DataFetcherThread(threading.Thread):
    """
    Separate thread for fetching market data from APIs.
    Runs continuously and puts data into a queue for the main trading thread.
    """

    def __init__(self, roostoo, horus, config, data_queue, logger):
        super().__init__()
        self.roostoo = roostoo
        self.horus = horus
        self.config = config
        self.data_queue = data_queue
        self.logger = logger
        self.running = True
        self.daemon = True

    def run(self):
        """Main data fetching loop"""
        self.logger.logger.info("[DataFetcher] Thread started")

        while self.running:
            try:
                data_batch = {
                    'timestamp': time.time(),
                    'balance': None,
                    'prices': {},
                    'klines': {},
                    'errors': []
                }

                # 1. Fetch account balance
                try:
                    balance = self.roostoo.get_account_balance()
                    if 'error' not in balance and isinstance(balance, dict):
                        data_batch['balance'] = balance
                    else:
                        data_batch['errors'].append(f"Balance error: {balance.get('error', 'Unknown')}")
                except Exception as e:
                    data_batch['errors'].append(f"Balance fetch error: {e}")

                # 2. Fetch current prices and klines for each trading pair
                for pair in self.config.SUPPORTED_PAIRS:
                    try:
                        coin = pair.split('/')[0]
                        price = self.horus.get_current_price(coin)
                        if price and price > 0:
                            data_batch['prices'][coin] = price
                        else:
                            data_batch['errors'].append(f"Price fetch error for {pair}")
                    except Exception as e:
                        data_batch['errors'].append(f"Price fetch error for {pair}: {e}")

                    try:
                        coin = pair.split('/')[0]
                        df = self.horus.get_klines(symbol=coin, interval='15m', limit=100)
                        if not df.empty:
                            data_batch['klines'][pair] = df
                        else:
                            data_batch['errors'].append(f"Empty klines for {pair}")
                    except Exception as e:
                        data_batch['errors'].append(f"Klines fetch error for {pair}: {e}")

                # Put data in queue, drop oldest if full
                try:
                    while self.data_queue.qsize() > 10:
                        try:
                            self.data_queue.get_nowait()
                        except:
                            break
                    self.data_queue.put(data_batch, timeout=5)
                except:
                    pass

                fetch_interval = getattr(self.config, 'DATA_FETCH_INTERVAL', 60)
                time.sleep(fetch_interval)

            except Exception as e:
                self.logger.logger.error(f"[DataFetcher] Thread error: {e}")
                time.sleep(5)

        self.logger.logger.info("[DataFetcher] Thread stopped")

    def stop(self):
        """Stop the thread gracefully"""
        self.running = False


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
            # Add compatibility attributes for cooldown logic
            self.strategies[pair].min_trade_interval_seconds = 300
            self.strategies[pair].last_trade_time = datetime.now()

        self.running = True
        self.enable_dashboard = enable_dashboard
        self.dashboard_thread = None
        self.data_queue = queue.Queue(maxsize=20)
        self.data_fetcher = None

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
        if not getattr(self.config, 'SAFE_BASELINE_MODE', True):
            self._recover_position_state()
        else:
            self.logger.logger.info("SAFE_BASELINE_MODE enabled: skipping strategy state recovery")

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
        """
        Recover strategy position state from actual balance and trade logs.
        Tries SQLite database first, falls back to trade_history.json.
        """
        try:
            self.logger.logger.info("=" * 60)
            self.logger.logger.info("CHECKING FOR EXISTING POSITIONS TO RECOVER")
            self.logger.logger.info("=" * 60)

            balance = self.roostoo.get_account_balance()
            if 'error' in balance:
                self.logger.logger.warning(f"Could not fetch balance for recovery: {balance.get('error')}")
                return

            # Try SQLite first, fall back to JSON
            import sqlite3
            db_path = self.logger.logs_dir / 'trading_bot.db'
            use_sqlite = db_path.exists()

            conn = None
            if use_sqlite:
                try:
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                except Exception as e:
                    self.logger.logger.warning(f"Could not open SQLite DB: {e}, falling back to JSON")
                    use_sqlite = False

            for pair in self.config.SUPPORTED_PAIRS:
                base_currency = pair.split('/')[0]
                base_holdings = balance.get(base_currency, {}).get('free', 0)

                self.logger.logger.info(f"[{pair}] Current Balance: {base_holdings:.8f} {base_currency}")

                if base_holdings < 0.0001:
                    self.logger.logger.info(f"[{pair}] No significant position detected - starting fresh")
                    continue

                self.logger.logger.warning(f"[{pair}] [WARNING] EXISTING POSITION DETECTED: {base_holdings:.8f} {base_currency}")

                # Trades as normalized tuples: (action, quantity, price, total, timestamp)
                buy_trades = []
                last_trade = None

                if use_sqlite:
                    try:
                        cursor.execute("""
                            SELECT action, quantity, price, total, timestamp
                            FROM trades
                            WHERE symbol = ?
                            ORDER BY timestamp ASC
                        """, (pair,))
                        all_trades = cursor.fetchall()
                        self.logger.logger.info(f"[{pair}] Found {len(all_trades)} trades in SQLite database")
                        buy_trades = [t for t in all_trades if t[0] == 'BUY']
                        if all_trades:
                            last_trade = all_trades[-1]
                    except Exception as e:
                        self.logger.logger.error(f"[{pair}] SQLite query failed: {e}")
                else:
                    import json
                    trade_log_path = self.logger.logs_dir / 'trade_history.json'
                    if not trade_log_path.exists():
                        self.logger.logger.error(f"[{pair}] [ERROR] Position exists but no trade history found!")
                        continue
                    with open(trade_log_path, 'r') as f:
                        all_trades_json = json.load(f)
                    pair_trades = [t for t in all_trades_json if t.get('symbol') == pair]
                    buy_trades = [
                        (t['action'], t['quantity'], t['price'], t['total'], t['timestamp'])
                        for t in pair_trades if t.get('action') == 'BUY'
                    ]
                    if pair_trades:
                        lt = pair_trades[-1]
                        last_trade = (lt['action'], lt['quantity'], lt['price'], lt['total'], lt['timestamp'])

                if not buy_trades:
                    self.logger.logger.error(f"[{pair}] No BUY trades found in history!")
                    continue

                total_bought_qty = sum(t[1] for t in buy_trades)
                total_bought_value = sum(t[3] for t in buy_trades)
                avg_entry_price = total_bought_value / total_bought_qty if total_bought_qty > 0 else buy_trades[-1][2]

                # Restore position in GridStrategy
                self.strategies[pair].add_position(avg_entry_price, base_holdings)
                self.strategies[pair].current_direction = GridDirection.UP

                self.logger.logger.warning(
                    f"[{pair}] [SUCCESS] POSITION RESTORED - Entry: ${avg_entry_price:.2f}, "
                    f"Qty: {base_holdings:.8f} {base_currency}"
                )

                # Restore cooldown from last trade timestamp
                if last_trade:
                    try:
                        last_trade_time_str = last_trade[4]
                        if isinstance(last_trade_time_str, str):
                            last_trade_dt = datetime.fromisoformat(last_trade_time_str.replace('Z', '+00:00'))
                        else:
                            last_trade_dt = datetime.fromisoformat(str(last_trade_time_str))

                        self.strategies[pair].last_trade_time = last_trade_dt
                        time_since = (datetime.now() - last_trade_dt).total_seconds()

                        if time_since < self.strategies[pair].min_trade_interval_seconds:
                            remaining = self.strategies[pair].min_trade_interval_seconds - time_since
                            self.logger.logger.warning(f"[{pair}] [COOLDOWN] Cooldown active: {remaining:.0f}s remaining")
                        else:
                            self.logger.logger.info(f"[{pair}] Cooldown expired - ready to trade")
                    except Exception as e:
                        self.logger.logger.error(f"[{pair}] Failed to restore cooldown state: {e}")

            if conn:
                conn.close()

            self.logger.logger.info("=" * 60)
            self.logger.logger.info("[SUCCESS] STRATEGY STATE RECOVERY COMPLETE FOR ALL PAIRS")
            self.logger.logger.info("=" * 60)

        except Exception as e:
            self.logger.logger.error(f"[ERROR] Failed to recover position state: {e}")
            import traceback
            traceback.print_exc()

    def get_pair_precision(self, pair: str) -> dict:
        """
        Get precision settings for a specific trading pair.
        Based on Roostoo API specs:
        - BTC/USD: AmountPrecision=5, PricePrecision=2, MiniOrder=1
        - ETH/USD: AmountPrecision=4, PricePrecision=2, MiniOrder=1
        """
        precision_map = {
            'BTC/USD': {'amount': 5, 'price': 2, 'min_order': 1.0},
            'ETH/USD': {'amount': 4, 'price': 2, 'min_order': 1.0},
        }
        return precision_map.get(pair, {'amount': 5, 'price': 2, 'min_order': 1.0})

    def get_portfolio_value(self, balance_data: Dict, current_prices: Dict) -> float:
        """Calculate total portfolio value"""
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
                except (TypeError, ValueError) as e:
                    self.logger.logger.error(f"Error calculating value for {coin}: {e}")
                    continue

            return total_value
        except Exception as e:
            self.logger.logger.error(f"Failed to calculate portfolio value: {e}")
            raise

    def crypto_risk_checks(self, current_price: float, balance_data: Dict, pair: str = None) -> bool:
        """
        Crypto-specific risk management checks.
        Returns True if trade is allowed, False otherwise.
        """
        try:
            from datetime import date
            today = date.today()

            if self.last_trade_date != today:
                self.daily_trade_count = 0
                self.last_trade_date = today

            if self.daily_trade_count >= self.config.DAILY_TRADE_LIMIT:
                self.logger.logger.warning(
                    f"[WARNING] Daily trade limit reached ({self.config.DAILY_TRADE_LIMIT}), skipping trade"
                )
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
                        self.logger.logger.warning(
                            f"[{pair}] [WARNING] Portfolio too concentrated ({coin_percentage:.1f}%), skipping BUY"
                        )
                        return False

            return True

        except Exception as e:
            self.logger.logger.error(f"Error in crypto_risk_checks: {e}")
            return False

    def monitor_performance(self, current_portfolio_value: float):
        """Monitor performance metrics for alerts."""
        try:
            if current_portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = current_portfolio_value

            if self.peak_portfolio_value > 0:
                drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
                if drawdown >= self.config.DRAWDOWN_ALERT:
                    self.logger.logger.warning(
                        f"[WARNING] DRAWDOWN ALERT: {drawdown*100:.1f}% from peak "
                        f"(Peak: ${self.peak_portfolio_value:.2f}, Current: ${current_portfolio_value:.2f})"
                    )

            if self.consecutive_losses >= self.config.CONSECUTIVE_LOSS_ALERT:
                self.logger.logger.warning(
                    f"[WARNING] CONSECUTIVE LOSSES: {self.consecutive_losses} trades in a row"
                )

        except Exception as e:
            self.logger.logger.error(f"Error in monitor_performance: {e}")

    def execute_trade(self, decision, balance_data: Dict, pair: str):
        """Execute trade with crypto risk checks for a specific trading pair"""
        try:
            symbol = pair
            base_currency = symbol.split('/')[0]
            quote_currency = symbol.split('/')[1]
            precision = self.get_pair_precision(pair)

            if decision.action == Action.BUY:
                if not self.crypto_risk_checks(decision.price, balance_data, pair):
                    return

                if getattr(decision, 'quantity', 0) > 0:
                    quantity = decision.quantity
                else:
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
                        'price': decision.price, 'total': quantity * decision.price,
                        'reason': decision.reason
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
                        'price': decision.price, 'total': quantity * decision.price,
                        'reason': decision.reason
                    }
                    self.logger.log_trade(trade_data)
                    self.daily_trade_count += 1

                    if 'stop loss' in decision.reason.lower():
                        self.consecutive_losses += 1
                    else:
                        self.consecutive_losses = 0

        except Exception as e:
            self.logger.logger.error(f"[{pair}] Failed to execute trade: {e}")

    def execute_initial_trade(self, pair: str = 'BTC/USD', current_price: float = None, balance_data: Dict = None):
        """Execute one-time $1.14 BUY trade to verify bot is deployed successfully"""
        try:
            symbol = pair
            base_currency = symbol.split('/')[0]
            quote_currency = symbol.split('/')[1]

            # Fetch balance if not provided
            if balance_data is None:
                balance_data = self.roostoo.get_account_balance()
                if 'error' in balance_data or not isinstance(balance_data, dict):
                    self.logger.logger.error(f"[{pair}] Failed to fetch balance for initial trade")
                    return False

            # Fetch price if not provided
            if current_price is None:
                current_price = self.horus.get_current_price(base_currency)

            if not current_price or current_price == 0:
                self.logger.logger.error(f"[{pair}] Failed to get price for initial trade")
                return False

            precision = self.get_pair_precision(pair)
            trade_value = 1.14
            quantity = trade_value / current_price
            quantity = round(quantity, precision['amount'])

            available_cash = balance_data.get(quote_currency, {}).get('free', 0)
            required_cash = quantity * current_price

            if available_cash < required_cash:
                self.logger.logger.error(
                    f"[{pair}] Insufficient balance for initial trade: "
                    f"need ${required_cash:.2f}, have ${available_cash:.2f}"
                )
                return False

            self.logger.logger.info(
                f"[{pair}] INITIAL TRADE: Executing $1.14 BUY - "
                f"{quantity:.5f} {base_currency} @ ${current_price:.2f}"
            )

            result = self.roostoo.place_order(coin=base_currency, side='BUY', quantity=quantity)

            if 'error' not in result:
                self.strategies[pair].add_position(current_price, quantity)
                trade_data = {
                    'action': 'BUY', 'symbol': symbol, 'quantity': quantity,
                    'price': current_price, 'total': quantity * current_price,
                    'reason': 'INITIAL TRADE: Competition requirement'
                }
                self.logger.log_trade(trade_data)
                self.logger.logger.info(f"[{pair}] Initial trade executed successfully!")
                return True
            else:
                self.logger.logger.error(f"[{pair}] Initial trade failed: {result.get('error')}")
                return False

        except Exception as e:
            self.logger.logger.error(f"Failed to execute initial trade for {pair}: {e}")
            return False

    def _run_safe_baseline_mode(self):
        """
        Safe baseline runtime:
        - Execute one BTC test trade at startup
        - Keep process alive without strategy analysis or follow-up trading
        """
        self.logger.logger.info("=" * 60)
        self.logger.logger.info("Starting SAFE baseline mode (test-only)")
        self.logger.logger.info("Behavior: one BTC initial trade, then passive heartbeat")
        self.logger.logger.info("=" * 60)

        try:
            if 'BTC/USD' in self.config.SUPPORTED_PAIRS:
                self.logger.logger.info("Executing one-time test trade at startup")
                self.execute_initial_trade(pair='BTC/USD')
            else:
                self.logger.logger.warning("BTC/USD not in SUPPORTED_PAIRS, skipping test trade")

            while self.running:
                try:
                    self.logger.logger.info("SAFE baseline mode heartbeat: no strategy trades are executed")
                    time.sleep(self.config.TRADE_INTERVAL)
                except KeyboardInterrupt:
                    self.logger.logger.info("User interrupted, stopping bot...")
                    self.running = False

        finally:
            self.logger.logger.info("Bot stopped.")

    def _run_full_trading_mode(self):
        """Main trading loop with threaded data fetching and AI signal generation"""

        # One-time initial trades at startup for all pairs
        for pair in self.config.SUPPORTED_PAIRS:
            if not self.initial_trades_executed[pair]:
                if self.execute_initial_trade(pair=pair):
                    self.initial_trades_executed[pair] = True

        # Start the data fetcher thread
        self.data_fetcher = DataFetcherThread(
            roostoo=self.roostoo,
            horus=self.horus,
            config=self.config,
            data_queue=self.data_queue,
            logger=self.logger
        )
        self.data_fetcher.start()

        self.logger.logger.info("=" * 60)
        self.logger.logger.info("Starting Fused Multi-Pair AI Trading Bot with threaded data fetching...")
        self.logger.logger.info(f"Trading pairs: {', '.join(self.config.SUPPORTED_PAIRS)}")
        self.logger.logger.info("=" * 60)

        iteration = 0
        while self.running:
            try:
                # Get latest data batch from the fetcher thread
                try:
                    data_batch = self.data_queue.get(timeout=30)
                except queue.Empty:
                    self.logger.logger.warning("No data received from fetcher thread for 30 seconds")
                    continue

                iteration += 1
                self.logger.logger.info(f"\n{'=' * 80}")
                self.logger.logger.info(f"Starting iteration {iteration}...")
                self.logger.logger.info(f"{'=' * 80}")

                # Log any errors from the data fetcher
                for error in data_batch.get('errors', []):
                    self.logger.logger.warning(f"[DataFetcher] {error}")

                balance_data = data_batch.get('balance', {})
                current_prices = data_batch.get('prices', {})
                klines_data = data_batch.get('klines', {})

                # Validate balance
                if not balance_data or 'error' in balance_data:
                    self.logger.logger.error("Invalid balance data, skipping iteration")
                    continue

                if 'USD' not in balance_data:
                    self.logger.logger.error("Balance data missing USD, skipping iteration")
                    continue

                for pair in self.config.SUPPORTED_PAIRS:
                    try:
                        self.logger.logger.info(f"\n--- Processing {pair} ---")

                        coin = pair.split('/')[0]
                        current_price = current_prices.get(coin, 0)
                        if current_price == 0:
                            self.logger.logger.warning(f"[{pair}] Could not fetch price, skipping.")
                            continue

                        self.logger.logger.info(f"[{pair}] Current price: ${current_price:.2f}")

                        df = klines_data.get(pair)
                        if df is None or df.empty:
                            self.logger.logger.warning(f"[{pair}] Empty klines data, skipping.")
                            continue

                        # --- AI SIGNAL GENERATION ---

                        # 1. Get Live Sentiment (Brain 1)
                        sentiment_score = self.sentiment_analyzer.get_crypto_sentiment()

                        # 2. Get XGBoost Prediction (Brain 2)
                        ml_signal, ml_confidence = 0, 0.0
                        if self.ml_model is not None and len(df) > 25:
                            # Feature Engineering
                            delta = df['close'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                            df['rsi'] = 100 - (100 / (1 + (gain / loss)))
                            df['high_low'] = df['high'] - df['low']
                            df['atr'] = df['high_low'].rolling(14).mean()
                            df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
                            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
                            df['trend_score'] = (df['ema_8'] - df['ema_21']) / df['ema_21']

                            # Cast to float32 to avoid object dtype error from NaN-affected rolling columns
                            current_features = df.iloc[-1][['rsi', 'trend_score', 'atr']].to_frame().T
                            current_features = current_features.astype('float32')
                            dmatrix = xgb.DMatrix(current_features)

                            probs = self.ml_model.predict(dmatrix)[0]
                            ml_signal = 1 if probs > 0.5 else 0
                            ml_confidence = float(probs if ml_signal == 1 else 1 - probs)

                        # 3. Run GridStrategy analysis with AI inputs
                        decision_dict = self.strategies[pair].analyze(
                            klines_data=df,
                            current_price=current_price,
                            ml_signal=(1 if ml_signal == 1 else -1),
                            ml_confidence=ml_confidence,
                            sentiment_score=sentiment_score
                        )

                        # 4. Apply AI Consensus Veto
                        # If technicals say BUY but sentiment is bearish or ML confidence is low, HOLD.
                        if decision_dict['action'] == 'OPEN_POSITION':
                            if sentiment_score < 0.0:
                                decision_dict['action'] = 'HOLD'
                                decision_dict['reason'] = "VETO: Bearish News Sentiment"
                            elif ml_confidence < 0.6:
                                decision_dict['action'] = 'HOLD'
                                decision_dict['reason'] = f"VETO: Low ML Confidence ({ml_confidence:.2f})"

                        # Map decision dict to decision object
                        class TempDecision: pass
                        decision = TempDecision()
                        decision.action = (
                            Action.BUY if decision_dict['action'] == 'OPEN_POSITION'
                            else Action.SELL if decision_dict['action'] in ['CLOSE_ALL', 'REVERSE_POSITION']
                            else Action.HOLD
                        )
                        decision.confidence = ml_confidence
                        decision.reason = decision_dict['reason']
                        decision.price = current_price
                        decision.quantity = decision_dict.get('quantity', 0)

                        self.logger.logger.info(
                            f"[{pair}] AI Decision: {decision.action.value} | "
                            f"Conf: {decision.confidence:.2f} | Reason: {decision.reason}"
                        )

                        # Log strategy signal
                        self.logger.log_strategy_signal({
                            'symbol': pair,
                            'action': decision.action.value,
                            'confidence': decision.confidence,
                            'price': current_price,
                            'reason': decision.reason
                        })

                        if decision.action != Action.HOLD:
                            self.execute_trade(decision, balance_data, pair)

                    except Exception as e:
                        self.logger.logger.error(f"[{pair}] Error processing pair: {e}")
                        continue

                # Log portfolio status
                portfolio_value = self.get_portfolio_value(balance_data, current_prices)
                self.logger.log_portfolio_update({
                    'total_value': portfolio_value,
                    'cash_value': balance_data.get('USD', {}).get('free', 0),
                    'btc_balance': balance_data.get('BTC', {}).get('free', 0),
                    'btc_value': balance_data.get('BTC', {}).get('free', 0) * current_prices.get('BTC', 0),
                    'eth_balance': balance_data.get('ETH', {}).get('free', 0),
                    'eth_value': balance_data.get('ETH', {}).get('free', 0) * current_prices.get('ETH', 0),
                    'btc_price': current_prices.get('BTC', 0),
                    'eth_price': current_prices.get('ETH', 0)
                })

                self.monitor_performance(portfolio_value)

                self.logger.logger.info(f"\n{'=' * 80}")
                self.logger.logger.info(f"Iteration {iteration} complete.")
                self.logger.logger.info(f"{'=' * 80}\n")

            except KeyboardInterrupt:
                self.logger.logger.info("User interrupted, stopping bot...")
                self.running = False

            except Exception as e:
                self.logger.logger.error(f"Main loop error: {e}")
                time.sleep(5)

        # Clean up: stop data fetcher thread
        if self.data_fetcher:
            self.data_fetcher.stop()
            self.data_fetcher.join(timeout=5)

        self.logger.logger.info("Bot stopped.")

    def run(self):
        """Runtime entrypoint with safety toggle for fast deploys."""
        if getattr(self.config, 'SAFE_BASELINE_MODE', True):
            self._run_safe_baseline_mode()
        else:
            self._run_full_trading_mode()


if __name__ == "__main__":
    bot = TradingBot(enable_dashboard=False)
    bot.run()