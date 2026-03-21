"""
Grid Trading Bot - Integrates GridStrategy with the existing trading infrastructure
"""

# Try to load the LLM Sentiment Analyzer
try:
    from src.sentiment_analyzer import SentimentAnalyzer
except ImportError:
    SentimentAnalyzer = None

import sys
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, Optional

# Setup path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.grid_strategy import GridStrategy, GridTradeConfig, GridDirection
from src.horus_client import HorusClient
from src.trading_logger import TradingLogger
from src.roostoo_client import RoostooClient
from config.config import Config


class GridTradingBot:
    """
    Trading bot that uses the adaptive grid strategy
    
    This bot:
    - Fetches market data from Binance via HorusClient
    - Executes trades via RoostooClient
    - Manages positions using GridStrategy
    """
    
    def __init__(self, pair: str = "BTC/USD", enable_trading: bool = False):
        """
        Initialize grid trading bot
        """
        self.config = Config()
        self.logger = TradingLogger()
        self.market_data_client = HorusClient()
        self.trading_client = RoostooClient()
        self.pair = pair
        self.enable_trading = enable_trading
        
        # --- NEW: Initialize LLM ---
        self.sentiment_analyzer = SentimentAnalyzer() if SentimentAnalyzer else None
        
        # Extract base currency
        self.base_currency = pair.split('/')[0]
        
        # Initialize grid strategy with default config
        grid_config = GridTradeConfig(
            unit_amount=0.01,
            direction_multiplier=2.0,
            stop_loss=1000.0,
            stop_profit=300.0,
            lookback_hours=24.0,
            drawdown_threshold=1000.0,
            price_climb_high_volatility=100.0,
            price_climb_low_volatility=200.0,
            initial_lot_sizes=[0.01, 0.02, 0.03, 0.04, 0.09],
        )
        self.grid_strategy = GridStrategy(config=grid_config)
        
        self.running = True
        
        self.logger.logger.info(f"Grid Trading Bot initialized for {pair}, trading_enabled={enable_trading}")
    
    def update_grid_config(self, **kwargs):
        """Update grid strategy configuration"""
        for key, value in kwargs.items():
            if hasattr(self.grid_strategy.config, key):
                setattr(self.grid_strategy.config, key, value)
                self.logger.logger.info(f"Updated grid config: {key} = {value}")
    
    def set_initial_lots(self, lot_string: str):
        """Set initial lot sizes from string"""
        self.grid_strategy.parse_initial_lots(lot_string)
    
    def fetch_market_data(self) -> Dict:
        """Fetch current market data and price history"""
        try:
            # Get current price
            current_price = self.market_data_client.get_current_price(self.base_currency)
            if current_price <= 0:
                self.logger.logger.error(f"Invalid price fetched: {current_price}")
                return None
            
            # Get price history (100 candles of 15m)
            end_time = int(time.time())
            start_time = end_time - (15 * 60 * 100)
            
            klines_list = self.market_data_client.get_price_history(
                symbol=self.base_currency,
                interval='15m',
                start=start_time,
                end=end_time
            )
            
            if not klines_list or 'error' in klines_list:
                self.logger.logger.error(f"Failed to fetch klines: {klines_list}")
                return None
            
            # Convert to DataFrame for strategy
            import pandas as pd
            klines_df = pd.DataFrame(klines_list)
            
            return {
                'current_price': current_price,
                'klines': klines_df,
                'timestamp': datetime.now()
            }
        
        except Exception as e:
            self.logger.logger.error(f"Error fetching market data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def execute_trade(self, action: Dict) -> bool:
        """Execute trade action"""
        action_type = action.get('action')
        
        if action_type == 'HOLD':
            return True
        if action_type == 'CLOSE_ALL':
            return self.close_all_positions(action)
        if action_type == 'OPEN_POSITION':
            return self.open_position(action)
        if action_type == 'REVERSE_POSITION':
            return self.reverse_position(action)
            
        return False
    
    def open_position(self, action: Dict) -> bool:
        """Open a new grid position"""
        try:
            direction = action['direction']
            quantity = action['quantity']
            price = action['price']
            
            self.logger.logger.info(
                f"[{self.pair}] Opening {direction} position: qty={quantity:.8f} at ${price:.2f}"
            )
            
            if self.enable_trading:
                side = 'BUY' if direction == 'UP' else 'SELL'
                result = self.trading_client.place_order(
                    symbol=self.pair,
                    side=side,
                    quantity=quantity
                )
                
                if 'error' in result:
                    self.logger.logger.error(f"Trade execution failed: {result}")
                    return False
                
                order_id = result.get('order_id')
            else:
                order_id = f"SIM_{int(time.time())}"
                self.logger.logger.info(f"[SIMULATION] Would execute order: {order_id}")
            
            # Track in strategy
            self.grid_strategy.add_position(price, quantity, order_id)
            self.grid_strategy.last_trade_price = price
            
            # Log
            signal_data = {
                'symbol': self.pair,
                'action': 'GRID_' + direction,
                'quantity': quantity,
                'price': price,
                'reason': action.get('reason', '')
            }
            self.logger.log_strategy_signal(signal_data)
            
            return True
            
        except Exception as e:
            self.logger.logger.error(f"Error opening position: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def reverse_position(self, action: Dict) -> bool:
        """Reverse positions (go opposite direction)"""
        try:
            direction = action['direction']
            quantity = action['quantity']
            price = action['price']
            
            self.logger.logger.info(
                f"[{self.pair}] Reversing to {direction} position: qty={quantity:.8f} at ${price:.2f}"
            )
            
            if self.enable_trading:
                side = 'BUY' if direction == 'UP' else 'SELL'
                result = self.trading_client.place_order(
                    symbol=self.pair,
                    side=side,
                    quantity=quantity
                )
                
                if 'error' in result:
                    self.logger.logger.error(f"Trade execution failed: {result}")
                    return False
                
                order_id = result.get('order_id')
            else:
                order_id = f"SIM_{int(time.time())}"
                self.logger.logger.info(f"[SIMULATION] Would execute reversal order: {order_id}")
            
            # Track in strategy
            self.grid_strategy.add_position(price, quantity, order_id)
            self.grid_strategy.last_trade_price = price
            
            # Log
            signal_data = {
                'symbol': self.pair,
                'action': 'GRID_REVERSE_' + direction,
                'quantity': quantity,
                'price': price,
                'reason': action.get('reason', '')
            }
            self.logger.log_strategy_signal(signal_data)
            
            return True
            
        except Exception as e:
            self.logger.logger.error(f"Error reversing position: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def close_all_positions(self, action: Dict) -> bool:
        """Close all open positions"""
        try:
            close_price = action.get('close_price', 0)
            reason = action.get('reason', '')
            
            self.logger.logger.warning(
                f"[{self.pair}] Closing all positions at ${close_price:.2f}. Reason: {reason}"
            )
            
            if self.enable_trading:
                balance = self.trading_client.get_account_balance()
                if 'error' in balance:
                    self.logger.logger.error(f"Failed to get balance: {balance}")
                    return False
                
                base = self.base_currency
                holdings = balance.get(base, {}).get('free', 0)
                
                if holdings > 0:
                    result = self.trading_client.place_order(
                        symbol=self.pair,
                        side='SELL',
                        quantity=holdings
                    )
                    if 'error' in result:
                        self.logger.logger.error(f"Failed to close positions: {result}")
                        return False
            else:
                self.logger.logger.info(f"[SIMULATION] Would close all positions")
            
            # Update strategy state
            self.grid_strategy.close_all_positions(close_price, reason)
            
            # Log
            signal_data = {
                'symbol': self.pair,
                'action': 'CLOSE_ALL',
                'price': close_price,
                'reason': reason
            }
            self.logger.log_strategy_signal(signal_data)
            
            return True
            
        except Exception as e:
            self.logger.logger.error(f"Error closing positions: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_ml_predictions(self, klines_df) -> dict:
            try:
                import xgboost as xgb
                import numpy as np
                import pandas as pd
                
                # Load the latest Trend Brain
                model = xgb.XGBClassifier()
                model.load_model('xgb_model.json')
                
                df = klines_df.copy()
                df.columns = [str(col).lower() for col in df.columns]
                
                # 1. NEW FEATURE ENGINEERING (Matches train_xgboost.py exactly)
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                df['rsi'] = 100 - (100 / (1 + (gain / loss)))
                
                df['high_low'] = df['high'] - df['low']
                df['atr'] = df['high_low'].rolling(14).mean()
                
                df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
                df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
                df['trend_score'] = (df['ema_8'] - df['ema_21']) / df['ema_21']
                
                # 2. Prepare the single latest row for prediction
                features = ['rsi', 'trend_score', 'atr']
                latest_data = df[features].iloc[-1:]
                
                # 3. Prediction Logic
                probs = model.predict_proba(latest_data)[0]
                max_prob = np.max(probs)
                predicted_class = np.argmax(probs) # 0 = Down, 1 = Up
                
                # Confidence Gate (52% minimum)
                if max_prob < 0.52:
                    return {'signal': 0, 'confidence': max_prob, 'volatility': float(df['atr'].iloc[-1])}

                # Map to Strategy signals: Class 0 -> -1 (Down), Class 1 -> 1 (Up)
                signal_map = {0: -1, 1: 1}
                bot_signal = signal_map[predicted_class]
                
                self.logger.logger.info(f"[TREND BRAIN] Signal: {bot_signal} | Conf: {max_prob*100:.1f}%")
                
                return {
                    'signal': bot_signal, 
                    'confidence': max_prob, 
                    'volatility': float(df['atr'].iloc[-1])
                }
                
            except Exception as e:
                self.logger.logger.error(f"ML Prediction Error: {e}")
                return {'signal': 0, 'confidence': 0.0, 'volatility': 0.0}
        
    def run_cycle(self) -> bool:
        """Run one trading cycle"""
        try:
            # Fetch market data
            market_data = self.fetch_market_data()
            if not market_data:
                self.logger.logger.warning("Failed to fetch market data, skipping cycle")
                return False
            
            current_price = market_data['current_price']
            klines = market_data['klines']
            
            # --- Fetch ML and LLM Data ---
            ml_preds = self.get_ml_predictions(klines)
            
            sentiment = 0.0
            if self.sentiment_analyzer:
                sample_news = "Bitcoin market remains relatively stable ahead of the weekend."
                sentiment = self.sentiment_analyzer.get_crypto_sentiment(sample_news)
                self.logger.logger.info(f"[{self.pair}] FinBERT Sentiment Score: {sentiment}")
            
            # Analyze with upgraded strategy
            action = self.grid_strategy.analyze(
                klines_data=klines, 
                current_price=current_price,
                ml_signal=ml_preds['signal'],
                ml_confidence=ml_preds['confidence'],
                ml_volatility=ml_preds['volatility'],
                sentiment_score=sentiment
            )
            
            # Log current state
            state = self.grid_strategy.get_state()
            self.logger.logger.info(
                f"[{self.pair}] Grid State - Direction: {state['current_direction']}, "
                f"Positions: {state['num_open_positions']}, Price: ${current_price:.2f}, "
                f"Gap: {state['price_gap']:.2f}%, Spacing: {state['current_grid_spacing']:.2f}%"
            )
            
            # Log action
            self.logger.logger.info(
                f"[{self.pair}] Grid Action - {action['action']}: {action.get('reason', '')}"
            )
            
            # Execute trade
            if action['action'] != 'HOLD':
                self.execute_trade(action)
            
            return True
        
        except Exception as e:
            self.logger.logger.error(f"Error in run_cycle: {e}")
            import traceback
            traceback.print_exc()
            return False

    def start(self, interval: int = 60):
        """Start the trading bot loop"""
        self.logger.logger.info(f"Starting grid trading bot for {self.pair}, interval={interval}s")
        
        try:
            while self.running:
                try:
                    self.run_cycle()
                    time.sleep(interval)
                except KeyboardInterrupt:
                    self.logger.logger.info("Interrupted by user")
                    break
                except Exception as e:
                    self.logger.logger.error(f"Cycle error: {e}")
                    time.sleep(interval)
        
        finally:
            self.logger.logger.info("Grid trading bot stopped")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Grid Trading Bot')
    parser.add_argument('--pair', default='BTC/USD', help='Trading pair')
    parser.add_argument('--interval', type=int, default=60, help='Trading interval in seconds')
    parser.add_argument('--enable-trading', action='store_true', help='Enable real trading')
    parser.add_argument('--unit-amount', type=float, default=0.01, help='Unit lot size')
    parser.add_argument('--stop-loss', type=float, default=1000.0, help='Stop loss amount')
    parser.add_argument('--stop-profit', type=float, default=300.0, help='Take profit amount')
    
    args = parser.parse_args()
    
    # Create bot
    bot = GridTradingBot(pair=args.pair, enable_trading=args.enable_trading)
    
    # Update config if provided
    bot.update_grid_config(
        unit_amount=args.unit_amount,
        stop_loss=args.stop_loss,
        stop_profit=args.stop_profit,
    )
    
    # Run
    bot.start(interval=args.interval)


if __name__ == '__main__':
    main()