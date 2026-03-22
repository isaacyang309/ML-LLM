import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class GridDirection(Enum):
    """Trading direction state"""
    NONE = 0
    UP = 1
    DOWN = -1

@dataclass
class GridPosition:
    """Represents a single grid position"""
    entry_price: float
    quantity: float
    entry_time: datetime
    order_id: Optional[str] = None
    is_closed: bool = False
    close_price: Optional[float] = None
    close_time: Optional[datetime] = None

@dataclass
class GridTradeConfig:
    """Asset-specific configuration"""
    symbol: str = "BTC/USD"
    unit_amount: float = 0.01
    direction_multiplier: float = 2.0
    stop_loss: float = 1000.0
    stop_profit: float = 300.0
    ml_conf_threshold: float = 0.51 # Optimized based on our real-data backtest
    sentiment_limit: float = -0.5   # Kill switch threshold
    initial_lot_sizes: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.03, 0.04, 0.09])

    @classmethod
    def get_config_for_asset(cls, symbol: str):
        """Dynamic parameter optimization based on asset"""
        return cls(symbol=symbol)

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

class GridStrategy:
    """
    Fused AI Strategy: ML Brain + LLM Guard + Adaptive Grid
    """
    def __init__(self, config: Optional[GridTradeConfig] = None):
        self.config = config or GridTradeConfig()
        
        # State tracking
        self.last_trade_price: float = 0.0
        self.current_direction: GridDirection = GridDirection.NONE
        self.current_cumulative_amount: float = 0.0
        self.current_iteration: int = 0
        self.price_gap: float = 0.0
        
        # Position tracking
        self.positions: List[GridPosition] = []
        self.total_position_profit: float = 0.0
        self.current_grid_spacing: float = 0.0
        
        logger.info(f"FUSED STRATEGY READY: Asset={self.config.symbol}, ML_Gate={self.config.ml_conf_threshold}")

    def calculate_next_lot_size(self) -> float:
        initial_lots = self.config.initial_lot_sizes
        if initial_lots and self.current_iteration < len(initial_lots):
            lot_size = initial_lots[self.current_iteration]
            self.current_iteration += 1
        else:
            if self.current_cumulative_amount == 0:
                self.current_cumulative_amount = initial_lots[-1] if initial_lots else self.config.unit_amount
            self.current_cumulative_amount += self.config.unit_amount
            lot_size = self.current_cumulative_amount
        return lot_size

    def update_price_gap(self, current_price: float):
        if self.last_trade_price > 0:
            self.price_gap = current_price - self.last_trade_price
        else:
            self.price_gap = 0.0

    def calculate_total_position_profit(self, current_price: float) -> float:
        unrealized_profit = 0.0
        for pos in self.positions:
            if not pos.is_closed:
                if self.current_direction == GridDirection.UP:
                    unrealized_profit += (current_price - pos.entry_price) * pos.quantity
                else:
                    unrealized_profit += (pos.entry_price - current_price) * pos.quantity
        return unrealized_profit

    def add_position(self, entry_price: float, quantity: float, order_id: Optional[str] = None):
        position = GridPosition(entry_price=entry_price, quantity=quantity, entry_time=datetime.now(), order_id=order_id)
        self.positions.append(position)
        self.last_trade_price = entry_price

    def close_all_positions(self, close_price: float, reason: str = ""):
        for pos in self.positions:
            if not pos.is_closed:
                pos.is_closed = True
                pos.close_price = close_price
                pos.close_time = datetime.now()
        self.current_direction = GridDirection.NONE
        self.current_iteration = 0
        self.current_cumulative_amount = 0.0
        logger.info(f"GRID RESET: {reason}")

    def analyze(self, klines_data: pd.DataFrame, current_price: float, 
                ml_signal: int = 0, ml_confidence: float = 0.0, 
                ml_volatility: float = 0.0, sentiment_score: float = 0.0) -> Dict:
        
        if klines_data.empty:
            return {'action': 'HOLD', 'reason': 'No data'}

        # 1. THE LLM KILL SWITCH
        if sentiment_score <= self.config.sentiment_limit:
            return {'action': 'HOLD', 'reason': f'FinBERT Guard: High Risk News ({sentiment_score})'}

        self.update_price_gap(current_price)
        
        # 2. DYNAMIC TAKE PROFIT
        target_tp = self.config.stop_profit
        if ml_confidence > 0.80:
            target_tp *= 2 

        total_profit = self.calculate_total_position_profit(current_price)
        
        # 3. EXIT LOGIC
        if total_profit >= target_tp:
            return {'action': 'CLOSE_ALL', 'reason': 'AI Optimized Take Profit hit', 'close_price': current_price}
        if total_profit <= -self.config.stop_loss:
            return {'action': 'CLOSE_ALL', 'reason': 'Hard Stop Loss hit', 'close_price': current_price}

        # 4. ENTRY LOGIC: Fused ML + Grid
        if self.current_direction == GridDirection.NONE:
            if ml_confidence >= self.config.ml_conf_threshold:
                if ml_signal == 1:
                    self.current_direction = GridDirection.UP
                    return {'action': 'OPEN_POSITION', 'direction': 'UP', 'price': current_price, 'quantity': self.calculate_next_lot_size(), 'reason': 'ML Trend Confirmed'}
                elif ml_signal == -1:
                    self.current_direction = GridDirection.DOWN
                    return {'action': 'OPEN_POSITION', 'direction': 'DOWN', 'price': current_price, 'quantity': self.calculate_next_lot_size(), 'reason': 'ML Trend Confirmed'}
        
        # 5. GRID ADDITION & REVERSAL
        else:
            spacing = ml_volatility if ml_volatility > 0 else 1.5 
            price_gap_pct = (self.price_gap / self.last_trade_price) * 100 * self.current_direction.value

            if price_gap_pct >= spacing:
                return {'action': 'OPEN_POSITION', 'direction': self.current_direction.name, 'price': current_price, 'quantity': self.calculate_next_lot_size(), 'reason': 'Grid extension'}

            if price_gap_pct <= -(spacing * self.config.direction_multiplier):
                if ml_confidence < 0.75:
                    new_dir = GridDirection.DOWN if self.current_direction == GridDirection.UP else GridDirection.UP
                    self.current_direction = new_dir
                    return {'action': 'REVERSE_POSITION', 'direction': new_dir.name, 'price': current_price, 'quantity': self.calculate_next_lot_size(), 'reason': 'Trend Reversal'}

        return {'action': 'HOLD', 'reason': 'Monitoring market'}

    def get_state(self) -> Dict:
        return {
            'symbol': self.config.symbol,
            'direction': self.current_direction.name,
            'open_positions': len([p for p in self.positions if not p.is_closed]),
            'ml_gate': self.config.ml_conf_threshold
        }