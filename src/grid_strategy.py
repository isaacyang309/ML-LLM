import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
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
    """Configuration for grid trading strategy"""
    unit_amount: float = 0.01  # Base lot size for each grid
    direction_multiplier: float = 2.0  # Multiplier for reverse signal threshold
    stop_loss: float = 1000.0  # Stop loss in currency units
    stop_profit: float = 300.0  # Take profit in currency units
    lookback_hours: float = 24.0  # Hours to look back for price range
    drawdown_threshold: float = 1000.0  # Price range threshold in 1%
    price_climb_threshold: bool = True  # Whether to use threshold-based spacing
    price_climb_high_volatility: float = 100.0  # Grid spacing when volatility > threshold (in 1%)
    price_climb_low_volatility: float = 200.0  # Grid spacing when volatility < threshold (in 1%)
    initial_lot_sizes: List[float] = field(default_factory=list)  # e.g., [0.01, 0.02, 0.03, 0.04, 0.09]
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            'unit_amount': self.unit_amount,
            'direction_multiplier': self.direction_multiplier,
            'stop_loss': self.stop_loss,
            'stop_profit': self.stop_profit,
            'lookback_hours': self.lookback_hours,
            'drawdown_threshold': self.drawdown_threshold,
            'price_climb_threshold': self.price_climb_threshold,
            'price_climb_high_volatility': self.price_climb_high_volatility,
            'price_climb_low_volatility': self.price_climb_low_volatility,
            'initial_lot_sizes': self.initial_lot_sizes,
        }


class GridStrategy:
    """
    Adaptive Grid Trading Strategy
    
    This strategy implements a grid trading system that:
    - Adapts grid spacing based on recent price volatility
    - Opens positions in a trending direction at regular price intervals
    - Reverses direction when price moves against the position significantly
    - Uses initial lot sequence, then switches to constant increments
    - Exits all positions on take profit or stop loss
    """
    
    def __init__(self, config: Optional[GridTradeConfig] = None):
        """
        Initialize grid strategy
        
        Args:
            config: GridTradeConfig instance with all parameters
        """
        self.config = config or GridTradeConfig()
        
        # State tracking
        self.last_trade_price: float = 0.0
        self.current_direction: GridDirection = GridDirection.NONE
        self.current_cumulative_amount: float = 0.0
        self.current_iteration: int = 0  # Tracks position in initial_lot_sizes array
        self.price_gap: float = 0.0  # Current price - last trade price
        
        # Position tracking
        self.positions: List[GridPosition] = []
        self.total_position_profit: float = 0.0
        
        # State for logging/debugging
        self.is_in_drawdown: bool = False
        self.current_grid_spacing: float = 0.0
        
        logger.info("Grid Strategy initialized with config: %s", self.config.to_dict())
    
    def parse_initial_lots(self, lot_string: str) -> List[float]:
        """
        Parse initial lot sizes from string format
        
        Args:
            lot_string: Format like "0.01,0.02,0.03,0.04,0.09"
            
        Returns:
            List of lot sizes
        """
        if not lot_string or not lot_string.strip():
            return []
        
        try:
            lots = [float(x.strip()) for x in lot_string.split(',')]
            self.config.initial_lot_sizes = lots
            logger.info("Parsed initial lot sizes: %s", lots)
            return lots
        except ValueError as e:
            logger.error("Failed to parse lot string '%s': %s", lot_string, e)
            return []
    
    def get_price_range(self, klines_data: pd.DataFrame) -> float:
        """
        Calculate price range from historical data as percentage
        
        Args:
            klines_data: DataFrame with OHLCV data
            
        Returns:
            Price range as percentage (high - low) / low * 100
        """
        if klines_data.empty or len(klines_data) < 2:
            return 0.0
        
        high = klines_data['high'].max()
        low = klines_data['low'].min()
        
        if low == 0:
            return 0.0
        
        price_range = ((high - low) / low) * 100
        return price_range
    
    def is_high_volatility(self, price_range: float) -> bool:
        """
        Check if current volatility is high (range > threshold)
        
        Args:
            price_range: Price range as percentage
            
        Returns:
            True if high volatility, False if low volatility
        """
        return price_range >= self.config.drawdown_threshold
    
    def get_grid_spacing(self, price_range: float) -> float:
        """
        Get grid spacing based on volatility state
        
        Args:
            price_range: Price range as percentage
            
        Returns:
            Grid spacing in percentage
        """
        if self.is_high_volatility(price_range):
            spacing = self.config.price_climb_high_volatility
        else:
            spacing = self.config.price_climb_low_volatility
        
        self.current_grid_spacing = spacing
        self.is_in_drawdown = self.is_high_volatility(price_range)
        
        return spacing
    
    def calculate_next_lot_size(self) -> float:
        """
        Calculate the lot size for the next trade
        
        Returns:
            Lot size to use for next order
        """
        initial_lots = self.config.initial_lot_sizes
        
        # Use initial lot sequence if available and not exhausted
        if initial_lots and self.current_iteration < len(initial_lots):
            lot_size = initial_lots[self.current_iteration]
            self.current_iteration += 1
        else:
            # Switch to constant increment after initial sequence
            if self.current_cumulative_amount > 0:
                self.current_cumulative_amount += self.config.unit_amount
            else:
                # Initialize with last of initial lots or unit amount
                if initial_lots:
                    self.current_cumulative_amount = initial_lots[-1] + self.config.unit_amount
                else:
                    self.current_cumulative_amount = self.config.unit_amount
            
            lot_size = self.current_cumulative_amount
        
        return lot_size
    
    def update_price_gap(self, current_price: float):
        """
        Update price gap from last trade price
        
        Args:
            current_price: Current market price
        """
        if self.last_trade_price > 0:
            self.price_gap = current_price - self.last_trade_price
        else:
            self.price_gap = 0.0
    
    def check_stop_loss_profit(self) -> Tuple[bool, str]:
        """
        Check if stop loss or take profit should be triggered
        
        Returns:
            Tuple of (should_close, reason)
        """
        total_profit = self.calculate_total_position_profit()
        
        if total_profit >= self.config.stop_profit:
            return True, f"Take Profit hit: {total_profit:.2f} >= {self.config.stop_profit}"
        
        if total_profit <= -self.config.stop_loss:
            return True, f"Stop Loss hit: {total_profit:.2f} <= -{self.config.stop_loss}"
        
        return False, ""
    
    def calculate_total_position_profit(self) -> float:
        """
        Calculate total unrealized profit of all open positions
        
        Returns:
            Total profit in currency
        """
        total_profit = 0.0
        for position in self.positions:
            if not position.is_closed:
                # Note: This would require current price from caller
                # For now, we'll track realized profit from closed positions
                pass
        
        return total_profit
    
    def add_position(self, entry_price: float, quantity: float, order_id: Optional[str] = None):
        """
        Add a new position to the grid
        
        Args:
            entry_price: Entry price
            quantity: Position size
            order_id: Exchange order ID if available
        """
        position = GridPosition(
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.now(),
            order_id=order_id
        )
        self.positions.append(position)
        logger.info(
            "Added position: price=%.2f, qty=%.8f, direction=%s",
            entry_price, quantity, self.current_direction.name
        )
    
    def close_all_positions(self, close_price: float, reason: str = ""):
        """
        Close all open positions at given price
        
        Args:
            close_price: Price at which positions close
            reason: Reason for closing
        """
        closed_count = 0
        for position in self.positions:
            if not position.is_closed:
                position.is_closed = True
                position.close_price = close_price
                position.close_time = datetime.now()
                
                # Calculate profit
                if self.current_direction == GridDirection.UP:
                    profit = (close_price - position.entry_price) * position.quantity
                else:
                    profit = (position.entry_price - close_price) * position.quantity
                
                self.total_position_profit += profit
                closed_count += 1
        
        logger.info(
            "Closed %d positions at %.2f. Reason: %s. Total profit: %.2f",
            closed_count, close_price, reason, self.total_position_profit
        )
        
        # Reset state
        self.current_direction = GridDirection.NONE
        self.current_cumulative_amount = 0.0
        self.current_iteration = 0
    
    def analyze(self, klines_data: pd.DataFrame, current_price: float, 
                    ml_signal: int = 0, ml_confidence: float = 0.0, 
                    ml_volatility: float = 0.0, sentiment_score: float = 0.0) -> Dict:
            """
            Upgraded Analysis function driven by XGBoost and FinBERT.
            ml_signal: 1 (UP), -1 (DOWN), 0 (NEUTRAL)
            sentiment_score: 1.0 (Bullish), -1.0 (Bearish), 0.0 (Neutral)
            """
            if klines_data.empty or len(klines_data) < 2:
                return {'action': 'HOLD', 'reason': 'Insufficient data', 'direction': self.current_direction.name}
                
            # 1. THE LLM KILL SWITCH (FinBERT)
            if sentiment_score == -1.0:
                return {
                    'action': 'HOLD',
                    'reason': 'FinBERT Kill Switch: Strongly Bearish News Detected. Pausing entries.',
                    'direction': self.current_direction.name
                }

            if self.last_trade_price == 0:
                self.last_trade_price = current_price
                
            self.update_price_gap(current_price)
            
            # 2. DYNAMIC GRID SPACING (Driven by ML Volatility)
            # Assuming ML volatility is output as a percentage (e.g., 1.5 for 1.5%)
            grid_spacing = ml_volatility if ml_volatility > 0 else self.get_grid_spacing(self.get_price_range(klines_data))
            self.current_grid_spacing = grid_spacing

            # 3. DYNAMIC TAKE PROFIT (Driven by ML Confidence)
            current_take_profit = self.config.stop_profit
            if ml_confidence >= 0.8:
                current_take_profit = self.config.stop_profit * 2 # Double TP to 4% if highly confident

            # Check Stop Loss / Take Profit
            total_profit = self.calculate_total_position_profit()
            if total_profit >= current_take_profit:
                return {'action': 'CLOSE_ALL', 'reason': f'Dynamic Take Profit hit (${total_profit:.2f})', 'close_price': current_price, 'direction': self.current_direction.name}
            if total_profit <= -self.config.stop_loss: # Flat $1000 Stop Loss
                return {'action': 'CLOSE_ALL', 'reason': f'Stop Loss hit (${total_profit:.2f})', 'close_price': current_price, 'direction': self.current_direction.name}

            # 4. ML DIRECTIONAL FILTER
            if self.current_direction == GridDirection.NONE:
                # ONLY open a position if the ML agrees with the grid's natural price gap detection
                if abs(self.price_gap) * 100 >= grid_spacing:
                    grid_wants_up = self.price_gap > 0
                    
                    if grid_wants_up and ml_signal == 1:
                        self.current_direction = GridDirection.UP
                        return {'action': 'OPEN_POSITION', 'direction': 'UP', 'price': current_price, 'quantity': self.calculate_next_lot_size(), 'reason': 'ML Confirmed UP Trend'}
                    elif not grid_wants_up and ml_signal == -1:
                        self.current_direction = GridDirection.DOWN
                        return {'action': 'OPEN_POSITION', 'direction': 'DOWN', 'price': current_price, 'quantity': self.calculate_next_lot_size(), 'reason': 'ML Confirmed DOWN Trend'}

            else:
                # Already in a position. Check trend following.
                price_gap_pct = self.price_gap * self.current_direction.value * 100
                
                if price_gap_pct >= grid_spacing:
                    return {'action': 'OPEN_POSITION', 'direction': self.current_direction.name, 'price': current_price, 'quantity': self.calculate_next_lot_size(), 'reason': 'Trend following'}
                
                # 5. DYNAMIC REVERSALS (Diamond Hands on High Confidence)
                reverse_threshold = -(grid_spacing * self.config.direction_multiplier)
                if price_gap_pct <= reverse_threshold:
                    if ml_confidence >= 0.8:
                        return {'action': 'HOLD', 'reason': 'High ML Confidence overriding reversal. Holding position through dip.', 'direction': self.current_direction.name}
                    else:
                        # Normal reversal
                        self.current_direction = GridDirection.DOWN if self.current_direction == GridDirection.UP else GridDirection.UP
                        return {'action': 'REVERSE_POSITION', 'direction': self.current_direction.name, 'price': current_price, 'quantity': self.calculate_next_lot_size(), 'reason': 'Reversal signal triggered'}

            return {'action': 'HOLD', 'reason': 'Waiting for signal', 'direction': self.current_direction.name}
    
    def get_state(self) -> Dict:
        """
        Get current strategy state for logging/debugging
        
        Returns:
            Dict with strategy state
        """
        return {
            'last_trade_price': self.last_trade_price,
            'current_direction': self.current_direction.name,
            'current_cumulative_amount': self.current_cumulative_amount,
            'current_iteration': self.current_iteration,
            'price_gap': self.price_gap,
            'num_open_positions': sum(1 for p in self.positions if not p.is_closed),
            'total_position_profit': self.total_position_profit,
            'is_in_drawdown': self.is_in_drawdown,
            'current_grid_spacing': self.current_grid_spacing,
        }
    
    def reset(self):
        """Reset strategy state"""
        self.last_trade_price = 0.0
        self.current_direction = GridDirection.NONE
        self.current_cumulative_amount = 0.0
        self.current_iteration = 0
        self.price_gap = 0.0
        self.positions = []
        self.total_position_profit = 0.0
        logger.info("Grid strategy reset")
