# Grid Trading Strategy Implementation

## Overview

This is a Python implementation of an adaptive grid trading strategy, ported from the MQL5 expert advisor (`EA.mq5`). The strategy is designed for cryptocurrency trading and adapts its grid spacing based on recent market volatility.

## Strategy Logic

### Core Concepts

1. **Adaptive Grid Spacing**: The strategy dynamically adjusts the grid spacing (distance between orders) based on recent price volatility:
   - **High Volatility** (price range > threshold): Tighter grid spacing for more frequent trades
   - **Low Volatility** (price range < threshold): Wider grid spacing to reduce noise

2. **Direction Detection**: 
   - Starts trading in a direction based on the first significant price move
   - Continues adding positions in that direction at regular price intervals
   - Reverses direction when price moves against the position by a larger multiplier

3. **Position Sizing**:
   - Uses an initial sequence of lot sizes (e.g., 0.01, 0.02, 0.03, 0.04, 0.09 BTC)
   - After exhausting the initial sequence, switches to constant increments
   - This creates an accelerating position size profile

4. **Exit Conditions**:
   - **Take Profit**: Closes all positions when total profit >= configured threshold
   - **Stop Loss**: Closes all positions when total loss >= configured threshold

## Files

### `src/grid_strategy.py`
Core strategy implementation with:
- `GridTradeConfig`: Configuration dataclass
- `GridStrategy`: Main strategy class
- `GridDirection`: Direction enumeration
- `GridPosition`: Position tracking

### `src/grid_trading_bot.py`
Integration layer that:
- Fetches market data from Binance via `HorusClient`
- Executes trades via `RoostooClient`
- Manages trading loop and logging

## Configuration

### GridTradeConfig Parameters

```python
GridTradeConfig(
    unit_amount=0.01,                    # Base lot size for each grid level
    direction_multiplier=2.0,             # Reverse signal threshold multiplier
    stop_loss=1000.0,                    # Stop loss in currency units
    stop_profit=300.0,                   # Take profit in currency units
    lookback_hours=24.0,                 # Hours to look back for volatility calculation
    drawdown_threshold=1000.0,           # Price range threshold in 1% (e.g., 1000 = 10%)
    price_climb_high_volatility=100.0,   # Grid spacing when above threshold (in 1%)
    price_climb_low_volatility=200.0,    # Grid spacing when below threshold (in 1%)
    initial_lot_sizes=[0.01, 0.02, 0.03, 0.04, 0.09],  # Initial position sizes
)
```

## Usage

### Basic Usage

```python
from src.grid_trading_bot import GridTradingBot

# Create bot (simulation mode)
bot = GridTradingBot(pair="BTC/USD", enable_trading=False)

# Run one cycle
success = bot.run_cycle()

# Run continuous trading loop
bot.start(interval=60)  # 60-second intervals
```

### Command Line

```bash
# Simulation mode
python src/grid_trading_bot.py --pair BTC/USD --interval 60

# Real trading (use with caution!)
python src/grid_trading_bot.py --pair BTC/USD --enable-trading --stop-loss 500 --stop-profit 200

# Custom configuration
python src/grid_trading_bot.py \
    --pair ETH/USD \
    --interval 30 \
    --unit-amount 0.1 \
    --stop-loss 1000 \
    --stop-profit 500
```

### Programmatic Configuration

```python
from src.grid_trading_bot import GridTradingBot

bot = GridTradingBot(pair="BTC/USD", enable_trading=False)

# Update configuration
bot.update_grid_config(
    unit_amount=0.01,
    stop_loss=1000.0,
    stop_profit=300.0,
    direction_multiplier=2.0,
    drawdown_threshold=1000.0,
    price_climb_high_volatility=100.0,
    price_climb_low_volatility=200.0,
)

# Set initial lot sizes
bot.set_initial_lots("0.01,0.02,0.03,0.04,0.09")

# Run
bot.start(interval=60)
```

## Strategy Workflow

```
1. Fetch Market Data
   ├─ Get current price from Binance
   └─ Get 100 15-minute candles

2. Calculate Volatility
   ├─ Find high and low prices over lookback period
   ├─ Calculate price range as percentage
   └─ Determine if market is in high or low volatility

3. Get Grid Spacing
   ├─ If high volatility: use price_climb_high_volatility
   └─ If low volatility: use price_climb_low_volatility

4. Check Exit Conditions
   ├─ Calculate total position profit
   ├─ If profit >= stop_profit: CLOSE_ALL
   └─ If loss >= stop_loss: CLOSE_ALL

5. Trading Logic
   ├─ If no position:
   │  └─ Wait for first significant move >= grid_spacing
   ├─ If trending:
   │  ├─ Add position if price moved in direction >= grid_spacing
   │  └─ Reverse if price moved against direction >= (grid_spacing * direction_multiplier)
   └─ Else: HOLD

6. Execute Trade
   ├─ Open position via Roostoo
   ├─ Track in GridStrategy
   └─ Log signal
```

## State Management

The strategy maintains state between iterations:

```python
strategy_state = strategy.get_state()
# Returns:
{
    'last_trade_price': 70000.0,           # Price of last trade
    'current_direction': 'UP',             # Current trading direction (UP/DOWN/NONE)
    'current_cumulative_amount': 0.15,     # Current lot size in progressive phase
    'current_iteration': 5,                # Position in initial_lot_sizes array
    'price_gap': 0.5,                      # Current price - last trade price
    'num_open_positions': 3,               # Number of open grid positions
    'total_position_profit': 50.0,         # Unrealized profit from all positions
    'is_in_drawdown': False,               # Whether in high volatility state
    'current_grid_spacing': 200.0,         # Current grid spacing in 1%
}
```

## Integration with Existing Bot

The grid strategy integrates seamlessly with the existing trading infrastructure:

- **Market Data**: Uses the new Binance API via `HorusClient`
- **Order Execution**: Uses existing `RoostooClient` for trades
- **Logging**: Uses existing `TradingLogger` for signals and state
- **Configuration**: Follows existing config structure

## Examples

### Example 1: Tight Grid on High Volatility

```python
from src.grid_trading_bot import GridTradingBot

bot = GridTradingBot(pair="BTC/USD", enable_trading=False)
bot.update_grid_config(
    drawdown_threshold=500.0,      # 5% volatility threshold
    price_climb_high_volatility=50.0,    # 0.5% grid when volatile
    price_climb_low_volatility=150.0,    # 1.5% grid when quiet
)
bot.start()
```

### Example 2: Aggressive Position Sizing

```python
bot = GridTradingBot(pair="ETH/USD", enable_trading=False)
bot.update_grid_config(
    unit_amount=0.1,
    initial_lot_sizes=[0.1, 0.2, 0.3, 0.5, 1.0],
)
bot.start()
```

### Example 3: Conservative Risk Management

```python
bot = GridTradingBot(pair="BTC/USD", enable_trading=False)
bot.update_grid_config(
    stop_loss=500.0,       # Tight stop loss
    stop_profit=200.0,     # Take small profits
    direction_multiplier=3.0,  # Need bigger reversal to flip
)
bot.start()
```

## Comparison with Original MQL5

| Feature | MQL5 | Python |
|---------|------|--------|
| Price data | Horus API | Binance REST API |
| Order execution | MT5 API | Roostoo API |
| State persistence | In-memory | Dictionary |
| Logging | MT5 Print() | Python logging |
| Configurability | Input parameters | GridTradeConfig + CLI args |
| Backtesting | MT5 built-in | Can add pandas-based |

## Performance Considerations

- **API Calls**: Minimized through efficient Binance API usage
- **Position Tracking**: O(1) state updates, O(n) profit calculation
- **Memory**: ~1 KB per open position
- **Network**: 2 API calls per cycle (price + klines)

## Testing

```bash
# Run strategy tests
python -m pytest tests/test_grid_strategy.py

# Run integration tests
python -m pytest tests/test_grid_trading_bot.py

# Run full cycle test
python src/grid_trading_bot.py --pair BTC/USD
```

## Safety Notes

⚠️ **Important**: When using `enable_trading=True`:

1. Start in simulation mode (`enable_trading=False`) first
2. Test with paper trading on a small amount
3. Monitor logs carefully during first live trades
4. Set conservative stop_loss values
5. Use limit orders when possible
6. Never set position sizes larger than you can afford to lose

## Future Enhancements

- [ ] Backtesting engine with historical data
- [ ] Position profit calculation improvements
- [ ] Multi-pair trading support
- [ ] Dynamic parameter optimization
- [ ] Performance metrics and reporting
- [ ] Telegram/Discord notifications
- [ ] Web dashboard for monitoring

## Troubleshooting

### Strategy not opening positions
- Check grid_spacing calculation (depends on volatility)
- Verify price gap is larger than grid spacing
- Check stop_loss/stop_profit haven't been triggered

### Too many positions
- Reduce `price_climb_high_volatility`
- Increase `drawdown_threshold`
- Increase `initial_lot_sizes`

### Frequent reversals
- Increase `direction_multiplier` (need bigger move to reverse)
- Reduce noise in market data with larger candle intervals

## License

Same as the main trading bot project
