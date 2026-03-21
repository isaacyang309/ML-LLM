"""
Tests for Grid Strategy implementation
"""

import sys
from pathlib import Path
import time

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from src.grid_strategy import GridStrategy, GridTradeConfig, GridDirection
from src.horus_client import HorusClient
from src.grid_trading_bot import GridTradingBot


def test_grid_config():
    """Test GridTradeConfig creation and serialization"""
    print("\n" + "=" * 60)
    print("Test 1: GridTradeConfig")
    print("=" * 60)
    
    config = GridTradeConfig(
        unit_amount=0.01,
        initial_lot_sizes=[0.01, 0.02, 0.03],
    )
    
    config_dict = config.to_dict()
    print(f"Config: {config_dict}")
    assert config.unit_amount == 0.01
    assert len(config.initial_lot_sizes) == 3
    print("PASSED")


def test_lot_size_progression():
    """Test that lot sizes progress correctly"""
    print("\n" + "=" * 60)
    print("Test 2: Lot Size Progression")
    print("=" * 60)
    
    config = GridTradeConfig(
        unit_amount=0.01,
        initial_lot_sizes=[0.01, 0.02, 0.03, 0.04, 0.09],
    )
    strategy = GridStrategy(config=config)
    
    # First 5 should be from initial_lot_sizes
    lots = []
    for i in range(7):
        lot = strategy.calculate_next_lot_size()
        lots.append(lot)
        print(f"Iteration {i}: lot={lot:.4f}, cumulative={strategy.current_cumulative_amount:.4f}")
    
    assert abs(lots[0] - 0.01) < 0.0001, f"Expected 0.01, got {lots[0]}"
    assert abs(lots[1] - 0.02) < 0.0001, f"Expected 0.02, got {lots[1]}"
    assert abs(lots[2] - 0.03) < 0.0001, f"Expected 0.03, got {lots[2]}"
    assert abs(lots[3] - 0.04) < 0.0001, f"Expected 0.04, got {lots[3]}"
    assert abs(lots[4] - 0.09) < 0.0001, f"Expected 0.09, got {lots[4]}"
    assert abs(lots[5] - 0.10) < 0.0001, f"Expected 0.10, got {lots[5]}"
    assert abs(lots[6] - 0.11) < 0.0001, f"Expected 0.11, got {lots[6]}"
    print("PASSED")


def test_volatility_calculation():
    """Test price range and volatility calculation"""
    print("\n" + "=" * 60)
    print("Test 3: Volatility Calculation")
    print("=" * 60)
    
    client = HorusClient()
    strategy = GridStrategy()
    
    # Fetch real market data
    klines_list = client.get_price_history("BTC", interval="15m", limit=100)
    if not klines_list:
        print("SKIPPED - Could not fetch market data")
        return
    
    klines_df = pd.DataFrame(klines_list)
    price_range = strategy.get_price_range(klines_df)
    
    print(f"High price: ${klines_df['high'].max():.2f}")
    print(f"Low price: ${klines_df['low'].min():.2f}")
    print(f"Price range: {price_range:.2f}%")
    
    is_high_vol = strategy.is_high_volatility(price_range)
    spacing = strategy.get_grid_spacing(price_range)
    
    print(f"Is high volatility: {is_high_vol}")
    print(f"Grid spacing: {spacing:.2f}%")
    
    # Verify spacing selection
    if price_range >= strategy.config.drawdown_threshold:
        assert spacing == strategy.config.price_climb_high_volatility
    else:
        assert spacing == strategy.config.price_climb_low_volatility
    
    print("PASSED")


def test_direction_detection():
    """Test that direction is correctly detected from first move"""
    print("\n" + "=" * 60)
    print("Test 4: Direction Detection")
    print("=" * 60)
    
    config = GridTradeConfig(
        drawdown_threshold=100.0,  # Make threshold small so we use high_vol spacing
        price_climb_high_volatility=50.0,  # 50% grid spacing (large for testing)
        price_climb_low_volatility=100.0,
    )
    strategy = GridStrategy(config=config)
    
    # Mock klines (high volatility to trigger high_vol spacing)
    klines_df = pd.DataFrame({
        'high': [150, 160, 170],
        'low': [90, 100, 110],
        'close': [120, 130, 140],
    })
    
    # First cycle: no direction yet, small price move
    strategy.last_trade_price = 100.0
    action = strategy.analyze(klines_df, 100.3)
    assert action['action'] == 'HOLD', f"Small move should be ignored, got {action['action']}"
    print(f"Small move (0.3%): {action['action']} - OK")
    
    # Second cycle: big upward move -> detect UP direction (need 50%+ move)
    strategy.reset()
    strategy.last_trade_price = 100.0
    action = strategy.analyze(klines_df, 151.0)  # 51% up
    assert action['action'] == 'OPEN_POSITION', f"Expected OPEN_POSITION, got {action['action']}"
    assert action['direction'] == 'UP'
    print(f"Big upward move (51%): {action['action']} direction={action['direction']} - OK")
    
    # Reset and try downward
    strategy.reset()
    strategy.last_trade_price = 100.0
    action = strategy.analyze(klines_df, 48.0)  # 52% down
    assert action['action'] == 'OPEN_POSITION'
    assert action['direction'] == 'DOWN'
    print(f"Downward move (-52%): {action['action']} direction={action['direction']} - OK")
    
    print("PASSED")


def test_reversal_logic():
    """Test that reversals are triggered correctly"""
    print("\n" + "=" * 60)
    print("Test 5: Reversal Logic")
    print("=" * 60)
    
    config = GridTradeConfig(
        direction_multiplier=2.0,
        drawdown_threshold=100.0,
        price_climb_high_volatility=50.0,  # 50% grid spacing
        price_climb_low_volatility=100.0,
    )
    strategy = GridStrategy(config=config)
    
    # Mock klines (high volatility)
    klines_df = pd.DataFrame({
        'high': [150, 160, 170],
        'low': [90, 100, 110],
        'close': [120, 130, 140],
    })
    
    # Open UP position
    strategy.last_trade_price = 100.0
    action = strategy.analyze(klines_df, 152.0)  # 52% up -> open
    strategy.add_position(152.0, 0.01)
    print(f"Opened UP position at 100.0 -> 152.0 (52% up)")
    
    # Large reversal - should reverse (need 50% * 2.0 = 100% down from 152)
    # 152 * 0.33 = 50 -> 67% down, which is > 100% reversal threshold
    strategy.last_trade_price = 152.0
    action = strategy.analyze(klines_df, 50.0)  
    assert action['action'] == 'REVERSE_POSITION', f"Expected REVERSE_POSITION, got {action['action']}"
    assert action['direction'] == 'DOWN'
    print(f"Big reversal (152.0 -> 50.0, -67%): REVERSE to DOWN - OK")
    
    print("PASSED")


def test_grid_bot_initialization():
    """Test GridTradingBot initialization"""
    print("\n" + "=" * 60)
    print("Test 6: GridTradingBot Initialization")
    print("=" * 60)
    
    bot = GridTradingBot(pair="BTC/USD", enable_trading=False)
    
    assert bot.pair == "BTC/USD"
    assert bot.base_currency == "BTC"
    assert bot.enable_trading == False
    assert bot.grid_strategy is not None
    
    print(f"Pair: {bot.pair}")
    print(f"Base currency: {bot.base_currency}")
    print(f"Trading enabled: {bot.enable_trading}")
    print(f"Grid strategy initialized: {bot.grid_strategy is not None}")
    print("PASSED")


def test_market_data_fetch():
    """Test fetching real market data"""
    print("\n" + "=" * 60)
    print("Test 7: Market Data Fetch")
    print("=" * 60)
    
    bot = GridTradingBot(pair="BTC/USD", enable_trading=False)
    market_data = bot.fetch_market_data()
    
    assert market_data is not None
    assert 'current_price' in market_data
    assert 'klines' in market_data
    assert 'timestamp' in market_data
    
    assert market_data['current_price'] > 0
    assert len(market_data['klines']) > 0
    
    print(f"Current price: ${market_data['current_price']:.2f}")
    print(f"Klines count: {len(market_data['klines'])}")
    print(f"Timestamp: {market_data['timestamp']}")
    print("PASSED")


def test_full_trading_cycle():
    """Test a full trading cycle"""
    print("\n" + "=" * 60)
    print("Test 8: Full Trading Cycle")
    print("=" * 60)
    
    bot = GridTradingBot(pair="BTC/USD", enable_trading=False)
    bot.set_initial_lots("0.01,0.02,0.03,0.04,0.09")
    
    # Run cycle
    success = bot.run_cycle()
    
    assert success == True
    
    state = bot.grid_strategy.get_state()
    print(f"Cycle completed successfully")
    print(f"Strategy state:")
    for key, value in state.items():
        print(f"  {key}: {value}")
    
    print("PASSED")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("GRID STRATEGY TEST SUITE")
    print("=" * 80)
    
    tests = [
        test_grid_config,
        test_lot_size_progression,
        test_volatility_calculation,
        test_direction_detection,
        test_reversal_logic,
        test_grid_bot_initialization,
        test_market_data_fetch,
        test_full_trading_cycle,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}")
            failed += 1
        except Exception as e:
            if "SKIPPED" in str(e):
                skipped += 1
            else:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 80 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
