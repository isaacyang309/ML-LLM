#!/usr/bin/env python3
"""
Interactive Binance API Testing Script

This script allows you to test HorusClient methods backed by Binance public endpoints.
It helps verify connectivity and inspect normalized responses.

Usage:
    python tests/test_binance_api.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.horus_client import HorusClient


class BinanceAPITester:
    def __init__(self):
        self.client = HorusClient()

    def print_separator(self):
        print("\n" + "=" * 80 + "\n")

    def print_json(self, data, title="Response"):
        """Pretty print JSON-serializable data."""
        print(f"\n{title}:")
        print("-" * 40)
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        print("-" * 40)

    def test_current_price(self, symbol="BTC"):
        """Test 1: Get current ticker price."""
        self.print_separator()
        print("TEST 1: GET CURRENT PRICE")
        print("Endpoint: /api/v3/ticker/price")
        print(f"Symbol: {symbol}")

        try:
            price = self.client.get_current_price(symbol)
            if price > 0:
                print("\nPrice Result:")
                print(f"  {symbol} price: ${price:,.2f}")
            else:
                print("\nRequest returned no valid price (0.0).")
        except Exception as e:
            print(f"Error: {e}")

    def test_klines(self, symbol="BTC", interval="15m", limit=20):
        """Test 2: Get candlestick data."""
        self.print_separator()
        print("TEST 2: GET KLINES")
        print("Endpoint: /api/v3/klines")
        print(f"Symbol: {symbol}")
        print(f"Interval: {interval}")
        print(f"Limit: {limit}")

        try:
            df = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)

            if df.empty:
                print("\nNo kline data returned.")
                return

            print(f"\nRetrieved {len(df)} candles")
            print("\nHead (first 5 rows):")
            print(df.head().to_string(index=False))

            summary = {
                "rows": int(len(df)),
                "start_timestamp": df.iloc[0]["timestamp"],
                "end_timestamp": df.iloc[-1]["timestamp"],
                "open_min": float(df["open"].min()),
                "high_max": float(df["high"].max()),
                "low_min": float(df["low"].min()),
                "close_last": float(df.iloc[-1]["close"]),
                "volume_sum": float(df["volume"].sum()),
            }
            self.print_json(summary, "Kline Summary")
        except Exception as e:
            print(f"Error: {e}")

    def test_price_history(self, symbol="BTC", interval="15m", limit=20):
        """Test 3: Get legacy-compatible price history."""
        self.print_separator()
        print("TEST 3: GET PRICE HISTORY")
        print("Method: HorusClient.get_price_history (compatibility format)")
        print(f"Symbol: {symbol}")
        print(f"Interval: {interval}")
        print(f"Limit: {limit}")

        try:
            history = self.client.get_price_history(symbol=symbol, interval=interval, limit=limit)

            if not history:
                print("\nNo history records returned.")
                return

            print(f"\nRetrieved {len(history)} history records")
            self.print_json(history[:3], "First 3 Records")

            prices = [item.get("price", 0) for item in history]
            print("\nPrice Stats:")
            print(f"  min: {min(prices):,.4f}")
            print(f"  max: {max(prices):,.4f}")
            print(f"  last: {prices[-1]:,.4f}")
        except Exception as e:
            print(f"Error: {e}")

    def test_interval_parser(self):
        """Test 4: Verify interval parsing helper."""
        self.print_separator()
        print("TEST 4: PARSE INTERVAL TO MINUTES")

        intervals = ["1m", "5m", "15m", "1h", "4h", "1d", "1w", "bad"]
        try:
            parsed = {item: self.client._parse_interval_to_minutes(item) for item in intervals}
            self.print_json(parsed, "Parsed Intervals")
        except Exception as e:
            print(f"Error: {e}")

    def test_all_readonly(self):
        """Run all safe tests."""
        print("\n" + "BINANCE API TESTING SUITE" + "\n")
        print("Running all read-only tests")
        print("=" * 80)

        self.test_current_price("BTC")
        self.test_klines(symbol="BTC", interval="15m", limit=20)
        self.test_price_history(symbol="BTC", interval="15m", limit=20)
        self.test_interval_parser()

        self.print_separator()
        print("Read-only tests completed")

    def run_interactive(self):
        """Interactive menu for testing methods."""
        while True:
            self.print_separator()
            print("BINANCE API TESTING MENU")
            print("\nSelect a test to run:")
            print("  1. Get Current Price - SAFE")
            print("  2. Get Klines - SAFE")
            print("  3. Get Price History - SAFE")
            print("  4. Parse Interval Helper - SAFE")
            print("  5. Run All Read-Only Tests - SAFE")
            print("  0. Exit")

            choice = input("\nEnter your choice (0-5): ").strip()

            if choice == "1":
                symbol = input("Symbol [BTC]: ").strip().upper() or "BTC"
                self.test_current_price(symbol)
            elif choice == "2":
                symbol = input("Symbol [BTC]: ").strip().upper() or "BTC"
                interval = input("Interval [15m]: ").strip() or "15m"
                limit_raw = input("Limit [20]: ").strip() or "20"
                limit = int(limit_raw)
                self.test_klines(symbol=symbol, interval=interval, limit=limit)
            elif choice == "3":
                symbol = input("Symbol [BTC]: ").strip().upper() or "BTC"
                interval = input("Interval [15m]: ").strip() or "15m"
                limit_raw = input("Limit [20]: ").strip() or "20"
                limit = int(limit_raw)
                self.test_price_history(symbol=symbol, interval=interval, limit=limit)
            elif choice == "4":
                self.test_interval_parser()
            elif choice == "5":
                self.test_all_readonly()
            elif choice == "0":
                print("\nGoodbye!")
                break
            else:
                print("\nInvalid choice, please try again.")

            input("\nPress Enter to continue...")


def main():
    """Main entry point."""
    print("=" * 80)
    print("BINANCE API TESTER")
    print("=" * 80)
    print("\nThis tool tests HorusClient methods using Binance public endpoints.")
    print("No authenticated account actions are performed.")
    print("=" * 80)

    tester = BinanceAPITester()

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in {"all", "readonly"}:
            tester.test_all_readonly()
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python tests/test_binance_api.py [all|readonly]")
    else:
        tester.run_interactive()


if __name__ == "__main__":
    main()
