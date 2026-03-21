import requests
import pandas as pd
from typing import Dict, List, Optional
from config.config import Config

class HorusClient:
    """
    Client for fetching market data from Binance public REST API
    (Previously used Horus API, now using Binance public endpoints that don't require API keys)
    """
    
    def __init__(self):
        self.config = Config()
        self.base_url = self.config.BINANCE_BASE_URL
        self.api_version = self.config.BINANCE_API_VERSION
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Send request to Binance API"""
        url = f"{self.base_url}/api/{self.api_version}{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Log summarized response
            if isinstance(data, list):
                print(f"Binance API: Retrieved {len(data)} data points from {endpoint}")
            else:
                print(f"Binance API: Response from {endpoint} - Status: {response.status_code}")
            
            return data
        except requests.exceptions.RequestException as e:
            print(f"Binance API request error: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Error response: {e.response.text[:200]}")
            return {'error': str(e)}
    
    def _binance_symbol(self, symbol: str) -> str:
        """
        Convert symbol to Binance format.
        E.g., "BTC" -> "BTCUSDT" (or "BTCUSD" if available)
        """
        # For now, use USDT as the quote currency (most liquid on Binance)
        return f"{symbol}USDT"
    
    def get_current_price(self, symbol: str = "BTC") -> float:
        """
        Get current price for a symbol
        
        Args:
            symbol: Asset symbol (e.g., "BTC", "ETH")
            
        Returns:
            Current price as float, or 0 if error
        """
        binance_symbol = self._binance_symbol(symbol)
        
        try:
            data = self._make_request('/ticker/price', {'symbol': binance_symbol})
            
            if 'error' in data:
                print(f"Error fetching price for {symbol}: {data['error']}")
                return 0.0
            
            price = float(data.get('price', 0))
            print(f"Current price for {symbol}: ${price:.2f}")
            return price
        except (KeyError, ValueError) as e:
            print(f"Error parsing price for {symbol}: {e}")
            return 0.0
    
    def get_klines(self, symbol: str = "BTC", interval: str = "15m", limit: int = 100) -> pd.DataFrame:
        """
        Get klines (candlestick data) from Binance
        
        Args:
            symbol: Asset symbol (e.g., "BTC", "ETH")
            interval: Time interval ('1m', '5m', '15m', '1h', '1d', etc.)
            limit: Number of candles to retrieve (max 1000)
            
        Returns:
            DataFrame with columns: open, high, low, close, volume, timestamp
            Returns empty DataFrame if error occurs
        """
        binance_symbol = self._binance_symbol(symbol)
        
        # Validate limit
        limit = min(max(limit, 1), 1000)
        
        try:
            data = self._make_request(
                '/klines',
                {
                    'symbol': binance_symbol,
                    'interval': interval,
                    'limit': limit
                }
            )
            
            if 'error' in data or not data:
                print(f"Error fetching klines for {symbol}: {data.get('error', 'No data')}")
                return pd.DataFrame()
            
            # Parse Binance kline format
            # Binance returns: [time, open, high, low, close, volume, close_time, quote_asset_volume, ...]
            klines = []
            for kline in data:
                klines.append({
                    'timestamp': pd.Timestamp(int(kline[0]), unit='ms'),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })
            
            df = pd.DataFrame(klines)
            print(f"Retrieved {len(df)} klines for {symbol} ({interval})")
            return df
            
        except (IndexError, ValueError) as e:
            print(f"Error parsing klines for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_price_history(self, symbol: str = "BTC", interval: str = "15m", 
                         start: Optional[int] = None, end: Optional[int] = None,
                         limit: int = 100) -> List:
        """
        Get historical price data (legacy method for backward compatibility with strategy.py)
        
        Args:
            symbol: Asset symbol (e.g., "BTC", "ETH")
            interval: Time interval ("1d", "1h", "15m", etc.)
            start: Start timestamp in seconds (will be converted to milliseconds)
            end: End timestamp in seconds (will be converted to milliseconds)
            limit: Number of candles to retrieve (default 100, estimated if start/end provided)
            
        Returns:
            List of dicts with 'price' key (for compatibility with existing code)
        """
        # Note: Binance doesn't support timestamp-based queries like Horus did.
        # Instead, we fetch the last N candles. For 100 candles at 15m intervals,
        # that's ~25 hours of data.
        
        # Determine limit based on interval and time range
        # If start/end provided, estimate candles needed
        if start is not None and end is not None:
            # Convert from seconds to milliseconds if needed
            time_range_sec = (end - start)
            interval_minutes = self._parse_interval_to_minutes(interval)
            estimated_candles = time_range_sec // (interval_minutes * 60)
            limit = min(int(estimated_candles) + 10, 1000)  # Add buffer, cap at 1000
        
        df = self.get_klines(symbol=symbol, interval=interval, limit=limit)
        
        if df.empty:
            return []
        
        # Convert to list of dicts with 'price' key for backward compatibility
        result = []
        for _, row in df.iterrows():
            result.append({
                'price': row['close'],
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'volume': row['volume']
            })
        
        return result
    
    @staticmethod
    def _parse_interval_to_minutes(interval: str) -> int:
        """Convert interval string to minutes"""
        multipliers = {'m': 1, 'h': 60, 'd': 1440, 'w': 10080}
        
        for unit, multiplier in multipliers.items():
            if interval.endswith(unit):
                try:
                    value = int(interval[:-1])
                    return value * multiplier
                except ValueError:
                    return 15  # Default to 15m if parsing fails
        
        return 15  # Default fallback