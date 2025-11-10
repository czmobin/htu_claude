"""
CSV Data Loader for Backtesting
Load historical data from CSV files for unlimited backtesting
"""

import pandas as pd
import os
from typing import Dict, Optional
import logging

logger = logging.getLogger('ICTBot')


class CSVDataLoader:
    """
    Load historical forex data from CSV files
    Supports multiple timeframes
    """

    def __init__(self, data_folder: str = "historical_data"):
        """
        Initialize CSV data loader

        Args:
            data_folder: Folder containing CSV files
        """
        self.data_folder = data_folder
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            logger.info(f"Created data folder: {data_folder}")

    def load_csv(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Load CSV file for specific symbol and timeframe

        Expected CSV format:
        - Columns: time, open, high, low, close, volume
        - OR: Date, Time, Open, High, Low, Close, Volume (MetaTrader format)

        Args:
            symbol: Trading symbol (e.g., EURUSD)
            timeframe: Timeframe (M1, M5, M15, H1, H4, D)

        Returns:
            DataFrame with OHLCV data or None
        """
        # Try different filename formats
        possible_files = [
            f"{symbol}_{timeframe}.csv",
            f"{symbol.replace('/', '')}_{timeframe}.csv",
            f"{symbol.upper()}_{timeframe}.csv",
            f"{timeframe}_{symbol}.csv"
        ]

        csv_path = None
        for filename in possible_files:
            path = os.path.join(self.data_folder, filename)
            if os.path.exists(path):
                csv_path = path
                break

        if csv_path is None:
            logger.warning(f"CSV file not found for {symbol} {timeframe}")
            logger.info(f"Tried: {possible_files}")
            return None

        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV: {csv_path} ({len(df)} rows)")

            # Detect format and standardize
            df = self._standardize_format(df)

            if df is None or df.empty:
                logger.error(f"Failed to parse CSV: {csv_path}")
                return None

            logger.info(f"Successfully loaded {len(df)} candles for {symbol} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}", exc_info=True)
            return None

    def _standardize_format(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Standardize CSV format to match bot's expected format

        Handles:
        - MetaTrader format: Date, Time, Open, High, Low, Close, Volume
        - Standard format: time, open, high, low, close, volume
        - Datetime column: datetime, open, high, low, close, volume
        """
        try:
            # Case 1: MetaTrader format (Date + Time columns)
            if 'Date' in df.columns and 'Time' in df.columns:
                logger.debug("Detected MetaTrader format")
                df['time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                df = df.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })

            # Case 2: Single datetime column
            elif 'datetime' in df.columns or 'Datetime' in df.columns:
                logger.debug("Detected datetime column format")
                datetime_col = 'datetime' if 'datetime' in df.columns else 'Datetime'
                df['time'] = pd.to_datetime(df[datetime_col])
                if 'Open' in df.columns:
                    df = df.rename(columns={
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })

            # Case 3: Already in correct format (lowercase)
            elif 'time' in df.columns:
                logger.debug("Detected standard format")
                df['time'] = pd.to_datetime(df['time'])

            # Case 4: Timestamp as index
            elif df.index.name in ['Date', 'date', 'datetime', 'Datetime']:
                logger.debug("Detected timestamp index format")
                df = df.reset_index()
                df = df.rename(columns={df.columns[0]: 'time'})
                df['time'] = pd.to_datetime(df['time'])
                if 'Open' in df.columns:
                    df = df.rename(columns={
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })

            else:
                logger.error(f"Unknown CSV format. Columns: {df.columns.tolist()}")
                return None

            # Ensure all required columns exist
            required_cols = ['time', 'open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns. Has: {df.columns.tolist()}")
                return None

            # Add volume if missing
            if 'volume' not in df.columns:
                df['volume'] = 0
                logger.warning("Volume column not found, using 0")

            # Select and reorder columns
            df = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()

            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove NaN rows
            df = df.dropna()

            # Sort by time
            df = df.sort_values('time').reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Error standardizing CSV format: {e}", exc_info=True)
            return None

    def load_all_timeframes(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Load all available timeframes for a symbol

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with timeframe: DataFrame
        """
        timeframes = ['M1', 'M5', 'M15', 'H1', 'H4', 'D']
        data = {}

        for tf in timeframes:
            df = self.load_csv(symbol, tf)
            if df is not None:
                data[tf] = df

        logger.info(f"Loaded {len(data)} timeframes for {symbol}: {list(data.keys())}")
        return data

    def validate_csv(self, symbol: str, timeframe: str) -> bool:
        """
        Validate CSV file format and data quality

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            True if valid, False otherwise
        """
        df = self.load_csv(symbol, timeframe)

        if df is None:
            print(f"❌ Failed to load CSV for {symbol} {timeframe}")
            return False

        # Check data quality
        issues = []

        if len(df) < 100:
            issues.append(f"Too few candles: {len(df)} (need at least 100)")

        if df['open'].isna().any():
            issues.append("Contains NaN values")

        if (df['high'] < df['low']).any():
            issues.append("High < Low in some candles")

        if (df['close'] > df['high']).any() or (df['close'] < df['low']).any():
            issues.append("Close outside High/Low range")

        if len(issues) > 0:
            print(f"⚠️ Validation issues for {symbol} {timeframe}:")
            for issue in issues:
                print(f"  - {issue}")
            return False

        print(f"✅ CSV is valid: {symbol} {timeframe}")
        print(f"  - Candles: {len(df)}")
        print(f"  - Date range: {df['time'].min()} to {df['time'].max()}")
        print(f"  - Price range: {df['low'].min():.5f} to {df['high'].max():.5f}")
        return True


def download_from_mt5(symbol: str, timeframe: str, start_date: str, end_date: str,
                      output_folder: str = "historical_data") -> bool:
    """
    Download historical data from MetaTrader 5 and save to CSV

    Args:
        symbol: MT5 symbol (e.g., "EURUSD")
        timeframe: M1, M5, M15, H1, H4, D
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_folder: Folder to save CSV

    Returns:
        True if successful, False otherwise
    """
    try:
        import MetaTrader5 as mt5
        from datetime import datetime

        # Initialize MT5
        if not mt5.initialize():
            print("❌ Failed to initialize MetaTrader5")
            return False

        print(f"✅ Connected to MetaTrader5")

        # Map timeframe
        tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D': mt5.TIMEFRAME_D1
        }

        mt5_tf = tf_map.get(timeframe)
        if mt5_tf is None:
            print(f"❌ Invalid timeframe: {timeframe}")
            return False

        # Get data
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        print(f"📥 Downloading {symbol} {timeframe} from {start_date} to {end_date}...")

        rates = mt5.copy_rates_range(symbol, mt5_tf, start, end)

        if rates is None or len(rates) == 0:
            print(f"❌ No data received from MT5")
            mt5.shutdown()
            return False

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Rename columns
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume'
        })

        # Select columns
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

        # Save to CSV
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        csv_path = os.path.join(output_folder, f"{symbol}_{timeframe}.csv")
        df.to_csv(csv_path, index=False)

        print(f"✅ Downloaded {len(df)} candles")
        print(f"💾 Saved to: {csv_path}")

        mt5.shutdown()
        return True

    except ImportError:
        print("❌ MetaTrader5 module not installed")
        print("   Install: pip install MetaTrader5")
        return False
    except Exception as e:
        print(f"❌ Error downloading from MT5: {e}")
        return False


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("CSV DATA LOADER - Testing")
    print("=" * 80)

    loader = CSVDataLoader("historical_data")

    # Test loading
    print("\n📂 Looking for CSV files in 'historical_data' folder...")
    print("   Expected format: SYMBOL_TIMEFRAME.csv (e.g., EURUSD_H1.csv)")

    # Try to load
    df = loader.load_csv("EURUSD", "H1")

    if df is not None:
        print(f"\n✅ Successfully loaded data")
        print(f"   Candles: {len(df)}")
        print(f"   Date range: {df['time'].min()} to {df['time'].max()}")
        print(f"\n   First 5 rows:")
        print(df.head())
    else:
        print("\n❌ No CSV files found")
        print("\n💡 To use CSV data loader:")
        print("   1. Create 'historical_data' folder")
        print("   2. Add CSV files with format: EURUSD_H1.csv")
        print("   3. CSV should have: time, open, high, low, close, volume")
        print("\n📖 See DATA_SOURCES_GUIDE_FA.md for download instructions")
