"""
Convert HistData.com format to backtest-ready CSV format
Also supports resampling M1 data to higher timeframes
"""

import pandas as pd
import os
import sys


def convert_histdata_to_csv(input_file, symbol, timeframe='M1', output_folder='historical_data'):
    """
    Convert HistData.com format to our CSV format

    Args:
        input_file: Input CSV file from HistData.com
        symbol: Symbol name (e.g., "EURUSD")
        timeframe: Timeframe of the data (M1, M5, etc.)
        output_folder: Folder to save converted files

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\n📥 Converting {input_file}...")
        print(f"   Symbol: {symbol}")
        print(f"   Timeframe: {timeframe}")

        # Read file (try different separators)
        try:
            df = pd.read_csv(input_file,
                           names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
        except:
            df = pd.read_csv(input_file,
                           names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'],
                           sep=';')

        print(f"   ✅ Loaded {len(df)} rows")

        # Convert datetime
        try:
            # Try format 1: 20240101 000000
            df['time'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H%M%S')
        except:
            try:
                # Try format 2: 2024-01-01 00:00:00
                df['time'] = pd.to_datetime(df['DateTime'])
            except:
                print(f"   ❌ Could not parse DateTime format")
                return False

        # Rename columns
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # Select columns
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()

        # Remove duplicates
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['time'])
        after_dedup = len(df)
        if before_dedup != after_dedup:
            print(f"   ⚠️  Removed {before_dedup - after_dedup} duplicate rows")

        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)

        # Remove NaN
        df = df.dropna()

        # Create output folder
        os.makedirs(output_folder, exist_ok=True)

        # Save
        output_file = os.path.join(output_folder, f"{symbol}_{timeframe}.csv")
        df.to_csv(output_file, index=False)

        print(f"   ✅ Saved to: {output_file}")
        print(f"   📊 {len(df)} candles")
        print(f"   📅 From: {df['time'].min()}")
        print(f"   📅 To: {df['time'].max()}")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def resample_to_higher_timeframes(symbol, output_folder='historical_data'):
    """
    Resample M1 data to M5, M15, H1, H4, D

    Args:
        symbol: Symbol name (e.g., "EURUSD")
        output_folder: Folder containing M1 CSV and where to save resampled files

    Returns:
        Number of timeframes created
    """
    try:
        m1_file = os.path.join(output_folder, f"{symbol}_M1.csv")

        if not os.path.exists(m1_file):
            print(f"❌ M1 file not found: {m1_file}")
            return 0

        print(f"\n📊 Resampling {symbol} M1 data to higher timeframes...")

        # Read M1
        df = pd.read_csv(m1_file)
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')

        print(f"   Loaded M1: {len(df)} candles")

        timeframes_created = 0

        # Resample to M5
        print("   Creating M5...")
        df_m5 = df.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        df_m5 = df_m5.reset_index()
        output_file = os.path.join(output_folder, f"{symbol}_M5.csv")
        df_m5.to_csv(output_file, index=False)
        print(f"   ✅ M5: {len(df_m5)} candles → {output_file}")
        timeframes_created += 1

        # Resample to M15
        print("   Creating M15...")
        df_m15 = df.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        df_m15 = df_m15.reset_index()
        output_file = os.path.join(output_folder, f"{symbol}_M15.csv")
        df_m15.to_csv(output_file, index=False)
        print(f"   ✅ M15: {len(df_m15)} candles → {output_file}")
        timeframes_created += 1

        # Resample to H1
        print("   Creating H1...")
        df_h1 = df.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        df_h1 = df_h1.reset_index()
        output_file = os.path.join(output_folder, f"{symbol}_H1.csv")
        df_h1.to_csv(output_file, index=False)
        print(f"   ✅ H1: {len(df_h1)} candles → {output_file}")
        timeframes_created += 1

        # Resample to H4
        print("   Creating H4...")
        df_h4 = df.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        df_h4 = df_h4.reset_index()
        output_file = os.path.join(output_folder, f"{symbol}_H4.csv")
        df_h4.to_csv(output_file, index=False)
        print(f"   ✅ H4: {len(df_h4)} candles → {output_file}")
        timeframes_created += 1

        # Resample to Daily
        print("   Creating D (Daily)...")
        df_d = df.resample('1d').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        df_d = df_d.reset_index()
        output_file = os.path.join(output_folder, f"{symbol}_D.csv")
        df_d.to_csv(output_file, index=False)
        print(f"   ✅ D: {len(df_d)} candles → {output_file}")
        timeframes_created += 1

        print(f"\n✅ Created {timeframes_created} timeframes successfully!")

        return timeframes_created

    except Exception as e:
        print(f"❌ Error resampling: {e}")
        return 0


def batch_convert(folder_path, symbol, output_folder='historical_data'):
    """
    Convert multiple HistData files in a folder

    Args:
        folder_path: Folder containing HistData CSV files
        symbol: Symbol name
        output_folder: Output folder
    """
    import glob

    print(f"\n📂 Scanning folder: {folder_path}")

    # Find all CSV files
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        print(f"❌ No CSV files found in {folder_path}")
        return

    print(f"   Found {len(csv_files)} CSV files")

    # Convert each file
    success_count = 0
    for csv_file in csv_files:
        if convert_histdata_to_csv(csv_file, symbol, 'M1', output_folder):
            success_count += 1

    print(f"\n✅ Converted {success_count}/{len(csv_files)} files successfully!")

    # Ask if user wants to resample
    if success_count > 0:
        print("\n📊 Do you want to create higher timeframes (M5, M15, H1, H4, D)?")
        response = input("   (y/n): ").lower()
        if response == 'y':
            resample_to_higher_timeframes(symbol, output_folder)


def main():
    """Main function"""
    print("=" * 80)
    print("HISTDATA.COM TO CSV CONVERTER")
    print("Convert HistData files for backtesting")
    print("=" * 80)

    if len(sys.argv) < 3:
        print("\n📖 Usage:")
        print("   python convert_histdata.py <input_file> <symbol>")
        print("\n   Example:")
        print("   python convert_histdata.py DAT_MT_EURUSD_M1_202401.csv EURUSD")
        print("\n📖 Batch convert:")
        print("   python convert_histdata.py <folder> <symbol> --batch")
        print("\n   Example:")
        print("   python convert_histdata.py ./downloads EURUSD --batch")
        return

    input_path = sys.argv[1]
    symbol = sys.argv[2].upper()

    # Check if batch mode
    if len(sys.argv) > 3 and sys.argv[3] == '--batch':
        batch_convert(input_path, symbol)
    else:
        # Single file conversion
        if convert_histdata_to_csv(input_path, symbol, 'M1'):
            print("\n📊 Do you want to create higher timeframes (M5, M15, H1, H4, D)?")
            response = input("   (y/n): ").lower()
            if response == 'y':
                resample_to_higher_timeframes(symbol)

    print("\n" + "=" * 80)
    print("✅ Done!")
    print("\nNext steps:")
    print("1. Check 'historical_data' folder for converted files")
    print("2. Edit run_backtest.py:")
    print("   - Set USE_CSV = True")
    print(f"   - Set SYMBOL = '{symbol}'")
    print("3. Run: python run_backtest.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
