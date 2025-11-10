"""
Demo script to show trade visualization feature
This creates sample data to demonstrate how trades are shown on charts
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtest_engine import BacktestEngine
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create sample price data for demonstration"""
    # Generate 300 candles (about 25 hours of M5 data)
    num_candles = 300
    start_time = datetime(2024, 11, 1, 0, 0)

    # Create time index
    time_index = [start_time + timedelta(minutes=5*i) for i in range(num_candles)]

    # Generate realistic price movement
    np.random.seed(42)
    base_price = 1.0900

    # Create price series with trend and noise
    trend = np.linspace(0, 0.005, num_candles)  # Slight upward trend
    noise = np.cumsum(np.random.randn(num_candles) * 0.00005)
    prices = base_price + trend + noise

    # Generate OHLC data
    data = []
    for i in range(num_candles):
        open_price = prices[i]
        close_price = prices[i] + np.random.randn() * 0.0002
        high_price = max(open_price, close_price) + abs(np.random.randn() * 0.0001)
        low_price = min(open_price, close_price) - abs(np.random.randn() * 0.0001)

        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': np.random.randint(1000, 10000)
        })

    df = pd.DataFrame(data, index=pd.DatetimeIndex(time_index))
    return df

def create_sample_trades(df):
    """Create sample trade data for demonstration"""
    trades = []

    # Trade 1: Winning BUY trade
    entry_idx = 50
    exit_idx = 80
    entry_price = df['low'].iloc[entry_idx] + 0.00010
    exit_price = entry_price + 0.00090  # +90 pips profit
    sl = entry_price - 0.00030  # 30 pips SL
    tp = entry_price + 0.00090  # 90 pips TP

    trades.append({
        'entry_time': df.index[entry_idx],
        'exit_time': df.index[exit_idx],
        'side': 'LONG',
        'direction': 'BUY',
        'entry_price': entry_price,
        'exit_price': exit_price,
        'sl': sl,
        'tp': tp,
        'result': 'win',
        'profit': 27.0,
        'risk': 9.0,
        'reward': 27.0
    })

    # Trade 2: Losing SELL trade
    entry_idx = 120
    exit_idx = 140
    entry_price = df['high'].iloc[entry_idx] - 0.00010
    exit_price = entry_price + 0.00035  # -35 pips loss (SL hit)
    sl = entry_price + 0.00030  # 30 pips SL
    tp = entry_price - 0.00090  # 90 pips TP

    trades.append({
        'entry_time': df.index[entry_idx],
        'exit_time': df.index[exit_idx],
        'side': 'SHORT',
        'direction': 'SELL',
        'entry_price': entry_price,
        'exit_price': exit_price,
        'sl': sl,
        'tp': tp,
        'result': 'loss',
        'profit': -10.5,
        'risk': 9.0,
        'reward': 27.0
    })

    # Trade 3: Winning BUY trade
    entry_idx = 180
    exit_idx = 220
    entry_price = df['low'].iloc[entry_idx] + 0.00015
    exit_price = entry_price + 0.00095  # +95 pips profit
    sl = entry_price - 0.00030  # 30 pips SL
    tp = entry_price + 0.00090  # 90 pips TP

    trades.append({
        'entry_time': df.index[entry_idx],
        'exit_time': df.index[exit_idx],
        'side': 'LONG',
        'direction': 'BUY',
        'entry_price': entry_price,
        'exit_price': exit_price,
        'sl': sl,
        'tp': tp,
        'result': 'win',
        'profit': 28.5,
        'risk': 9.0,
        'reward': 27.0
    })

    # Trade 4: Winning SELL trade
    entry_idx = 250
    exit_idx = 275
    entry_price = df['high'].iloc[entry_idx] - 0.00012
    exit_price = entry_price - 0.00085  # +85 pips profit
    sl = entry_price + 0.00030  # 30 pips SL
    tp = entry_price - 0.00090  # 90 pips TP

    trades.append({
        'entry_time': df.index[entry_idx],
        'exit_time': df.index[exit_idx],
        'side': 'SHORT',
        'direction': 'SELL',
        'entry_price': entry_price,
        'exit_price': exit_price,
        'sl': sl,
        'tp': tp,
        'result': 'win',
        'profit': 25.5,
        'risk': 9.0,
        'reward': 27.0
    })

    # Trade 5: Losing BUY trade
    entry_idx = 290
    exit_idx = 298
    entry_price = df['low'].iloc[entry_idx] + 0.00008
    exit_price = entry_price - 0.00028  # -28 pips loss (SL hit)
    sl = entry_price - 0.00030  # 30 pips SL
    tp = entry_price + 0.00090  # 90 pips TP

    trades.append({
        'entry_time': df.index[entry_idx],
        'exit_time': df.index[exit_idx],
        'side': 'LONG',
        'direction': 'BUY',
        'entry_price': entry_price,
        'exit_price': exit_price,
        'sl': sl,
        'tp': tp,
        'result': 'loss',
        'profit': -8.4,
        'risk': 9.0,
        'reward': 27.0
    })

    return trades

def main():
    print("\n" + "=" * 80)
    print("📊 TRADE VISUALIZATION DEMO")
    print("Demonstrating the new chart feature that shows trades on price action")
    print("=" * 80 + "\n")

    # Create sample data
    print("🔄 Creating sample price data...")
    df_m5 = create_sample_data()

    data = {
        'M5': df_m5,
        'M15': df_m5.iloc[::3]  # Downsample for M15
    }

    print(f"✅ Created {len(df_m5)} M5 candles")

    # Create sample trades
    print("🔄 Creating sample trades...")
    trades = create_sample_trades(df_m5)

    print(f"✅ Created {len(trades)} sample trades:")
    for i, trade in enumerate(trades, 1):
        result_emoji = "✅" if trade['result'] == 'win' else "❌"
        print(f"   {result_emoji} Trade {i}: {trade['direction']} - ${trade['profit']:.2f}")

    # Create results structure
    results = {
        'trades': trades,
        'total_trades': len(trades),
        'winning_trades': sum(1 for t in trades if t['result'] == 'win'),
        'losing_trades': sum(1 for t in trades if t['result'] == 'loss'),
        'total_profit': sum(t['profit'] for t in trades),
        'equity_curve': []
    }

    # Calculate metrics
    results['win_rate'] = (results['winning_trades'] / results['total_trades']) * 100
    results['roi'] = (results['total_profit'] / 10000) * 100

    print(f"\n📊 Statistics:")
    print(f"   Win Rate: {results['win_rate']:.1f}%")
    print(f"   Total P/L: ${results['total_profit']:.2f}")

    # Initialize engine and create chart
    print("\n🎨 Creating trade visualization chart...")
    engine = BacktestEngine(symbol="EURUSD (DEMO)", initial_balance=10000)

    # Generate the chart
    engine.plot_trades_on_chart(
        results=results,
        data=data,
        filename="demo_trades_chart.png",
        max_trades=10
    )

    print("\n✅ Demo completed successfully!")
    print("\nGenerated file:")
    print("└─ demo_trades_chart.png - Shows all trades on price chart")
    print("\nThe chart includes:")
    print("  • 📈 Candlestick price action")
    print("  • 🔵 BUY entry points (blue up arrows)")
    print("  • 🟠 SELL entry points (orange down arrows)")
    print("  • 🟢 Winning exits (green circles)")
    print("  • 🔴 Losing exits (red circles)")
    print("  • ━━ Stop Loss levels (red dotted lines)")
    print("  • ━━ Take Profit levels (green dotted lines)")
    print("  • 💰 Profit/Loss labels at each exit")

    print("\n" + "=" * 80)
    print("🎯 This feature is now available in your backtesting system!")
    print("Just run 'python run_backtest.py' to see it in action with real data.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
