"""
Run Backtest for ICT Trading Bot
Test your strategy on historical data and see Win Rate and performance

Usage:
    python run_backtest.py

Customize the parameters below to test different periods and symbols
"""

from backtest_engine import BacktestEngine
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def main():
    """Run backtest with your parameters"""

    print("\n" + "=" * 80)
    print("ICT TRADING BOT - BACKTESTING SYSTEM")
    print("Test your strategy on historical data")
    print("=" * 80 + "\n")

    # ========== CONFIGURATION ==========

    # Symbol to test (Yahoo Finance format)
    # Examples:
    # - "EURUSD=X" for EUR/USD
    # - "GBPUSD=X" for GBP/USD
    # - "USDJPY=X" for USD/JPY
    # - "AAPL" for Apple stock
    SYMBOL = "EURUSD=X"

    # Date range for backtest
    # IMPORTANT: Yahoo Finance limitation:
    # - M15/M5 data: Only last 60 days available
    # - H1 data: Several months available
    # For longer periods, the system will automatically use H1 data
    START_DATE = "2024-09-01"  # Format: YYYY-MM-DD (last 60 days recommended)
    END_DATE = "2024-11-01"    # Format: YYYY-MM-DD

    # Account settings
    INITIAL_BALANCE = 10000  # Starting balance in dollars
    LOT_SIZE = 1000          # Position size (1000 = micro lot)

    # Strategy selection
    USE_M5_M1 = False  # True = M5+M1 strategy, False = M15+M5 strategy

    # ===================================

    print(f"📊 Symbol: {SYMBOL}")
    print(f"📅 Period: {START_DATE} to {END_DATE}")
    print(f"💰 Initial Balance: ${INITIAL_BALANCE}")
    print(f"📈 Lot Size: {LOT_SIZE}")
    print(f"⚙️  Strategy: {'M5+M1' if USE_M5_M1 else 'M15+M5'}")
    print("\n" + "-" * 80 + "\n")

    # Initialize backtest engine
    engine = BacktestEngine(
        symbol=SYMBOL,
        initial_balance=INITIAL_BALANCE,
        lot_size=LOT_SIZE,
        timezone_str="America/New_York"
    )

    # Load historical data
    print("📥 Loading historical data from Yahoo Finance...")
    print("⏳ This may take a moment...\n")

    data = engine.load_historical_data(START_DATE, END_DATE)

    if not data:
        print("❌ Failed to load historical data. Please check:")
        print("   1. Symbol format is correct (e.g., EURUSD=X)")
        print("   2. Date range is valid")
        print("   3. Internet connection is working")
        print("   4. yfinance package is installed: pip install yfinance")
        return

    print("✅ Data loaded successfully!\n")

    # Run backtest
    print("🚀 Running backtest...")
    print("⏳ Analyzing historical candles and generating signals...\n")

    results = engine.run_backtest(data, use_m5_m1=USE_M5_M1)

    # Print results
    engine.print_results(results)

    # Save results to files
    print("\n📁 Saving results to files...")

    engine.save_results_to_csv(results, filename="backtest_trades.csv")
    engine.plot_equity_curve(results, filename="backtest_equity_curve.png")

    print("\n✅ Backtest completed successfully!")
    print("\nGenerated files:")
    print("├─ backtest_trades.csv - Detailed trade history")
    print("└─ backtest_equity_curve.png - Account balance chart")

    # Performance summary
    print("\n" + "=" * 80)
    print("🎯 QUICK SUMMARY")
    print("=" * 80)

    if results['total_trades'] > 0:
        emoji_wr = "🎯" if results['win_rate'] >= 50 else "⚠️"
        emoji_pf = "✅" if results['profit_factor'] > 1 else "❌"
        emoji_roi = "📈" if results['roi'] > 0 else "📉"

        print(f"{emoji_wr} Win Rate: {results['win_rate']:.1f}%")
        print(f"{emoji_pf} Profit Factor: {results['profit_factor']:.2f}")
        print(f"{emoji_roi} ROI: {results['roi']:.1f}%")
        print(f"📊 Total Trades: {results['total_trades']}")

        if results['win_rate'] >= 40 and results['profit_factor'] > 1.5:
            print("\n🌟 Strategy Performance: GOOD - Consider live testing")
        elif results['win_rate'] >= 35 and results['profit_factor'] > 1.2:
            print("\n⭐ Strategy Performance: ACCEPTABLE - Needs optimization")
        else:
            print("\n⚠️ Strategy Performance: POOR - Requires major adjustments")
    else:
        print("❌ No trades were generated during this period")
        print("💡 Try adjusting:")
        print("   - Longer date range")
        print("   - Different symbol")
        print("   - Different timeframe strategy (M5+M1 vs M15+M5)")

    print("\n" + "=" * 80 + "\n")


def run_quick_test():
    """
    Quick test with recent data (last 60 days)
    Useful for fast testing
    """
    from datetime import datetime, timedelta

    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)

    print("\n🚀 QUICK TEST MODE (Last 60 days)")

    engine = BacktestEngine(
        symbol="EURUSD=X",
        initial_balance=10000,
        lot_size=1000
    )

    data = engine.load_historical_data(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )

    if data:
        results = engine.run_backtest(data, use_m5_m1=False)
        engine.print_results(results)

        if results['total_trades'] > 0:
            print(f"\n✅ Win Rate: {results['win_rate']:.1f}%")
            print(f"✅ Profit Factor: {results['profit_factor']:.2f}")
    else:
        print("❌ Failed to load data for quick test")


if __name__ == "__main__":
    # Run full backtest
    main()

    # Uncomment below to run quick test instead:
    # run_quick_test()
