"""
ICT Trading Bot - Backtesting Engine
Test your strategy on historical data to calculate Win Rate and performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import logging
from typing import Dict, List, Optional, Tuple
import warnings
from io import BytesIO

warnings.filterwarnings('ignore')

# Import strategy components from main bot
from ict_bot_complete import ICTStrategy, SimulatedProvider, logger


class BacktestEngine:
    """
    Backtesting engine for ICT Trading Strategy
    Tests strategy on historical data and calculates performance metrics
    """

    def __init__(self, symbol: str, initial_balance: float = 10000,
                 lot_size: float = 1000, timezone_str: str = "America/New_York"):
        """
        Initialize backtesting engine

        Args:
            symbol: Trading symbol (e.g., "EURUSD=X" for Yahoo Finance)
            initial_balance: Starting account balance
            lot_size: Position size in units
            timezone_str: Timezone for session filtering
        """
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.lot_size = lot_size

        self.strategy = ICTStrategy(symbol, timezone_str)
        self.trades = []
        self.equity_curve = []

        logger.info(f"Backtest Engine initialized for {symbol}")
        logger.info(f"Initial Balance: ${initial_balance}")

    def load_historical_data(self, start_date: str, end_date: str,
                             use_csv: bool = False, csv_folder: str = "historical_data") -> Dict[str, pd.DataFrame]:
        """
        Load historical data for multiple timeframes

        Args:
            start_date: Start date in format "YYYY-MM-DD"
            end_date: End date in format "YYYY-MM-DD"
            use_csv: If True, load from CSV files instead of Yahoo Finance
            csv_folder: Folder containing CSV files

        Returns:
            Dictionary with dataframes for each timeframe
        """
        logger.info(f"Loading historical data from {start_date} to {end_date}")

        # Option 1: Load from CSV files
        if use_csv:
            logger.info(f"Loading data from CSV files in '{csv_folder}'")
            return self._load_from_csv(csv_folder, start_date, end_date)

        # Option 2: Load from Yahoo Finance
        try:
            import yfinance as yf
            ticker = yf.Ticker(self.symbol)

            # Calculate period
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            period_days = (end - start).days

            # Load different timeframes
            data = {}

            # M1 data (limited to recent months due to Yahoo limitations)
            if period_days <= 7:
                df_m1 = ticker.history(start=start_date, end=end_date, interval='1m')
                if not df_m1.empty:
                    data['M1'] = self._format_dataframe(df_m1)
                    logger.info(f"M1: {len(data['M1'])} candles loaded")

            # M5 data
            if period_days <= 60:
                df_m5 = ticker.history(start=start_date, end=end_date, interval='5m')
                if not df_m5.empty:
                    data['M5'] = self._format_dataframe(df_m5)
                    logger.info(f"M5: {len(data['M5'])} candles loaded")

            # M15 data (for fallback strategy with H1)
            if period_days <= 60:
                df_m15 = ticker.history(start=start_date, end=end_date, interval='15m')
                if not df_m15.empty:
                    data['M15'] = self._format_dataframe(df_m15)
                    logger.info(f"M15: {len(data['M15'])} candles loaded")
            else:
                logger.info(f"M15 skipped (period > 60 days). Will use H1 as fallback.")

            # H1 data
            df_h1 = ticker.history(start=start_date, end=end_date, interval='1h')
            if not df_h1.empty:
                data['H1'] = self._format_dataframe(df_h1)
                logger.info(f"H1: {len(data['H1'])} candles loaded")

            # Daily data
            df_d = ticker.history(start=start_date, end=end_date, interval='1d')
            if not df_d.empty:
                data['D'] = self._format_dataframe(df_d)
                logger.info(f"D: {len(data['D'])} candles loaded")

            return data

        except Exception as e:
            logger.error(f"Error loading historical data: {e}", exc_info=True)
            return {}

    def _load_from_csv(self, csv_folder: str, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Load data from CSV files

        Args:
            csv_folder: Folder containing CSV files
            start_date: Filter data from this date
            end_date: Filter data until this date

        Returns:
            Dictionary with timeframes
        """
        try:
            from csv_data_loader import CSVDataLoader

            loader = CSVDataLoader(csv_folder)

            # Extract symbol name (remove =X suffix for CSV)
            csv_symbol = self.symbol.replace('=X', '').replace('/', '')

            # Load all timeframes
            data = loader.load_all_timeframes(csv_symbol)

            # Filter by date range
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)

            filtered_data = {}
            for tf, df in data.items():
                mask = (df['time'] >= start) & (df['time'] <= end)
                df_filtered = df[mask].copy()
                if len(df_filtered) > 0:
                    filtered_data[tf] = df_filtered
                    logger.info(f"{tf}: {len(df_filtered)} candles (filtered)")

            return filtered_data

        except ImportError:
            logger.error("csv_data_loader module not found")
            return {}
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}", exc_info=True)
            return {}

    def _format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format dataframe to match bot's expected format"""
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df = df.reset_index()
        df = df.rename(columns={'index': 'time', 'Datetime': 'time', 'Date': 'time'})
        return df

    def run_backtest(self, data: Dict[str, pd.DataFrame],
                     use_m5_m1: bool = True) -> Dict:
        """
        Run backtest on historical data

        Args:
            data: Dictionary with historical data for each timeframe
            use_m5_m1: If True, use M5+M1 strategy; if False, use M15+M5

        Returns:
            Dictionary with backtest results and statistics
        """
        logger.info("=" * 80)
        logger.info("STARTING BACKTEST")
        logger.info("=" * 80)

        # Choose timeframes based on strategy
        if use_m5_m1:
            htf = 'M5'  # Higher timeframe for liquidity
            ltf = 'M1'  # Lower timeframe for MSS
            logger.info("Strategy: M5 (liquidity) + M1 (MSS)")
        else:
            htf = 'M15'
            ltf = 'M5'
            logger.info("Strategy: M15 (liquidity) + M5 (MSS)")

        # Fallback: Use H1 if M15/M5 not available (for longer date ranges)
        if (htf not in data or ltf not in data) and 'H1' in data:
            logger.warning(f"{htf} or {ltf} not available. Falling back to H1 (liquidity) + M15 (MSS)")
            htf = 'H1'
            ltf = 'M15' if 'M15' in data else 'H1'
            logger.info(f"Using fallback strategy: {htf} + {ltf}")

        if htf not in data or ltf not in data or 'D' not in data:
            logger.error(f"Required timeframes not available. Need: {htf}, {ltf}, D")
            logger.error(f"Available timeframes: {list(data.keys())}")
            logger.error("TIP: For date ranges > 60 days, Yahoo Finance doesn't provide M15/M5 data.")
            logger.error("     Try: 1) Shorter date range (last 60 days), or 2) Use H1 data")
            return self._generate_results()

        df_htf = data[htf]
        df_ltf = data[ltf]
        df_daily = data['D']

        # Track last signal time to avoid duplicates
        last_signal_time = None
        cooldown_minutes = 60  # 1 hour cooldown between signals

        # Iterate through HTF candles
        for htf_idx in range(100, len(df_htf)):
            current_time = df_htf['time'].iloc[htf_idx]

            # Check trading session
            is_trading, current_hour = self.strategy.is_trading_session(current_time)
            if not is_trading:
                continue

            # Cooldown check
            if last_signal_time:
                time_diff = (current_time - last_signal_time).total_seconds() / 60
                if time_diff < cooldown_minutes:
                    continue

            # Get HTF window
            htf_window = df_htf.iloc[max(0, htf_idx-200):htf_idx+1].copy()

            # STEP 1: Check liquidity sweep on HTF
            swings_htf = self.strategy.identify_swing_high_low(htf_window)
            current_price = htf_window['close'].iloc[-1]

            # Get daily open for bias
            daily_idx = self._find_daily_index(df_daily, current_time)
            if daily_idx is None or daily_idx < 0:
                continue

            daily_open = df_daily['open'].iloc[daily_idx]

            liquidity_swept = None
            direction = None

            # Search for SSL Sweep (Buy setup)
            if current_price < daily_open:
                if len(swings_htf['lows']) > 0:
                    for swing in reversed(swings_htf['lows'][-3:]):
                        swept, idx = self.strategy.check_liquidity_sweep(
                            htf_window, swing['price'], is_high=False
                        )
                        if swept:
                            liquidity_swept = swing
                            direction = 'bullish'
                            break

            # Search for BSL Sweep (Sell setup)
            elif current_price > daily_open:
                if len(swings_htf['highs']) > 0:
                    for swing in reversed(swings_htf['highs'][-3:]):
                        swept, idx = self.strategy.check_liquidity_sweep(
                            htf_window, swing['price'], is_high=True
                        )
                        if swept:
                            liquidity_swept = swing
                            direction = 'bearish'
                            break

            if liquidity_swept is None:
                continue

            # STEP 2: Get LTF data and check for MSS
            ltf_idx = self._find_ltf_index(df_ltf, current_time)
            if ltf_idx is None or ltf_idx < 50:
                continue

            ltf_window = df_ltf.iloc[max(0, ltf_idx-500):ltf_idx+1].copy()
            swings_ltf = self.strategy.identify_swing_high_low(ltf_window)

            # Find intermediate swing
            intermediate_swing, swing_type = self.strategy.find_intermediate_swing(
                ltf_window, swings_ltf, direction
            )

            if intermediate_swing is None:
                continue

            # Confirm MSS
            mss_confirmed, mss_idx = self.strategy.detect_mss(
                ltf_window, intermediate_swing, direction
            )

            if not mss_confirmed:
                continue

            # STEP 3: Find entry zone
            fvgs_ltf = self.strategy.identify_fvg(ltf_window)
            entry_zone = None

            # Priority 1: FVG
            if direction == 'bullish' and len(fvgs_ltf['bullish']) > 0:
                for fvg in reversed(fvgs_ltf['bullish']):
                    if fvg['index'] >= mss_idx:
                        entry_zone = fvg
                        entry_zone['entry_type'] = 'FVG'
                        break
            elif direction == 'bearish' and len(fvgs_ltf['bearish']) > 0:
                for fvg in reversed(fvgs_ltf['bearish']):
                    if fvg['index'] >= mss_idx:
                        entry_zone = fvg
                        entry_zone['entry_type'] = 'FVG'
                        break

            # Priority 2: Order Block
            if entry_zone is None:
                ob = self.strategy.identify_order_block(ltf_window, mss_idx, direction)
                if ob:
                    entry_zone = ob
                    entry_zone['entry_type'] = 'OB'

            if entry_zone is None:
                continue

            # STEP 4: Calculate trade parameters
            if direction == 'bullish':
                entry = entry_zone.get('bottom', entry_zone.get('low'))
                zone_height = entry_zone.get('top', entry_zone.get('high')) - entry
                sl = entry - (zone_height * 0.2)
                risk = entry - sl
                tp = entry + (risk * 3)
                side = "BUY"
            else:
                entry = entry_zone.get('top', entry_zone.get('high'))
                zone_height = entry - entry_zone.get('bottom', entry_zone.get('low'))
                sl = entry + (zone_height * 0.2)
                risk = sl - entry
                tp = entry - (risk * 3)
                side = "SELL"

            # STEP 5: Simulate trade execution
            trade_result = self._simulate_trade(
                df_ltf, ltf_idx, entry, sl, tp, direction, current_time
            )

            if trade_result:
                trade_result.update({
                    'entry_time': current_time,
                    'side': side,
                    'direction': direction,
                    'entry_price': entry,
                    'sl': sl,
                    'tp': tp,
                    'risk': abs(risk),
                    'reward': abs(risk * 3),
                    'liquidity_type': liquidity_swept['type'],
                    'entry_type': entry_zone['entry_type'],
                    'daily_open': daily_open
                })

                self.trades.append(trade_result)
                self.current_balance += trade_result['profit']
                self.equity_curve.append({
                    'time': trade_result['exit_time'],
                    'balance': self.current_balance
                })

                logger.info(f"Trade #{len(self.trades)}: {side} | Result: {trade_result['result']} | "
                           f"P/L: ${trade_result['profit']:.2f} | Balance: ${self.current_balance:.2f}")

                last_signal_time = current_time

        logger.info("=" * 80)
        logger.info(f"BACKTEST COMPLETED: {len(self.trades)} trades executed")
        logger.info("=" * 80)

        return self._generate_results()

    def _simulate_trade(self, df: pd.DataFrame, start_idx: int,
                       entry: float, sl: float, tp: float,
                       direction: str, entry_time: datetime) -> Optional[Dict]:
        """
        Simulate trade execution and find exit point

        Returns:
            Dictionary with trade result or None if trade not executed
        """
        # Look forward to find entry, SL, or TP hit
        max_forward_candles = 200  # Maximum candles to check

        for i in range(start_idx, min(start_idx + max_forward_candles, len(df))):
            candle = df.iloc[i]

            if direction == 'bullish':
                # Check if entry hit
                if candle['low'] <= entry:
                    # Now check for SL or TP
                    for j in range(i, min(i + max_forward_candles, len(df))):
                        exit_candle = df.iloc[j]

                        # Check SL hit first (more conservative)
                        if exit_candle['low'] <= sl:
                            pips = sl - entry
                            profit = self._calculate_profit(pips, self.lot_size)
                            return {
                                'exit_time': exit_candle['time'],
                                'exit_price': sl,
                                'result': 'LOSS',
                                'profit': profit,
                                'pips': pips,
                                'duration_candles': j - i
                            }

                        # Check TP hit
                        if exit_candle['high'] >= tp:
                            pips = tp - entry
                            profit = self._calculate_profit(pips, self.lot_size)
                            return {
                                'exit_time': exit_candle['time'],
                                'exit_price': tp,
                                'result': 'WIN',
                                'profit': profit,
                                'pips': pips,
                                'duration_candles': j - i
                            }

                    # If neither hit within max candles, consider it a loss
                    return None

            else:  # bearish
                # Check if entry hit
                if candle['high'] >= entry:
                    # Now check for SL or TP
                    for j in range(i, min(i + max_forward_candles, len(df))):
                        exit_candle = df.iloc[j]

                        # Check SL hit first
                        if exit_candle['high'] >= sl:
                            pips = entry - sl
                            profit = self._calculate_profit(-pips, self.lot_size)
                            return {
                                'exit_time': exit_candle['time'],
                                'exit_price': sl,
                                'result': 'LOSS',
                                'profit': profit,
                                'pips': -pips,
                                'duration_candles': j - i
                            }

                        # Check TP hit
                        if exit_candle['low'] <= tp:
                            pips = entry - tp
                            profit = self._calculate_profit(pips, self.lot_size)
                            return {
                                'exit_time': exit_candle['time'],
                                'exit_price': tp,
                                'result': 'WIN',
                                'profit': profit,
                                'pips': pips,
                                'duration_candles': j - i
                            }

                    return None

        return None

    def _calculate_profit(self, pips: float, lot_size: float) -> float:
        """Calculate profit/loss in dollars based on pips and lot size"""
        # For Forex: 1 pip = 0.0001 for most pairs
        # Standard lot (100,000 units): 1 pip = $10
        # Mini lot (10,000 units): 1 pip = $1
        # Micro lot (1,000 units): 1 pip = $0.10

        pip_value = (lot_size / 1000) * 0.10  # Assuming micro lot calculation
        profit = pips * pip_value * 10000  # Convert pips to dollars
        return profit

    def _find_daily_index(self, df_daily: pd.DataFrame, current_time: datetime) -> Optional[int]:
        """Find the daily candle index for given datetime"""
        for i in range(len(df_daily) - 1, -1, -1):
            if df_daily['time'].iloc[i].date() <= current_time.date():
                return i
        return None

    def _find_ltf_index(self, df_ltf: pd.DataFrame, current_time: datetime) -> Optional[int]:
        """Find the LTF candle index closest to given datetime"""
        for i in range(len(df_ltf) - 1, -1, -1):
            if df_ltf['time'].iloc[i] <= current_time:
                return i
        return None

    def _generate_results(self) -> Dict:
        """Generate comprehensive backtest results and statistics"""
        if not self.trades:
            logger.warning("No trades executed during backtest period")
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'total_wins_profit': 0,
                'total_loss_profit': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'initial_balance': self.initial_balance,
                'final_balance': self.initial_balance,
                'roi': 0,
                'trades': [],
                'equity_curve': []
            }

        # Calculate statistics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['result'] == 'WIN']
        losing_trades = [t for t in self.trades if t['result'] == 'LOSS']

        wins = len(winning_trades)
        losses = len(losing_trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        total_profit = sum(t['profit'] for t in self.trades)
        total_wins_profit = sum(t['profit'] for t in winning_trades)
        total_loss_profit = abs(sum(t['profit'] for t in losing_trades))

        avg_win = total_wins_profit / wins if wins > 0 else 0
        avg_loss = total_loss_profit / losses if losses > 0 else 0

        profit_factor = total_wins_profit / total_loss_profit if total_loss_profit > 0 else 0

        # Calculate max drawdown
        max_balance = self.initial_balance
        max_drawdown = 0
        for equity_point in self.equity_curve:
            if equity_point['balance'] > max_balance:
                max_balance = equity_point['balance']
            drawdown = ((max_balance - equity_point['balance']) / max_balance * 100)
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        roi = ((self.current_balance - self.initial_balance) / self.initial_balance * 100)

        results = {
            'total_trades': total_trades,
            'winning_trades': wins,
            'losing_trades': losses,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_wins_profit': total_wins_profit,
            'total_loss_profit': total_loss_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'roi': roi,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }

        return results

    def print_results(self, results: Dict):
        """Print formatted backtest results"""
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        print(f"\n📊 TRADING STATISTICS:")
        print(f"├─ Total Trades: {results['total_trades']}")
        print(f"├─ Winning Trades: {results['winning_trades']} ✅")
        print(f"├─ Losing Trades: {results['losing_trades']} ❌")
        print(f"└─ Win Rate: {results['win_rate']:.2f}% {'🎯' if results['win_rate'] >= 50 else '⚠️'}")

        print(f"\n💰 PROFIT & LOSS:")
        print(f"├─ Total Profit: ${results['total_profit']:.2f}")
        print(f"├─ Gross Wins: ${results['total_wins_profit']:.2f}")
        print(f"├─ Gross Losses: ${results['total_loss_profit']:.2f}")
        print(f"├─ Average Win: ${results['avg_win']:.2f}")
        print(f"├─ Average Loss: ${results['avg_loss']:.2f}")
        print(f"└─ Profit Factor: {results['profit_factor']:.2f} {'✅' if results['profit_factor'] > 1 else '❌'}")

        print(f"\n📈 ACCOUNT PERFORMANCE:")
        print(f"├─ Initial Balance: ${results['initial_balance']:.2f}")
        print(f"├─ Final Balance: ${results['final_balance']:.2f}")
        print(f"├─ ROI: {results['roi']:.2f}%")
        print(f"└─ Max Drawdown: {results['max_drawdown']:.2f}%")

        print("\n" + "=" * 80)

        # Print sample trades
        if results['trades']:
            print("\n📋 SAMPLE TRADES (Last 10):")
            for i, trade in enumerate(results['trades'][-10:], 1):
                emoji = "✅" if trade['result'] == 'WIN' else "❌"
                print(f"{emoji} Trade #{len(results['trades']) - 10 + i}: {trade['side']} | "
                      f"{trade['result']} | P/L: ${trade['profit']:.2f} | "
                      f"Entry: {trade['entry_type']} | Time: {trade['entry_time']}")

    def save_results_to_csv(self, results: Dict, filename: str = "backtest_results.csv"):
        """Save trade results to CSV file"""
        if not results['trades']:
            logger.warning("No trades to save")
            return

        df_trades = pd.DataFrame(results['trades'])
        df_trades.to_csv(filename, index=False)
        logger.info(f"Results saved to {filename}")
        print(f"\n💾 Trade history saved to: {filename}")

    def plot_equity_curve(self, results: Dict, filename: str = "equity_curve.png"):
        """Plot and save equity curve"""
        if not results['equity_curve']:
            logger.warning("No equity data to plot")
            return

        df_equity = pd.DataFrame(results['equity_curve'])

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(range(len(df_equity)), df_equity['balance'],
                linewidth=2, color='#2196F3', label='Account Balance')
        ax.axhline(self.initial_balance, color='gray', linestyle='--',
                   linewidth=1, label='Initial Balance', alpha=0.7)

        ax.set_title('Equity Curve - Account Balance Over Time',
                     fontsize=16, weight='bold', pad=20)
        ax.set_xlabel('Trade Number', fontsize=12)
        ax.set_ylabel('Balance ($)', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)

        # Add statistics box
        stats_text = (
            f"Total Trades: {results['total_trades']}\n"
            f"Win Rate: {results['win_rate']:.1f}%\n"
            f"Total P/L: ${results['total_profit']:.2f}\n"
            f"ROI: {results['roi']:.1f}%"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Equity curve saved to {filename}")
        print(f"📊 Equity curve chart saved to: {filename}")

    def plot_trades_on_chart(self, results: Dict, data: Dict[str, pd.DataFrame],
                             filename: str = "trades_chart.png", max_trades: int = 10):
        """
        Plot trades on price chart with entry/exit points

        Args:
            results: Backtest results dictionary
            data: Historical data dictionary with timeframes
            filename: Output filename
            max_trades: Maximum number of recent trades to display
        """
        if not results['trades']:
            logger.warning("No trades to plot")
            return

        # Get recent trades
        trades = results['trades'][-max_trades:]

        if not trades:
            logger.warning("No trades to plot")
            return

        # Determine which timeframe to use
        timeframe = 'M15' if 'M15' in data else ('H1' if 'H1' in data else 'M5')
        if timeframe not in data:
            logger.error(f"Timeframe {timeframe} not available in data")
            return

        df = data[timeframe].copy()

        # Get time range for trades
        trade_times = []
        for trade in trades:
            if isinstance(trade['entry_time'], str):
                trade_times.append(pd.to_datetime(trade['entry_time']))
            else:
                trade_times.append(trade['entry_time'])

            if isinstance(trade['exit_time'], str):
                trade_times.append(pd.to_datetime(trade['exit_time']))
            else:
                trade_times.append(trade['exit_time'])

        # Filter dataframe to relevant time period with some padding
        min_time = min(trade_times) - timedelta(hours=12)
        max_time = max(trade_times) + timedelta(hours=12)

        df_plot = df[(df.index >= min_time) & (df.index <= max_time)].copy()

        if df_plot.empty:
            logger.error("No price data in the trade time range")
            return

        # Limit to reasonable number of candles
        if len(df_plot) > 500:
            df_plot = df_plot.iloc[-500:]

        df_plot = df_plot.reset_index(drop=False)
        df_plot.rename(columns={'index': 'time'}, inplace=True)

        # Create figure
        fig, ax = plt.subplots(figsize=(20, 10))

        # Draw candlesticks
        for idx in range(len(df_plot)):
            o = df_plot['open'].iloc[idx]
            c = df_plot['close'].iloc[idx]
            h = df_plot['high'].iloc[idx]
            l = df_plot['low'].iloc[idx]

            color = '#26a69a' if c >= o else '#ef5350'  # Green/Red

            # Body
            height = abs(c - o)
            bottom = min(o, c)

            if height > 0:
                rect = Rectangle((idx - 0.3, bottom), 0.6, height,
                                facecolor=color, edgecolor=color, alpha=0.8, linewidth=1)
                ax.add_patch(rect)
            else:
                ax.plot([idx - 0.3, idx + 0.3], [o, o], color=color, linewidth=1.5)

            # Wick
            ax.plot([idx, idx], [l, h], color=color, linewidth=1, alpha=0.6)

        # Plot trades
        for trade_idx, trade in enumerate(trades):
            # Find candle indices for entry and exit
            entry_time = pd.to_datetime(trade['entry_time']) if isinstance(trade['entry_time'], str) else trade['entry_time']
            exit_time = pd.to_datetime(trade['exit_time']) if isinstance(trade['exit_time'], str) else trade['exit_time']

            # Find closest candle index
            entry_candle_idx = None
            exit_candle_idx = None

            for idx, row in df_plot.iterrows():
                if entry_candle_idx is None and abs((row['time'] - entry_time).total_seconds()) < 3600:
                    entry_candle_idx = idx
                if exit_candle_idx is None and abs((row['time'] - exit_time).total_seconds()) < 3600:
                    exit_candle_idx = idx

            if entry_candle_idx is None or exit_candle_idx is None:
                continue

            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            sl = trade['sl']
            tp = trade['tp']
            is_win = trade['result'] == 'win'
            is_buy = trade['direction'].upper() == 'BUY'

            # Colors
            trade_color = '#4CAF50' if is_win else '#F44336'  # Green for win, Red for loss
            entry_marker_color = '#2196F3' if is_buy else '#FF9800'  # Blue for buy, Orange for sell

            # Plot entry point
            if is_buy:
                ax.scatter(entry_candle_idx, entry_price, marker='^', s=300,
                          color=entry_marker_color, edgecolors='black', linewidths=2,
                          zorder=10, label='_nolegend_')
                ax.text(entry_candle_idx, entry_price - (df_plot['high'].max() - df_plot['low'].min()) * 0.02,
                       f'BUY #{trade_idx+1}', fontsize=9, ha='center', va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=entry_marker_color, alpha=0.7),
                       fontweight='bold')
            else:
                ax.scatter(entry_candle_idx, entry_price, marker='v', s=300,
                          color=entry_marker_color, edgecolors='black', linewidths=2,
                          zorder=10, label='_nolegend_')
                ax.text(entry_candle_idx, entry_price + (df_plot['high'].max() - df_plot['low'].min()) * 0.02,
                       f'SELL #{trade_idx+1}', fontsize=9, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=entry_marker_color, alpha=0.7),
                       fontweight='bold')

            # Plot exit point
            ax.scatter(exit_candle_idx, exit_price, marker='o', s=200,
                      color=trade_color, edgecolors='black', linewidths=2,
                      zorder=10, label='_nolegend_')

            # Draw line from entry to exit
            ax.plot([entry_candle_idx, exit_candle_idx], [entry_price, exit_price],
                   color=trade_color, linewidth=2, linestyle='--', alpha=0.7, zorder=5)

            # Plot SL and TP levels (only for portion of trade)
            x_range_start = entry_candle_idx
            x_range_end = exit_candle_idx

            # Stop Loss line
            ax.plot([x_range_start, x_range_end], [sl, sl],
                   color='red', linewidth=1.5, linestyle=':', alpha=0.5, zorder=4)

            # Take Profit line
            ax.plot([x_range_start, x_range_end], [tp, tp],
                   color='green', linewidth=1.5, linestyle=':', alpha=0.5, zorder=4)

            # Add profit/loss label at exit
            profit = trade['profit']
            profit_text = f"+${profit:.2f}" if profit > 0 else f"-${abs(profit):.2f}"
            ax.text(exit_candle_idx, exit_price, f'  {profit_text}',
                   fontsize=8, ha='left', va='center',
                   color=trade_color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=trade_color))

        # Configure chart
        ax.set_xlim(-1, len(df_plot))
        y_margin = (df_plot['high'].max() - df_plot['low'].min()) * 0.1
        ax.set_ylim(df_plot['low'].min() - y_margin, df_plot['high'].max() + y_margin)

        ax.set_title(f'Trading Results - Last {len(trades)} Trades on {self.symbol}',
                    fontsize=18, weight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel('Price', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_facecolor('#f9f9f9')

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#2196F3',
                   markersize=12, label='BUY Entry', markeredgecolor='black', markeredgewidth=1.5),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='#FF9800',
                   markersize=12, label='SELL Entry', markeredgecolor='black', markeredgewidth=1.5),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50',
                   markersize=10, label='Win Exit', markeredgecolor='black', markeredgewidth=1.5),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336',
                   markersize=10, label='Loss Exit', markeredgecolor='black', markeredgewidth=1.5),
            Line2D([0], [0], color='red', linewidth=2, linestyle=':', label='Stop Loss'),
            Line2D([0], [0], color='green', linewidth=2, linestyle=':', label='Take Profit')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

        # Add statistics box
        stats_text = (
            f"Trades Shown: {len(trades)}\n"
            f"Wins: {sum(1 for t in trades if t['result'] == 'win')}\n"
            f"Losses: {sum(1 for t in trades if t['result'] == 'loss')}\n"
            f"Total P/L: ${sum(t['profit'] for t in trades):.2f}"
        )
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5))

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Trades chart saved to {filename}")
        print(f"📈 Trades visualization chart saved to: {filename}")
