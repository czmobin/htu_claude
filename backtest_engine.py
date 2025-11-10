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

    def load_historical_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Load historical data for multiple timeframes

        Args:
            start_date: Start date in format "YYYY-MM-DD"
            end_date: End date in format "YYYY-MM-DD"

        Returns:
            Dictionary with dataframes for each timeframe
        """
        logger.info(f"Loading historical data from {start_date} to {end_date}")

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

            # M15 data
            df_m15 = ticker.history(start=start_date, end=end_date, interval='15m')
            if not df_m15.empty:
                data['M15'] = self._format_dataframe(df_m15)
                logger.info(f"M15: {len(data['M15'])} candles loaded")

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

        if htf not in data or ltf not in data or 'D' not in data:
            logger.error(f"Required timeframes not available. Need: {htf}, {ltf}, D")
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
                'win_rate': 0,
                'total_profit': 0,
                'trades': []
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
