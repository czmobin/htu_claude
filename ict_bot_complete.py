"""
ICT Trading Bot v2.0 - Completely Rewritten
Based on HTU Trading Strategy PDF
No MetaTrader dependency - Works on Linux VPS
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import time
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from io import BytesIO
import logging
from logging.handlers import RotatingFileHandler
import os
from PIL import Image
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import json

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None


# ======================== LOGGING SETUP ========================
def setup_logger():
    """راه‌اندازی سیستم لاگ‌گیری"""
    if not os.path.exists('logs'):
        os.makedirs('logs')

    logger = logging.getLogger('ICTBot')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handlers
    file_handler = RotatingFileHandler(
        'logs/bot_full.log',
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()


# ======================== DATA PROVIDER ABSTRACTION ========================
class DataProvider(ABC):
    """Abstract base class for data providers"""

    @abstractmethod
    def get_candles(self, symbol: str, timeframe: str, count: int) -> Optional[pd.DataFrame]:
        """Get candles from data source"""
        pass

    @abstractmethod
    def place_order(self, symbol: str, side: str, amount: float, entry: float, sl: float, tp: float) -> Optional[Dict]:
        """Place order"""
        pass


class OANDAProvider(DataProvider):
    """OANDA API Provider for Forex trading"""

    def __init__(self, api_key: str, account_id: str, practice: bool = True):
        self.api_key = api_key
        self.account_id = account_id
        self.base_url = "https://api-fxpractice.oanda.com" if practice else "https://api-fxtrade.oanda.com"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        logger.info(f"OANDA Provider initialized (Practice: {practice})")

    def get_candles(self, symbol: str, timeframe: str, count: int) -> Optional[pd.DataFrame]:
        """Get candles from OANDA"""
        try:
            # Map timeframe to OANDA format
            tf_map = {
                'M1': 'M1',
                'M5': 'M5',
                'M15': 'M15',
                'H1': 'H1',
                'H4': 'H4',
                'D': 'D'
            }

            granularity = tf_map.get(timeframe, 'M1')

            url = f"{self.base_url}/v3/instruments/{symbol}/candles"
            params = {
                "count": count,
                "granularity": granularity,
                "price": "MBA"  # Mid, Bid, Ask
            }

            response = requests.get(url, headers=self.headers, params=params)

            if response.status_code == 200:
                data = response.json()
                candles = []

                for candle in data['candles']:
                    if candle['complete']:
                        candles.append({
                            'time': pd.to_datetime(candle['time']),
                            'open': float(candle['mid']['o']),
                            'high': float(candle['mid']['h']),
                            'low': float(candle['mid']['l']),
                            'close': float(candle['mid']['c']),
                            'volume': int(candle['volume'])
                        })

                df = pd.DataFrame(candles)
                logger.debug(f"Retrieved {len(df)} candles for {symbol} {timeframe}")
                return df
            else:
                logger.error(f"Failed to get candles: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error getting candles from OANDA: {e}", exc_info=True)
            return None

    def place_order(self, symbol: str, side: str, amount: float, entry: float, sl: float, tp: float) -> Optional[Dict]:
        """Place order on OANDA"""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/orders"

            order_data = {
                "order": {
                    "type": "MARKET",
                    "instrument": symbol,
                    "units": str(int(amount)) if side == "BUY" else str(-int(amount)),
                    "stopLossOnFill": {
                        "price": str(sl)
                    },
                    "takeProfitOnFill": {
                        "price": str(tp)
                    }
                }
            }

            response = requests.post(url, headers=self.headers, json=order_data)

            if response.status_code == 201:
                result = response.json()
                logger.info(f"Order placed successfully: {result}")
                return result
            else:
                logger.error(f"Failed to place order: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error placing order: {e}", exc_info=True)
            return None


class SimulatedProvider(DataProvider):
    """Simulated provider using historical data from CSV or Yahoo Finance"""

    def __init__(self, data_source: str = "yfinance"):
        self.data_source = data_source
        self.cache = {}
        logger.info(f"Simulated Provider initialized with {data_source}")

    def get_candles(self, symbol: str, timeframe: str, count: int) -> Optional[pd.DataFrame]:
        """Get simulated candles"""
        try:
            if self.data_source == "yfinance":
                import yfinance as yf

                # Map timeframe
                tf_map = {
                    'M1': '1m',
                    'M5': '5m',
                    'M15': '15m',
                    'H1': '1h',
                    'H4': '4h',
                    'D': '1d'
                }

                interval = tf_map.get(timeframe, '5m')
                period_days = max(7, count // 78)  # Approximate days needed

                ticker = yf.Ticker(symbol)
                df = ticker.history(period=f"{period_days}d", interval=interval)

                if df.empty:
                    logger.warning(f"No data received for {symbol}")
                    return None

                # Rename columns to match our format
                df = df.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })

                df = df[['open', 'high', 'low', 'close', 'volume']].tail(count).reset_index()
                df = df.rename(columns={'index': 'time'})

                logger.debug(f"Retrieved {len(df)} candles for {symbol} {timeframe}")
                return df
            else:
                logger.error(f"Unsupported data source: {self.data_source}")
                return None

        except Exception as e:
            logger.error(f"Error getting simulated candles: {e}", exc_info=True)
            return None

    def place_order(self, symbol: str, side: str, amount: float, entry: float, sl: float, tp: float) -> Optional[Dict]:
        """Simulated order placement"""
        order_id = f"SIM_{int(time.time() * 1000)}"
        logger.info(f"[SIMULATED] Order placed: {side} {amount} {symbol} @ {entry}, SL: {sl}, TP: {tp}")
        return {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "status": "simulated"
        }


# ======================== TELEGRAM NOTIFIER ========================
class TelegramNotifier:
    """Telegram notification handler"""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        logger.info(f"Telegram Notifier initialized for chat_id: {chat_id}")

    def send_message(self, text: str) -> bool:
        """Send text message"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data, timeout=10)
            result = response.json()

            if result.get('ok'):
                logger.debug("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Telegram send failed: {result}")
                return False

        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}", exc_info=True)
            return False

    def send_photo(self, photo_bytes: BytesIO, caption: str = "") -> bool:
        """Send photo"""
        try:
            photo_bytes.seek(0)
            img = Image.open(photo_bytes)
            width, height = img.size

            # Resize if needed
            max_dimension = 2560
            if width > max_dimension or height > max_dimension:
                ratio = min(max_dimension / width, max_dimension / height)
                new_width = int(width * ratio * 0.95)
                new_height = int(height * ratio * 0.95)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                output = BytesIO()
                img.save(output, format='PNG', optimize=True, quality=95)
                output.seek(0)
                photo_bytes = output
            else:
                photo_bytes.seek(0)

            url = f"{self.base_url}/sendPhoto"
            files = {'photo': ('chart.png', photo_bytes, 'image/png')}
            data = {
                "chat_id": self.chat_id,
                "caption": caption[:1024] if len(caption) > 1024 else caption,
                "parse_mode": "HTML"
            }

            response = requests.post(url, data=data, files=files, timeout=30)
            result = response.json()

            if result.get('ok'):
                logger.debug("Telegram photo sent successfully")
                return True
            else:
                logger.error(f"Telegram photo send failed: {result}")
                return False

        except Exception as e:
            logger.error(f"Error sending Telegram photo: {e}", exc_info=True)
            return False


# ======================== CHART GENERATOR ========================
class ChartGenerator:
    """Generate trading charts with annotations"""

    @staticmethod
    def create_chart(df: pd.DataFrame, title: str, swings: Optional[Dict] = None,
                     fvgs: Optional[Dict] = None, order_blocks: Optional[List] = None,
                     mss_index: Optional[int] = None, liquidity_swept: Optional[Dict] = None,
                     entry_zone: Optional[Dict] = None) -> Optional[BytesIO]:
        """Create annotated chart"""
        try:
            logger.debug(f"Creating chart: {title}")

            # Limit candles
            max_candles = 100
            if len(df) > max_candles:
                df = df.iloc[-max_candles:].copy()

            df_plot = df.reset_index(drop=True).copy()

            # Create figure
            fig, ax = plt.subplots(figsize=(14, 7), dpi=100)

            # Draw candles
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
                    rect = Rectangle((idx - 0.4, bottom), 0.8, height,
                                    facecolor=color, edgecolor=color, alpha=0.9, linewidth=1.5)
                    ax.add_patch(rect)
                else:
                    ax.plot([idx - 0.4, idx + 0.4], [o, o], color=color, linewidth=2)

                # Wick
                ax.plot([idx, idx], [l, h], color=color, linewidth=1.2, alpha=0.7)

            # Configure axes
            ax.set_xlim(-1, len(df_plot))
            y_margin = (df_plot['high'].max() - df_plot['low'].min()) * 0.05
            ax.set_ylim(df_plot['low'].min() - y_margin, df_plot['high'].max() + y_margin)
            ax.set_title(title, fontsize=16, weight='bold', pad=20)
            ax.set_xlabel('Candles', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_facecolor('#f5f5f5')

            offset = len(df) - len(df_plot)

            # Draw Swings
            if swings:
                if 'highs' in swings and swings['highs']:
                    for swing in swings['highs'][-5:]:
                        adj_idx = swing['index'] - offset
                        if 0 <= adj_idx < len(df_plot):
                            ax.plot(adj_idx, swing['price'], 'rv', markersize=10, zorder=5)
                            ax.text(adj_idx, swing['price'], ' BSL', fontsize=9,
                                   color='darkred', va='bottom', weight='bold')

                if 'lows' in swings and swings['lows']:
                    for swing in swings['lows'][-5:]:
                        adj_idx = swing['index'] - offset
                        if 0 <= adj_idx < len(df_plot):
                            ax.plot(adj_idx, swing['price'], 'g^', markersize=10, zorder=5)
                            ax.text(adj_idx, swing['price'], ' SSL', fontsize=9,
                                   color='darkgreen', va='top', weight='bold')

            # Draw FVGs
            if fvgs:
                if 'bullish' in fvgs and fvgs['bullish']:
                    for fvg in fvgs['bullish'][-3:]:
                        adj_idx = fvg['index'] - offset
                        if 0 <= adj_idx < len(df_plot):
                            start_idx = max(0, adj_idx - 5)
                            end_idx = min(len(df_plot), adj_idx + 10)

                            rect = Rectangle(
                                (start_idx, fvg['bottom']),
                                end_idx - start_idx,
                                fvg['top'] - fvg['bottom'],
                                linewidth=1.5, edgecolor='green',
                                facecolor='green', alpha=0.2, zorder=2
                            )
                            ax.add_patch(rect)
                            ax.text(adj_idx, fvg['mid'], 'FVG↑', fontsize=10,
                                   color='darkgreen', weight='bold', zorder=6)

                if 'bearish' in fvgs and fvgs['bearish']:
                    for fvg in fvgs['bearish'][-3:]:
                        adj_idx = fvg['index'] - offset
                        if 0 <= adj_idx < len(df_plot):
                            start_idx = max(0, adj_idx - 5)
                            end_idx = min(len(df_plot), adj_idx + 10)

                            rect = Rectangle(
                                (start_idx, fvg['bottom']),
                                end_idx - start_idx,
                                fvg['top'] - fvg['bottom'],
                                linewidth=1.5, edgecolor='red',
                                facecolor='red', alpha=0.2, zorder=2
                            )
                            ax.add_patch(rect)
                            ax.text(adj_idx, fvg['mid'], 'FVG↓', fontsize=10,
                                   color='darkred', weight='bold', zorder=6)

            # Draw Order Blocks
            if order_blocks:
                for ob in order_blocks:
                    if ob:
                        adj_idx = ob['index'] - offset
                        if 0 <= adj_idx < len(df_plot):
                            start_idx = max(0, adj_idx - 3)
                            end_idx = min(len(df_plot), adj_idx + 15)

                            color = 'blue' if 'bullish' in ob['type'] else 'orange'

                            rect = Rectangle(
                                (start_idx, ob['low']),
                                end_idx - start_idx,
                                ob['high'] - ob['low'],
                                linewidth=2, edgecolor=color,
                                facecolor=color, alpha=0.25, zorder=2
                            )
                            ax.add_patch(rect)
                            ax.text(adj_idx, ob['mid'], 'OB', fontsize=11,
                                   color=color, weight='bold', zorder=6)

            # Mark MSS
            if mss_index is not None:
                adj_mss = mss_index - offset
                if 0 <= adj_mss < len(df_plot):
                    ax.axvline(adj_mss, color='purple', linestyle='--',
                              linewidth=3, alpha=0.7, zorder=4)
                    y_pos = ax.get_ylim()[1] * 0.98
                    ax.text(adj_mss, y_pos, ' MSS ✓', fontsize=12,
                           color='purple', weight='bold', va='top', zorder=6,
                           bbox=dict(boxstyle='round', facecolor='white',
                                   alpha=0.9, edgecolor='purple', linewidth=2))

            # Mark Liquidity Swept
            if liquidity_swept:
                ax.axhline(liquidity_swept['price'], color='gold',
                          linestyle=':', linewidth=3, alpha=0.8, zorder=3)
                ax.text(len(df_plot) - 1, liquidity_swept['price'],
                       f" {liquidity_swept['type']} ", fontsize=10,
                       color='black', weight='bold', zorder=6,
                       bbox=dict(boxstyle='round', facecolor='gold',
                               alpha=0.9, edgecolor='darkorange', linewidth=2))

            # Mark Entry Zone
            if entry_zone:
                top = entry_zone.get('top', entry_zone.get('high'))
                bottom = entry_zone.get('bottom', entry_zone.get('low'))

                ax.axhspan(bottom, top, color='cyan', alpha=0.25, zorder=1)
                ax.text(len(df_plot) - 2, (top + bottom) / 2,
                       ' ENTRY ', fontsize=11, color='black',
                       weight='bold', va='center', zorder=6,
                       bbox=dict(boxstyle='round', facecolor='cyan',
                               alpha=0.9, edgecolor='blue', linewidth=2))

            plt.tight_layout()

            # Save to buffer
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            plt.close(fig)

            logger.debug(f"Chart created successfully")
            return buf

        except Exception as e:
            logger.error(f"Error creating chart: {e}", exc_info=True)
            return None


# Continue to Part 2...
# ======================== ICT STRATEGY IMPLEMENTATION ========================
class ICTStrategy:
    """Implementation of ICT Trading Strategy from PDF"""

    def __init__(self, symbol: str, timezone_str: str = "America/New_York"):
        self.symbol = symbol
        self.timezone = pytz.timezone(timezone_str)
        logger.info(f"ICT Strategy initialized for {symbol}")

    def identify_swing_high_low(self, df: pd.DataFrame) -> Dict:
        """
        Identify Swing High/Low using 3-candle fractal
        PDF Page 3: Fractal سه گانه
        """
        swings = {'highs': [], 'lows': []}

        for i in range(2, len(df) - 2):
            # Swing High (BSL)
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and
                df['high'].iloc[i] > df['high'].iloc[i-2] and
                df['high'].iloc[i] > df['high'].iloc[i+1] and
                df['high'].iloc[i] > df['high'].iloc[i+2]):

                swings['highs'].append({
                    'index': i,
                    'price': df['high'].iloc[i],
                    'time': df['time'].iloc[i] if 'time' in df.columns else i,
                    'type': 'BSL'
                })

            # Swing Low (SSL)
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and
                df['low'].iloc[i] < df['low'].iloc[i-2] and
                df['low'].iloc[i] < df['low'].iloc[i+1] and
                df['low'].iloc[i] < df['low'].iloc[i+2]):

                swings['lows'].append({
                    'index': i,
                    'price': df['low'].iloc[i],
                    'time': df['time'].iloc[i] if 'time' in df.columns else i,
                    'type': 'SSL'
                })

        # Filter: Remove BSL if higher BSL exists later
        filtered_highs = []
        for i, swing in enumerate(swings['highs']):
            is_valid = True
            for j in range(i + 1, len(swings['highs'])):
                if swings['highs'][j]['price'] > swing['price']:
                    is_valid = False
                    break
            if is_valid:
                filtered_highs.append(swing)

        # Filter: Remove SSL if lower SSL exists later
        filtered_lows = []
        for i, swing in enumerate(swings['lows']):
            is_valid = True
            for j in range(i + 1, len(swings['lows'])):
                if swings['lows'][j]['price'] < swing['price']:
                    is_valid = False
                    break
            if is_valid:
                filtered_lows.append(swing)

        swings['highs'] = filtered_highs
        swings['lows'] = filtered_lows

        logger.debug(f"Identified {len(filtered_highs)} BSL and {len(filtered_lows)} SSL")
        return swings

    def check_liquidity_sweep(self, df: pd.DataFrame, liquidity_price: float,
                              is_high: bool = True) -> Tuple[bool, int]:
        """
        Check if liquidity has been swept
        PDF Page 7: جستجوی SSL/BSL Sweep
        """
        for i in range(max(0, len(df) - 10), len(df)):
            if is_high:
                if df['high'].iloc[i] >= liquidity_price:
                    logger.debug(f"BSL swept at index {i}, price {liquidity_price}")
                    return True, i
            else:
                if df['low'].iloc[i] <= liquidity_price:
                    logger.debug(f"SSL swept at index {i}, price {liquidity_price}")
                    return True, i
        return False, -1

    def identify_fvg(self, df: pd.DataFrame, min_gap_pips: float = 0.0001) -> Dict:
        """
        Identify Fair Value Gaps
        PDF Page 4: تشخیص FVG
        Improved: Only significant FVGs
        """
        fvgs = {'bullish': [], 'bearish': []}

        for i in range(2, len(df)):
            # Bullish FVG: Gap between candle i-2 high and candle i low
            gap_size = df['low'].iloc[i] - df['high'].iloc[i-2]
            if gap_size > min_gap_pips:  # Minimum gap filter
                fvgs['bullish'].append({
                    'index': i,
                    'top': df['low'].iloc[i],
                    'bottom': df['high'].iloc[i-2],
                    'mid': (df['low'].iloc[i] + df['high'].iloc[i-2]) / 2,
                    'time': df['time'].iloc[i] if 'time' in df.columns else i,
                    'size': gap_size
                })

            # Bearish FVG: Gap between candle i-2 low and candle i high
            gap_size = df['low'].iloc[i-2] - df['high'].iloc[i]
            if gap_size > min_gap_pips:
                fvgs['bearish'].append({
                    'index': i,
                    'top': df['low'].iloc[i-2],
                    'bottom': df['high'].iloc[i],
                    'mid': (df['low'].iloc[i-2] + df['high'].iloc[i]) / 2,
                    'time': df['time'].iloc[i] if 'time' in df.columns else i,
                    'size': gap_size
                })

        logger.debug(f"Found {len(fvgs['bullish'])} bullish FVGs and {len(fvgs['bearish'])} bearish FVGs")
        return fvgs

    def check_displacement(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
        """
        Check for Displacement
        PDF Page 9: Consecutive Candles or Full Body Candles
        """
        if end_idx <= start_idx or end_idx - start_idx < 2:
            return False

        # Check for consecutive same-color candles
        consecutive_count = 0
        first_bullish = df['close'].iloc[start_idx] > df['open'].iloc[start_idx]

        for i in range(start_idx, min(end_idx, len(df))):
            is_bullish = df['close'].iloc[i] > df['open'].iloc[i]
            if is_bullish == first_bullish:
                consecutive_count += 1
            else:
                break

        if consecutive_count >= 3:
            logger.debug(f"Displacement confirmed: {consecutive_count} consecutive candles")
            return True

        # Check for full body candles
        full_body_count = 0
        for i in range(start_idx, min(end_idx, len(df))):
            body = abs(df['close'].iloc[i] - df['open'].iloc[i])
            candle_range = df['high'].iloc[i] - df['low'].iloc[i]
            if candle_range > 0 and (body / candle_range) >= 0.7:
                full_body_count += 1

        if full_body_count >= 2:
            logger.debug(f"Displacement confirmed: {full_body_count} full body candles")
            return True

        return False

    def find_intermediate_swing(self, df: pd.DataFrame, swings: Dict,
                                direction: str = 'bullish') -> Tuple[Optional[Dict], Optional[str]]:
        """
        Find Intermediate Swing
        PDF Page 8-9: شناسایی Intermediate Swing
        """
        if direction == 'bullish' and len(swings['lows']) >= 2:
            last_low = swings['lows'][-1]

            # Check if inside FVG
            fvgs = self.identify_fvg(df)
            for fvg in fvgs['bullish']:
                if (fvg['bottom'] <= last_low['price'] <= fvg['top'] and
                    fvg['index'] <= last_low['index']):
                    logger.info(f"Intermediate Swing found in FVG at {last_low['price']}")
                    return last_low, 'fvg'

            # Check if short term swing
            if len(swings['lows']) >= 3:
                prev_low = swings['lows'][-2]
                if prev_low['price'] > last_low['price']:
                    logger.info(f"Intermediate Swing found (Short Term) at {last_low['price']}")
                    return last_low, 'short_term'

        elif direction == 'bearish' and len(swings['highs']) >= 2:
            last_high = swings['highs'][-1]

            fvgs = self.identify_fvg(df)
            for fvg in fvgs['bearish']:
                if (fvg['bottom'] <= last_high['price'] <= fvg['top'] and
                    fvg['index'] <= last_high['index']):
                    logger.info(f"Intermediate Swing found in FVG at {last_high['price']}")
                    return last_high, 'fvg'

            if len(swings['highs']) >= 3:
                prev_high = swings['highs'][-2]
                if prev_high['price'] < last_high['price']:
                    logger.info(f"Intermediate Swing found (Short Term) at {last_high['price']}")
                    return last_high, 'short_term'

        return None, None

    def detect_mss(self, df: pd.DataFrame, intermediate_swing: Dict,
                   direction: str = 'bullish') -> Tuple[bool, int]:
        """
        Detect Market Structure Shift (MSS)
        PDF Page 8: تشخیص MSS
        """
        if intermediate_swing is None:
            return False, -1

        swing_price = intermediate_swing['price']
        swing_idx = intermediate_swing['index']

        for i in range(swing_idx, len(df)):
            if direction == 'bullish':
                if df['high'].iloc[i] > swing_price:
                    if self.check_displacement(df, swing_idx, i):
                        logger.info(f"MSS confirmed (Bullish) at index {i}")
                        return True, i
            else:
                if df['low'].iloc[i] < swing_price:
                    if self.check_displacement(df, swing_idx, i):
                        logger.info(f"MSS confirmed (Bearish) at index {i}")
                        return True, i

        return False, -1

    def identify_order_block(self, df: pd.DataFrame, mss_index: int,
                             direction: str = 'bullish') -> Optional[Dict]:
        """
        Identify Order Block
        PDF Page 10: شناسایی Order Block
        """
        if mss_index < 3:
            return None

        if direction == 'bullish':
            # Look for last bearish candle before MSS
            for i in range(mss_index - 1, max(0, mss_index - 10), -1):
                if df['close'].iloc[i] < df['open'].iloc[i]:
                    ob = {
                        'type': 'bullish_ob',
                        'high': df['high'].iloc[i],
                        'low': df['low'].iloc[i],
                        'open': df['open'].iloc[i],
                        'mid': (df['high'].iloc[i] + df['low'].iloc[i]) / 2,
                        'index': i
                    }
                    logger.debug(f"Order Block found (Bullish) at index {i}")
                    return ob
        else:
            # Look for last bullish candle before MSS
            for i in range(mss_index - 1, max(0, mss_index - 10), -1):
                if df['close'].iloc[i] > df['open'].iloc[i]:
                    ob = {
                        'type': 'bearish_ob',
                        'high': df['high'].iloc[i],
                        'low': df['low'].iloc[i],
                        'open': df['open'].iloc[i],
                        'mid': (df['high'].iloc[i] + df['low'].iloc[i]) / 2,
                        'index': i
                    }
                    logger.debug(f"Order Block found (Bearish) at index {i}")
                    return ob

        return None

    def is_trading_session(self, current_time: datetime = None) -> Tuple[bool, int]:
        """
        Check if within trading sessions
        PDF Page 19: تایم‌های معاملاتی
        London: 2-5, New York: 7-11, PM: 14-15 (UTC-4)
        """
        if current_time is None:
            current_time = datetime.now(self.timezone)

        hour = current_time.hour

        trading_sessions = [(2, 5), (7, 11), (14, 15)]

        for start, end in trading_sessions:
            if start <= hour < end:
                return True, hour

        return False, hour

    def get_daily_open(self, df_daily: pd.DataFrame) -> Optional[float]:
        """
        Get daily open price
        PDF Page 6: Daily Open برای تعیین Bias
        """
        if df_daily is not None and len(df_daily) > 0:
            daily_open = df_daily['open'].iloc[-1]
            logger.debug(f"Daily Open: {daily_open}")
            return daily_open
        return None


# Continue to Part 3 (Main Bot Class)...
# ======================== MAIN ICT TRADING BOT ========================
class ICTTradingBotV2:
    """
    Main ICT Trading Bot - Version 2.0
    No MetaTrader dependency - Works on Linux VPS
    """

    def __init__(self, symbol: str, lot_size: float,
                 data_provider: DataProvider,
                 telegram_token: Optional[str] = None,
                 telegram_chat_id: Optional[str] = None):

        self.symbol = symbol
        self.lot_size = lot_size
        self.data_provider = data_provider
        self.strategy = ICTStrategy(symbol)
        self.chart_gen = ChartGenerator()

        # Telegram setup
        if telegram_token and telegram_chat_id:
            self.telegram = TelegramNotifier(telegram_token, telegram_chat_id)
            self.send_telegram = True
            logger.info("Telegram notifications enabled")
        else:
            self.telegram = None
            self.send_telegram = False
            logger.warning("Telegram notifications disabled")

        self.positions = []
        logger.info(f"ICT Trading Bot V2 initialized for {symbol}")

    def get_candles(self, timeframe: str, count: int) -> Optional[pd.DataFrame]:
        """Get candles from data provider"""
        return self.data_provider.get_candles(self.symbol, timeframe, count)

    def run_strategy(self):
        """
        Main strategy execution loop
        Based on PDF Page 19-20: Complete trading flow
        """
        logger.info("=" * 60)
        logger.info("ICT Trading Bot V2 Started")
        logger.info(f"Symbol: {self.symbol} | Lot Size: {self.lot_size}")
        logger.info("=" * 60)

        if self.send_telegram:
            self.telegram.send_message(
                "🤖 <b>ICT Trading Bot V2 Started</b>\n\n"
                f"📊 Symbol: <code>{self.symbol}</code>\n"
                f"💰 Lot Size: <code>{self.lot_size}</code>\n"
                f"🕐 Time: {datetime.now(self.strategy.timezone).strftime('%Y-%m-%d %H:%M:%S')}"
            )

        while True:
            try:
                # Check trading session
                is_trading, current_hour = self.strategy.is_trading_session()
                if not is_trading:
                    logger.debug(f"Outside trading session (Hour: {current_hour})")
                    time.sleep(60)
                    continue

                logger.info(f"\n{'='*60}")
                logger.info(f"Cycle Start: {datetime.now(self.strategy.timezone).strftime('%Y-%m-%d %H:%M:%S')}")

                # ============ STEP 1: M5 Timeframe - Liquidity Sweep ============
                logger.info("STEP 1: Checking M5 for Liquidity Sweep...")

                df_m5 = self.get_candles('M5', 200)
                if df_m5 is None:
                    logger.warning("Failed to get M5 data")
                    time.sleep(30)
                    continue

                swings_m5 = self.strategy.identify_swing_high_low(df_m5)
                current_price = df_m5['close'].iloc[-1]

                logger.info(f"Current Price: {current_price:.5f}")

                # Get daily open for bias
                df_daily = self.get_candles('D', 5)
                daily_open = self.strategy.get_daily_open(df_daily)

                if daily_open:
                    logger.info(f"Daily Open: {daily_open:.5f}")
                    bias = "SELL" if current_price > daily_open else "BUY"
                    logger.info(f"Bias: {bias} (Price {'above' if bias == 'SELL' else 'below'} Daily Open)")
                else:
                    logger.warning("Could not determine daily open, using price action")
                    bias = None

                liquidity_swept = None
                direction = None

                # Search for SSL Sweep (for Buy setup)
                if daily_open is None or current_price < daily_open:
                    logger.info("Searching for SSL Sweep (for Buy setup)...")
                    if len(swings_m5['lows']) > 0:
                        for swing in reversed(swings_m5['lows'][-3:]):
                            swept, idx = self.strategy.check_liquidity_sweep(df_m5, swing['price'], is_high=False)
                            if swept:
                                liquidity_swept = swing
                                direction = 'bullish'
                                logger.info(f"✅ SSL Swept at {swing['price']:.5f}")
                                break

                # Search for BSL Sweep (for Sell setup)
                elif daily_open and current_price > daily_open:
                    logger.info("Searching for BSL Sweep (for Sell setup)...")
                    if len(swings_m5['highs']) > 0:
                        for swing in reversed(swings_m5['highs'][-3:]):
                            swept, idx = self.strategy.check_liquidity_sweep(df_m5, swing['price'], is_high=True)
                            if swept:
                                liquidity_swept = swing
                                direction = 'bearish'
                                logger.info(f"✅ BSL Swept at {swing['price']:.5f}")
                                break

                if liquidity_swept is None:
                    logger.info("No liquidity sweep detected yet. Waiting...")
                    time.sleep(30)
                    continue

                # ============ STEP 2: M1 Timeframe - MSS Confirmation ============
                logger.info("STEP 2: Moving to M1 for MSS confirmation...")

                df_m1 = self.get_candles('M1', 500)
                if df_m1 is None:
                    logger.warning("Failed to get M1 data")
                    time.sleep(30)
                    continue

                swings_m1 = self.strategy.identify_swing_high_low(df_m1)

                # Find Intermediate Swing
                intermediate_swing, swing_type = self.strategy.find_intermediate_swing(
                    df_m1, swings_m1, direction
                )

                if intermediate_swing is None:
                    logger.info(f"Waiting for Intermediate Swing ({direction})...")
                    time.sleep(30)
                    continue

                logger.info(f"Intermediate Swing found: {swing_type} at {intermediate_swing['price']:.5f}")

                # Confirm MSS
                mss_confirmed, mss_idx = self.strategy.detect_mss(df_m1, intermediate_swing, direction)

                if not mss_confirmed:
                    logger.info("MSS not confirmed yet. Waiting...")
                    time.sleep(30)
                    continue

                logger.info("✅ MSS Confirmed!")

                # ============ STEP 3: Find Entry Zone ============
                logger.info("STEP 3: Searching for Entry Zone...")

                fvgs_m1 = self.strategy.identify_fvg(df_m1)
                entry_zone = None
                ob = None

                # Priority 1: FVG
                if direction == 'bullish' and len(fvgs_m1['bullish']) > 0:
                    for fvg in reversed(fvgs_m1['bullish']):
                        if fvg['index'] >= mss_idx:
                            entry_zone = fvg
                            entry_zone['entry_type'] = 'FVG'
                            logger.info(f"Entry Zone: Bullish FVG ({fvg['bottom']:.5f} - {fvg['top']:.5f})")
                            break

                elif direction == 'bearish' and len(fvgs_m1['bearish']) > 0:
                    for fvg in reversed(fvgs_m1['bearish']):
                        if fvg['index'] >= mss_idx:
                            entry_zone = fvg
                            entry_zone['entry_type'] = 'FVG'
                            logger.info(f"Entry Zone: Bearish FVG ({fvg['bottom']:.5f} - {fvg['top']:.5f})")
                            break

                # Priority 2: Order Block
                if entry_zone is None:
                    ob = self.strategy.identify_order_block(df_m1, mss_idx, direction)
                    if ob:
                        entry_zone = ob
                        entry_zone['entry_type'] = 'OB'
                        logger.info(f"Entry Zone: Order Block ({ob['low']:.5f} - {ob['high']:.5f})")

                if entry_zone is None:
                    logger.info("No Entry Zone found. Waiting...")
                    time.sleep(30)
                    continue

                # ============ STEP 4: Calculate Trade Parameters ============
                logger.info("STEP 4: Preparing Trade Signal...")

                if direction == 'bullish':
                    entry = entry_zone.get('bottom', entry_zone.get('low'))
                    zone_height = entry_zone.get('top', entry_zone.get('high')) - entry
                    sl = entry - (zone_height * 0.2)
                    risk = entry - sl
                    tp = entry + (risk * 3)
                    side = "BUY"
                    emoji = "📈"
                else:
                    entry = entry_zone.get('top', entry_zone.get('high'))
                    zone_height = entry - entry_zone.get('bottom', entry_zone.get('low'))
                    sl = entry + (zone_height * 0.2)
                    risk = sl - entry
                    tp = entry - (risk * 3)
                    side = "SELL"
                    emoji = "📉"

                # Log trade signal
                logger.info("="*60)
                logger.info(f"TRADE SIGNAL GENERATED: {side}")
                logger.info(f"Symbol: {self.symbol}")
                logger.info(f"Liquidity Swept: {liquidity_swept['type']} at {liquidity_swept['price']:.5f}")
                logger.info(f"Entry Type: {entry_zone['entry_type']}")
                logger.info(f"Entry: {entry:.5f}")
                logger.info(f"Stop Loss: {sl:.5f}")
                logger.info(f"Take Profit: {tp:.5f}")
                logger.info(f"Risk: {risk:.5f} | Reward: {risk*3:.5f} | R:R = 1:3")
                logger.info("="*60)

                # ============ STEP 5: Send to Telegram ============
                if self.send_telegram:
                    try:
                        # Prepare message
                        message = (
                            f"{emoji} <b>═══ TRADE SIGNAL {side} ═══</b>\n\n"
                            f"📊 <b>Base Info:</b>\n"
                            f"├ Symbol: <code>{self.symbol}</code>\n"
                            f"├ ⏰ Time: {datetime.now(self.strategy.timezone).strftime('%H:%M:%S')}\n"
                            f"├ 📅 Date: {datetime.now(self.strategy.timezone).strftime('%Y-%m-%d')}\n"
                            f"└ 🎯 Direction: <b>{direction.upper()}</b>\n\n"

                            f"💰 <b>Key Prices:</b>\n"
                            f"├ Daily Open: <code>{daily_open:.5f if daily_open else 'N/A'}</code>\n"
                            f"├ Current: <code>{current_price:.5f}</code>\n"
                            f"└ Position: {'🔴 Above' if daily_open and current_price > daily_open else '🟢 Below'} Open\n\n"

                            f"⚡ <b>Liquidity Swept:</b>\n"
                            f"├ Type: <b>{liquidity_swept['type']}</b> ✅\n"
                            f"└ Price: <code>{liquidity_swept['price']:.5f}</code>\n\n"

                            f"✅ <b>MSS Confirmed</b> at Index {mss_idx}\n\n"

                            f"🎯 <b>Entry Zone:</b>\n"
                            f"├ Type: <b>{entry_zone['entry_type']}</b>\n"
                            f"├ Top: <code>{entry_zone.get('top', entry_zone.get('high')):.5f}</code>\n"
                            f"└ Bottom: <code>{entry_zone.get('bottom', entry_zone.get('low')):.5f}</code>\n\n"

                            f"💰 <b>═══ TRADE DETAILS ═══</b>\n"
                            f"├ 📍 Entry: <code>{entry:.5f}</code>\n"
                            f"├ 🛑 Stop Loss: <code>{sl:.5f}</code>\n"
                            f"├ 🎯 Take Profit: <code>{tp:.5f}</code>\n"
                            f"├ ⚠️ Risk: <code>{abs(risk):.5f}</code>\n"
                            f"├ 💎 Reward: <code>{abs(risk*3):.5f}</code>\n"
                            f"├ 📊 R:R: <b>1:3</b> 🎯\n"
                            f"└ 💰 Lot Size: <code>{self.lot_size}</code>\n\n"

                            f"{'🟢' if direction == 'bullish' else '🔴'} <b>Setup Ready!</b>"
                        )

                        self.telegram.send_message(message)

                        # Send M5 chart
                        logger.info("Generating M5 chart...")
                        chart_m5 = self.chart_gen.create_chart(
                            df_m5.tail(80),
                            f"{self.symbol} - M5 Timeframe",
                            swings=swings_m5,
                            fvgs=self.strategy.identify_fvg(df_m5.tail(80)),
                            liquidity_swept=liquidity_swept
                        )

                        if chart_m5:
                            self.telegram.send_photo(
                                chart_m5,
                                f"📊 <b>M5 Chart</b>\n🎯 {liquidity_swept['type']} Swept"
                            )

                        # Send M1 chart
                        logger.info("Generating M1 chart...")
                        chart_m1 = self.chart_gen.create_chart(
                            df_m1.tail(100),
                            f"{self.symbol} - M1 Timeframe",
                            swings=swings_m1,
                            fvgs=fvgs_m1,
                            order_blocks=[ob] if ob else [],
                            mss_index=mss_idx if mss_idx >= len(df_m1) - 100 else None,
                            entry_zone=entry_zone
                        )

                        if chart_m1:
                            self.telegram.send_photo(
                                chart_m1,
                                f"📊 <b>M1 Chart</b>\n✅ MSS + Entry Zone"
                            )

                    except Exception as e:
                        logger.error(f"Error sending to Telegram: {e}", exc_info=True)

                # ============ STEP 6: Place Order (Optional) ============
                # Uncomment to enable real trading
                # result = self.data_provider.place_order(self.symbol, side, self.lot_size, entry, sl, tp)
                # if result:
                #     logger.info(f"Order placed: {result}")

                logger.info("Waiting 5 minutes before next check...")
                logger.info(f"{'='*60}\n")
                time.sleep(300)

            except KeyboardInterrupt:
                logger.info("Bot stopped by user (Ctrl+C)")
                raise

            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
                if self.send_telegram:
                    self.telegram.send_message(
                        f"⚠️ <b>Error in Bot</b>\n\n"
                        f"<code>{str(e)}</code>\n\n"
                        f"Bot attempting to recover..."
                    )
                time.sleep(30)

    def shutdown(self):
        """Shutdown bot"""
        logger.info("Shutting down bot...")
        if self.send_telegram:
            self.telegram.send_message(
                "🔴 <b>Bot Stopped</b>\n\n"
                f"⏰ Time: {datetime.now(self.strategy.timezone).strftime('%Y-%m-%d %H:%M:%S')}"
            )


# ======================== CONFIGURATION & MAIN ========================
def load_config(config_file: str = "config.json") -> Dict:
    """Load configuration from JSON file"""
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        logger.warning(f"Config file {config_file} not found, using defaults")
        return {}


def main():
    """Main entry point"""
    # Load configuration
    config = load_config()

    # Get settings
    symbol = config.get("symbol", "EUR_USD")
    lot_size = config.get("lot_size", 1000)
    provider_type = config.get("provider", "simulated")  # "oanda" or "simulated"

    telegram_token = config.get("telegram_token")
    telegram_chat_id = config.get("telegram_chat_id")

    # Initialize data provider
    if provider_type == "oanda":
        api_key = config.get("oanda_api_key")
        account_id = config.get("oanda_account_id")
        practice = config.get("oanda_practice", True)

        if not api_key or not account_id:
            logger.error("OANDA credentials not found in config")
            return

        data_provider = OANDAProvider(api_key, account_id, practice)

    else:  # simulated
        data_provider = SimulatedProvider("yfinance")

    # Create bot
    bot = ICTTradingBotV2(
        symbol=symbol,
        lot_size=lot_size,
        data_provider=data_provider,
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id
    )

    try:
        logger.info("Starting main strategy loop...")
        bot.run_strategy()
    except KeyboardInterrupt:
        logger.info("\n⏹️ Bot stopped by user")
        bot.shutdown()
    except Exception as e:
        logger.critical(f"Critical error: {e}", exc_info=True)
        bot.shutdown()


if __name__ == "__main__":
    main()
