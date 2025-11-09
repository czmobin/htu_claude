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
