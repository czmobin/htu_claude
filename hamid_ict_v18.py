import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import time
import requests
import matplotlib
matplotlib.use('Agg')  # استفاده از backend غیر GUI
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from io import BytesIO
import logging
from logging.handlers import RotatingFileHandler
import os
from PIL import Image
import warnings

# غیرفعال کردن warnings خاص
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)

# افزایش محدودیت سایز تصویر PIL
Image.MAX_IMAGE_PIXELS = None

# ======================== LOGGING SETUP ========================
def setup_logger():
    """راه‌اندازی سیستم لاگ گیری کامل"""
    # ایجاد پوشه logs اگر وجود نداشت
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # تنظیمات Logger اصلی
    logger = logging.getLogger('ICTBot')
    logger.setLevel(logging.DEBUG)
    
    # فرمت لاگ
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File Handler - تمام لاگ‌ها (با Rotation)
    file_handler = RotatingFileHandler(
        'logs/bot_full.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # File Handler - فقط خطاها
    error_handler = RotatingFileHandler(
        'logs/bot_errors.log',
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    
    # File Handler - سیگنال‌های معاملاتی
    trade_handler = RotatingFileHandler(
        'logs/bot_trades.log',
        maxBytes=5*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    trade_handler.setLevel(logging.INFO)
    trade_formatter = logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    trade_handler.setFormatter(trade_formatter)
    trade_handler.addFilter(lambda record: 'TRADE' in record.getMessage())
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # اضافه کردن handlers
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(trade_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()


# ======================== TELEGRAM NOTIFIER ========================
class TelegramNotifier:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        logger.info(f"Telegram Notifier initialized for chat_id: {chat_id}")
    
    def send_message(self, text):
        """ارسال پیام متنی"""
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
                logger.debug(f"Telegram message sent successfully")
                return result
            else:
                logger.error(f"Telegram send failed: {result}")
                return None
                
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}", exc_info=True)
            return None
    
    def send_photo(self, photo_bytes, caption=""):
        """ارسال عکس"""
        try:
            # بارگذاری و بررسی نهایی تصویر
            photo_bytes.seek(0)
            img = Image.open(photo_bytes)
            width, height = img.size
            
            logger.debug(f"Preparing to send image: {width}x{height}")
            
            # تلگرام فقط ابعاد کوچکتر از 10000 را قبول می‌کند
            # و sum باید کمتر از 10000 باشد
            max_single_dimension = 2560
            max_sum = 5120
            
            if width > max_single_dimension or height > max_single_dimension or (width + height) > max_sum:
                # محاسبه ratio برای resize
                ratio = min(
                    max_single_dimension / width,
                    max_single_dimension / height,
                    max_sum / (width + height)
                )
                
                new_width = max(10, int(width * ratio * 0.95))  # 5% کمتر برای اطمینان
                new_height = max(10, int(height * ratio * 0.95))
                
                logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # ذخیره با بهینه‌سازی
                output = BytesIO()
                img.save(output, format='PNG', optimize=True, quality=95)
                output.seek(0)
                photo_bytes = output
            else:
                photo_bytes.seek(0)
            
            # ارسال به تلگرام
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
                return result
            else:
                logger.error(f"Telegram photo send failed: {result}")
                return None
                
        except Exception as e:
            logger.error(f"Error sending Telegram photo: {e}", exc_info=True)
            return None


# ======================== CHART GENERATOR ========================
class ChartGenerator:
    @staticmethod
    def create_chart(df, title, swings=None, fvgs=None, order_blocks=None, 
                     mss_index=None, liquidity_swept=None, entry_zone=None):
        """ایجاد نمودار با نشان‌گذاری‌ها"""
        try:
            logger.debug(f"Creating chart: {title}")
            
            # محدود کردن تعداد کندل‌ها برای جلوگیری از ابعاد خیلی بزرگ
            max_candles = 100
            if len(df) > max_candles:
                df = df.iloc[-max_candles:].copy()
                logger.debug(f"Limited to last {max_candles} candles")
            
            df_plot = df.reset_index(drop=True).copy()
            
            # ایجاد figure با ابعاد ثابت و مناسب برای تلگرام
            fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
            
            # رسم کندل‌ها
            for idx in range(len(df_plot)):
                open_price = df_plot['open'].iloc[idx]
                close_price = df_plot['close'].iloc[idx]
                high_price = df_plot['high'].iloc[idx]
                low_price = df_plot['low'].iloc[idx]
                
                # رنگ کندل
                color = 'green' if close_price >= open_price else 'red'
                
                # بدنه کندل
                height = abs(close_price - open_price)
                bottom = min(open_price, close_price)
                
                if height > 0:
                    rect = Rectangle((idx - 0.3, bottom), 0.6, height,
                                    facecolor=color, edgecolor=color, alpha=0.8)
                    ax.add_patch(rect)
                else:
                    # اگر open == close، یک خط افقی بکشیم
                    ax.plot([idx - 0.3, idx + 0.3], [open_price, open_price],
                           color=color, linewidth=1.5)
                
                # سایه کندل
                ax.plot([idx, idx], [low_price, high_price], 
                       color=color, linewidth=1, alpha=0.6)
            
            # تنظیمات محور
            ax.set_xlim(-1, len(df_plot))
            y_margin = (df_plot['high'].max() - df_plot['low'].min()) * 0.05
            ax.set_ylim(df_plot['low'].min() - y_margin, df_plot['high'].max() + y_margin)
            ax.set_title(title, fontsize=14, weight='bold', pad=15)
            ax.set_xlabel('Candles', fontsize=11)
            ax.set_ylabel('Price', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # اصلاح indices برای swings
            original_len = len(df)
            offset = original_len - len(df_plot)
            
            # رسم Swing Highs/Lows
            if swings:
                if 'highs' in swings and len(swings['highs']) > 0:
                    for swing in swings['highs'][-5:]:
                        adjusted_idx = swing['index'] - offset
                        if 0 <= adjusted_idx < len(df_plot):
                            ax.plot(adjusted_idx, swing['price'], 
                                   'rv', markersize=9, zorder=5)
                            ax.text(adjusted_idx, swing['price'], 
                                   ' BSL', fontsize=8, color='red', va='bottom', weight='bold')
                
                if 'lows' in swings and len(swings['lows']) > 0:
                    for swing in swings['lows'][-5:]:
                        adjusted_idx = swing['index'] - offset
                        if 0 <= adjusted_idx < len(df_plot):
                            ax.plot(adjusted_idx, swing['price'], 
                                   'g^', markersize=9, zorder=5)
                            ax.text(adjusted_idx, swing['price'], 
                                   ' SSL', fontsize=8, color='green', va='top', weight='bold')
            
            # رسم FVGs
            if fvgs:
                if 'bullish' in fvgs:
                    for fvg in fvgs['bullish'][-3:]:
                        adjusted_idx = fvg['index'] - offset
                        if 0 <= adjusted_idx < len(df_plot):
                            start_idx = max(0, adjusted_idx - 5)
                            end_idx = min(len(df_plot), adjusted_idx + 10)
                            
                            rect = Rectangle(
                                (start_idx, fvg['bottom']),
                                end_idx - start_idx,
                                fvg['top'] - fvg['bottom'],
                                linewidth=1, edgecolor='green', 
                                facecolor='green', alpha=0.2, zorder=2
                            )
                            ax.add_patch(rect)
                            ax.text(adjusted_idx, fvg['mid'], 
                                   'FVG↑', fontsize=9, color='darkgreen', weight='bold', zorder=6)
                
                if 'bearish' in fvgs:
                    for fvg in fvgs['bearish'][-3:]:
                        adjusted_idx = fvg['index'] - offset
                        if 0 <= adjusted_idx < len(df_plot):
                            start_idx = max(0, adjusted_idx - 5)
                            end_idx = min(len(df_plot), adjusted_idx + 10)
                            
                            rect = Rectangle(
                                (start_idx, fvg['bottom']),
                                end_idx - start_idx,
                                fvg['top'] - fvg['bottom'],
                                linewidth=1, edgecolor='red', 
                                facecolor='red', alpha=0.2, zorder=2
                            )
                            ax.add_patch(rect)
                            ax.text(adjusted_idx, fvg['mid'], 
                                   'FVG↓', fontsize=9, color='darkred', weight='bold', zorder=6)
            
            # رسم Order Blocks
            if order_blocks:
                for ob in order_blocks:
                    if ob:
                        adjusted_idx = ob['index'] - offset
                        if 0 <= adjusted_idx < len(df_plot):
                            start_idx = max(0, adjusted_idx - 3)
                            end_idx = min(len(df_plot), adjusted_idx + 15)
                            
                            color = 'blue' if 'bullish' in ob['type'] else 'orange'
                            
                            rect = Rectangle(
                                (start_idx, ob['low']),
                                end_idx - start_idx,
                                ob['high'] - ob['low'],
                                linewidth=2, edgecolor=color, 
                                facecolor=color, alpha=0.25, zorder=2
                            )
                            ax.add_patch(rect)
                            ax.text(adjusted_idx, ob['mid'], 
                                   'OB', fontsize=10, color=color, weight='bold', zorder=6)
            
            # نشان‌گذاری MSS
            if mss_index is not None:
                adjusted_mss = mss_index - offset
                if 0 <= adjusted_mss < len(df_plot):
                    ax.axvline(adjusted_mss, color='purple', 
                              linestyle='--', linewidth=2.5, alpha=0.7, zorder=4)
                    y_pos = ax.get_ylim()[1] * 0.98
                    ax.text(adjusted_mss, y_pos, 
                           ' MSS ✓', fontsize=11, color='purple', 
                           weight='bold', va='top', zorder=6,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='purple'))
            
            # نشان‌گذاری Liquidity Swept
            if liquidity_swept:
                ax.axhline(liquidity_swept['price'], color='gold', 
                          linestyle=':', linewidth=2.5, alpha=0.8, zorder=3)
                ax.text(len(df_plot) - 1, liquidity_swept['price'], 
                       f" {liquidity_swept['type']} ", 
                       fontsize=9, color='black', weight='bold', zorder=6,
                       bbox=dict(boxstyle='round', facecolor='gold', alpha=0.9, edgecolor='darkorange'))
            
            # نشان‌گذاری Entry Zone
            if entry_zone:
                top = entry_zone.get('top', entry_zone.get('high'))
                bottom = entry_zone.get('bottom', entry_zone.get('low'))
                
                ax.axhspan(bottom, top, color='cyan', alpha=0.25, zorder=1)
                ax.text(len(df_plot) - 2, (top + bottom) / 2, 
                       ' ENTRY ', fontsize=10, color='black', 
                       weight='bold', va='center', zorder=6,
                       bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.9, edgecolor='blue'))
            
            plt.tight_layout()
            
            # ذخیره با ابعاد مشخص
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', pad_inches=0.1)
            buf.seek(0)
            plt.close(fig)
            
            # بررسی و بهینه‌سازی نهایی
            img = Image.open(buf)
            width, height = img.size
            logger.debug(f"Chart initial dimensions: {width}x{height}")
            
            # اطمینان از ابعاد مناسب برای تلگرام
            # تلگرام حداکثر: width <= 10000, height <= 10000, width+height <= 10000
            max_dimension = 1920
            max_sum = 2560
            
            if width > max_dimension or height > max_dimension or (width + height) > max_sum:
                ratio = min(
                    max_dimension / width,
                    max_dimension / height,
                    max_sum / (width + height)
                )
                
                new_width = max(100, int(width * ratio * 0.9))
                new_height = max(100, int(height * ratio * 0.9))
                
                logger.info(f"Resizing chart from {width}x{height} to {new_width}x{new_height}")
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # ذخیره نهایی
            final_buf = BytesIO()
            img.save(final_buf, format='PNG', optimize=True, quality=92)
            final_buf.seek(0)
            
            file_size = final_buf.getbuffer().nbytes
            logger.debug(f"Final chart size: {file_size / 1024:.2f} KB, dimensions: {img.size}")
            
            return final_buf
            
        except Exception as e:
            logger.error(f"Error creating chart: {e}", exc_info=True)
            return None


# ======================== ICT TRADING BOT ========================
class ICTTradingBot:
    def __init__(self, symbol="EURUSD", lot_size=0.01, telegram_token=None, telegram_chat_id=None):
        self.symbol = symbol
        self.lot_size = lot_size
        self.timezone = pytz.timezone("America/New_York")
        self.positions = []
        
        logger.info(f"Initializing ICT Trading Bot for {symbol} with lot size {lot_size}")
        
        # Telegram Setup
        if telegram_token and telegram_chat_id:
            self.telegram = TelegramNotifier(telegram_token, telegram_chat_id)
            self.send_telegram = True
            logger.info("Telegram notifications enabled")
        else:
            self.telegram = None
            self.send_telegram = False
            logger.warning("Telegram notifications disabled - no credentials provided")
        
    def initialize_mt5(self, login, password, server):
        """اتصال به MetaTrader 5"""
        logger.info(f"Attempting to connect to MT5 - Server: {server}, Login: {login}")
        
        if not mt5.initialize():
            logger.error("Failed to initialize MT5")
            return False
        
        if not mt5.login(login, password, server):
            error = mt5.last_error()
            logger.error(f"Failed to login to MT5: {error}")
            return False
        
        account_info = mt5.account_info()
        logger.info(f"MT5 connected successfully - Balance: {account_info.balance}, Server: {account_info.server}")
        
        if self.send_telegram:
            self.telegram.send_message(
                "🤖 <b>ربات معاملاتی ICT راه‌اندازی شد</b>\n\n"
                f"📊 سمبل: <code>{self.symbol}</code>\n"
                f"💰 حجم: <code>{self.lot_size}</code>\n"
                f"💵 موجودی: <code>{account_info.balance}</code>\n"
                f"🖥️ سرور: <code>{account_info.server}</code>\n"
                f"🕐 زمان: {datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S')}"
            )
        
        return True
    
    def get_candles(self, timeframe, count=500):
        """دریافت کندل‌ها"""
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)
            if rates is None:
                logger.error(f"Failed to get candles for {self.symbol}")
                return None
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            logger.debug(f"Retrieved {len(df)} candles for timeframe {timeframe}")
            return df
        except Exception as e:
            logger.error(f"Error getting candles: {e}", exc_info=True)
            return None
    
    def identify_swing_high_low(self, df):
        """شناسایی Swing High/Low با فیلتر BSL/SSL معتبر"""
        swings = {'highs': [], 'lows': []}
        
        for i in range(2, len(df)-2):
            # Swing High (BSL)
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i-2] and
                df['high'].iloc[i] > df['high'].iloc[i+1] and 
                df['high'].iloc[i] > df['high'].iloc[i+2]):
                swings['highs'].append({
                    'index': i, 
                    'price': df['high'].iloc[i], 
                    'time': df['time'].iloc[i],
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
                    'time': df['time'].iloc[i],
                    'type': 'SSL'
                })
        
        # فیلتر BSL: حذف BSL هایی که بعدشان BSL با High بالاتر دارند
        filtered_highs = []
        for i, swing in enumerate(swings['highs']):
            is_valid = True
            # بررسی BSL های بعدی
            for j in range(i + 1, len(swings['highs'])):
                if swings['highs'][j]['price'] > swing['price']:
                    is_valid = False
                    break
            if is_valid:
                filtered_highs.append(swing)
        
        # فیلتر SSL: حذف SSL هایی که بعدشان SSL با Low پایین‌تر دارند
        filtered_lows = []
        for i, swing in enumerate(swings['lows']):
            is_valid = True
            # بررسی SSL های بعدی
            for j in range(i + 1, len(swings['lows'])):
                if swings['lows'][j]['price'] < swing['price']:
                    is_valid = False
                    break
            if is_valid:
                filtered_lows.append(swing)
        
        swings['highs'] = filtered_highs
        swings['lows'] = filtered_lows
        
        logger.debug(f"Identified {len(filtered_highs)} valid BSL and {len(filtered_lows)} valid SSL")
        
        return swings
    
    def check_liquidity_sweep(self, df, liquidity_price, is_high=True):
        """بررسی sweep شدن لیکوئیدیتی"""
        for i in range(max(0, len(df)-10), len(df)):
            if is_high:
                if df['high'].iloc[i] >= liquidity_price:
                    logger.debug(f"BSL swept at index {i}, price {liquidity_price}")
                    return True, i
            else:
                if df['low'].iloc[i] <= liquidity_price:
                    logger.debug(f"SSL swept at index {i}, price {liquidity_price}")
                    return True, i
        return False, -1
    
    def identify_fvg(self, df):
        """شناسایی Fair Value Gap"""
        fvgs = {'bullish': [], 'bearish': []}
        
        for i in range(2, len(df)):
            # Bullish FVG
            gap_size = df['low'].iloc[i] - df['high'].iloc[i-2]
            if gap_size > 0:
                fvgs['bullish'].append({
                    'index': i,
                    'top': df['low'].iloc[i],
                    'bottom': df['high'].iloc[i-2],
                    'mid': (df['low'].iloc[i] + df['high'].iloc[i-2]) / 2,
                    'time': df['time'].iloc[i],
                    'size': gap_size
                })
            
            # Bearish FVG
            gap_size = df['low'].iloc[i-2] - df['high'].iloc[i]
            if gap_size > 0:
                fvgs['bearish'].append({
                    'index': i,
                    'top': df['low'].iloc[i-2],
                    'bottom': df['high'].iloc[i],
                    'mid': (df['low'].iloc[i-2] + df['high'].iloc[i]) / 2,
                    'time': df['time'].iloc[i],
                    'size': gap_size
                })
        
        logger.debug(f"Found {len(fvgs['bullish'])} bullish FVGs and {len(fvgs['bearish'])} bearish FVGs")
        return fvgs
    
    def check_displacement(self, df, start_idx, end_idx):
        """بررسی Displacement"""
        if end_idx <= start_idx or end_idx - start_idx < 2:
            return False
        
        # کندل‌های متوالی
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
        
        # Full Body Candles
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
    
    def find_intermediate_swing(self, df, swings, direction='bullish'):
        """یافتن Swing Intermediate"""
        if direction == 'bullish' and len(swings['lows']) >= 2:
            last_low = swings['lows'][-1]
            
            # شرط اول: داخل FVG
            fvgs = self.identify_fvg(df)
            for fvg in fvgs['bullish']:
                if (fvg['bottom'] <= last_low['price'] <= fvg['top'] and 
                    fvg['index'] <= last_low['index']):
                    logger.info(f"Intermediate Swing found in FVG at {last_low['price']}")
                    return last_low, 'fvg'
            
            # شرط دوم: Short Term Swing
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
    
    def detect_mss_m1(self, df_m1, intermediate_swing, direction='bullish'):
        """تشخیص MSS در تایم فریم 1 دقیقه"""
        if intermediate_swing is None:
            return False, -1
        
        swing_price = intermediate_swing['price']
        swing_idx = intermediate_swing['index']
        
        for i in range(swing_idx, len(df_m1)):
            if direction == 'bullish':
                if df_m1['high'].iloc[i] > swing_price:
                    if self.check_displacement(df_m1, swing_idx, i):
                        logger.info(f"MSS confirmed (Bullish) at index {i}")
                        return True, i
            else:
                if df_m1['low'].iloc[i] < swing_price:
                    if self.check_displacement(df_m1, swing_idx, i):
                        logger.info(f"MSS confirmed (Bearish) at index {i}")
                        return True, i
        
        return False, -1
    
    def is_trading_session(self):
        """بررسی بازه‌های زمانی معاملاتی"""
        now = datetime.now(self.timezone)
        hour = now.hour
        
        trading_sessions = [(2, 5), (7, 11), (14, 15)]
        
        for start, end in trading_sessions:
            if start <= hour < end:
                return True, hour
        
        return False, hour
    
    def get_daily_open(self):
        """دریافت قیمت باز شدن روز"""
        try:
            now = datetime.now(self.timezone)
            today_open = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_D1, 
                                          today_open - timedelta(days=2), 
                                          now)
            if rates is not None and len(rates) > 0:
                daily_open = rates[-1]['open']
                logger.debug(f"Daily Open: {daily_open}")
                return daily_open
            return None
        except Exception as e:
            logger.error(f"Error getting daily open: {e}", exc_info=True)
            return None
    
    def identify_order_block(self, df_m1, mss_index, direction='bullish'):
        """شناسایی Order Block"""
        if mss_index < 3:
            return None
        
        if direction == 'bullish':
            for i in range(mss_index - 1, max(0, mss_index - 10), -1):
                if df_m1['close'].iloc[i] < df_m1['open'].iloc[i]:
                    ob = {
                        'type': 'bullish_ob',
                        'high': df_m1['high'].iloc[i],
                        'low': df_m1['low'].iloc[i],
                        'open': df_m1['open'].iloc[i],
                        'mid': (df_m1['high'].iloc[i] + df_m1['low'].iloc[i]) / 2,
                        'index': i
                    }
                    logger.debug(f"Order Block found (Bullish) at index {i}")
                    return ob
        else:
            for i in range(mss_index - 1, max(0, mss_index - 10), -1):
                if df_m1['close'].iloc[i] > df_m1['open'].iloc[i]:
                    ob = {
                        'type': 'bearish_ob',
                        'high': df_m1['high'].iloc[i],
                        'low': df_m1['low'].iloc[i],
                        'open': df_m1['open'].iloc[i],
                        'mid': (df_m1['high'].iloc[i] + df_m1['low'].iloc[i]) / 2,
                        'index': i
                    }
                    logger.debug(f"Order Block found (Bearish) at index {i}")
                    return ob
        
        return None
    
    def place_order(self, order_type, entry_price, sl_price, tp_price):
        """ثبت سفارش"""
        try:
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"Symbol {self.symbol} not found")
                return None
            
            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    logger.error(f"Failed to select symbol {self.symbol}")
                    return None
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": order_type,
                "price": entry_price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 20,
                "magic": 234000,
                "comment": "ICT Bot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.retcode} - {result.comment}")
                return None
            
            logger.info(f"TRADE: Order placed successfully - Ticket: {result.order}")
            return result
            
        except Exception as e:
            logger.error(f"Error placing order: {e}", exc_info=True)
            return None
    
    def run_strategy(self):
        """اجرای استراتژی معاملاتی"""
        logger.info("=" * 60)
        logger.info("ICT Trading Bot Started")
        logger.info(f"Symbol: {self.symbol} | Lot Size: {self.lot_size}")
        logger.info("=" * 60)
        
        while True:
            try:
                is_trading, current_hour = self.is_trading_session()
                if not is_trading:
                    logger.debug(f"Outside trading session (Hour: {current_hour})")
                    time.sleep(60)
                    continue
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Cycle Start: {datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S')}")
                
                # گام 1: بررسی M5 برای Liquidity Sweep
                logger.info("Step 1: Checking M5 for Liquidity Sweep...")
                df_m5 = self.get_candles(mt5.TIMEFRAME_M5, 200)
                if df_m5 is None:
                    logger.warning("Failed to get M5 data")
                    time.sleep(30)
                    continue
                
                swings_m5 = self.identify_swing_high_low(df_m5)
                daily_open = self.get_daily_open()
                current_price = df_m5['close'].iloc[-1]
                
                logger.info(f"Current Price: {current_price:.5f}")
                if daily_open:
                    logger.info(f"Daily Open: {daily_open:.5f}")
                    bias = "SELL" if current_price > daily_open else "BUY"
                    logger.info(f"Bias: {bias} (Price {'above' if bias == 'SELL' else 'below'} Daily Open)")
                
                liquidity_swept = None
                direction = None
                
                # جستجوی SSL Sweep برای Buy
                if daily_open and current_price < daily_open:
                    logger.info("Searching for SSL Sweep (for Buy setup)...")
                    if len(swings_m5['lows']) > 0:
                        for swing in reversed(swings_m5['lows'][-3:]):
                            swept, idx = self.check_liquidity_sweep(df_m5, swing['price'], is_high=False)
                            if swept:
                                liquidity_swept = swing
                                direction = 'bullish'
                                logger.info(f"✅ SSL Swept at {swing['price']:.5f}")
                                break
                
                # جستجوی BSL Sweep برای Sell
                elif daily_open and current_price > daily_open:
                    logger.info("Searching for BSL Sweep (for Sell setup)...")
                    if len(swings_m5['highs']) > 0:
                        for swing in reversed(swings_m5['highs'][-3:]):
                            swept, idx = self.check_liquidity_sweep(df_m5, swing['price'], is_high=True)
                            if swept:
                                liquidity_swept = swing
                                direction = 'bearish'
                                logger.info(f"✅ BSL Swept at {swing['price']:.5f}")
                                break
                
                if liquidity_swept is None:
                    logger.info("No liquidity sweep detected yet. Waiting...")
                    time.sleep(30)
                    continue
                
                # گام 2: انتقال به M1 برای MSS
                logger.info("Step 2: Moving to M1 for MSS confirmation...")
                df_m1 = self.get_candles(mt5.TIMEFRAME_M1, 500)
                if df_m1 is None:
                    logger.warning("Failed to get M1 data")
                    time.sleep(30)
                    continue
                
                swings_m1 = self.identify_swing_high_low(df_m1)
                
                # یافتن Intermediate Swing
                intermediate_swing, swing_type = self.find_intermediate_swing(
                    df_m1, swings_m1, direction
                )
                
                if intermediate_swing is None:
                    logger.info(f"Waiting for Intermediate Swing ({direction})...")
                    time.sleep(30)
                    continue
                
                # تایید MSS
                mss_confirmed, mss_idx = self.detect_mss_m1(df_m1, intermediate_swing, direction)
                
                if not mss_confirmed:
                    logger.info("MSS not confirmed yet. Waiting...")
                    time.sleep(30)
                    continue
                
                logger.info("✅ MSS Confirmed!")
                
                # گام 3: یافتن Entry Zone
                logger.info("Step 3: Searching for Entry Zone...")
                fvgs_m1 = self.identify_fvg(df_m1)
                entry_zone = None
                ob = None
                
                # اولویت اول: FVG
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
                
                # اولویت دوم: Order Block
                if entry_zone is None:
                    ob = self.identify_order_block(df_m1, mss_idx, direction)
                    if ob:
                        entry_zone = ob
                        entry_zone['entry_type'] = 'OB'
                        logger.info(f"Entry Zone: Order Block ({ob['low']:.5f} - {ob['high']:.5f})")
                
                if entry_zone is None:
                    logger.info("No Entry Zone found. Waiting...")
                    time.sleep(30)
                    continue
                
                # گام 4: محاسبه و ارسال سیگنال
                logger.info("Step 4: Preparing Trade Signal...")
                
                if direction == 'bullish':
                    entry = entry_zone.get('bottom', entry_zone.get('low'))
                    zone_height = entry_zone.get('top', entry_zone.get('high')) - entry
                    sl = entry - (zone_height * 0.2)
                    risk = entry - sl
                    tp = entry + (risk * 3)
                    order_type_str = "BUY"
                    order_type_mt5 = mt5.ORDER_TYPE_BUY
                    emoji = "📈"
                else:
                    entry = entry_zone.get('top', entry_zone.get('high'))
                    zone_height = entry - entry_zone.get('bottom', entry_zone.get('low'))
                    sl = entry + (zone_height * 0.2)
                    risk = sl - entry
                    tp = entry - (risk * 3)
                    order_type_str = "SELL"
                    order_type_mt5 = mt5.ORDER_TYPE_SELL
                    emoji = "📉"
                
                # لاگ سیگنال معاملاتی
                logger.info("="*60)
                logger.info(f"TRADE SIGNAL GENERATED: {order_type_str}")
                logger.info(f"Symbol: {self.symbol}")
                logger.info(f"Liquidity Swept: {liquidity_swept['type']} at {liquidity_swept['price']:.5f}")
                logger.info(f"Entry Type: {entry_zone['entry_type']}")
                logger.info(f"Entry: {entry:.5f}")
                logger.info(f"Stop Loss: {sl:.5f}")
                logger.info(f"Take Profit: {tp:.5f}")
                logger.info(f"Risk: {risk:.5f} | Reward: {risk*3:.5f} | R:R = 1:3")
                logger.info("="*60)
                
                # ارسال به تلگرام
                if self.send_telegram:
                    try:
                        # آماده‌سازی اطلاعات BSL/SSL
                        bsl_info = ""
                        ssl_info = ""
                        
                        if len(swings_m5['highs']) > 0:
                            recent_bsl = swings_m5['highs'][-3:] if len(swings_m5['highs']) >= 3 else swings_m5['highs']
                            bsl_info = "\n".join([f"   • {bsl['price']:.5f}" for bsl in recent_bsl])
                        else:
                            bsl_info = "   • یافت نشد"
                        
                        if len(swings_m5['lows']) > 0:
                            recent_ssl = swings_m5['lows'][-3:] if len(swings_m5['lows']) >= 3 else swings_m5['lows']
                            ssl_info = "\n".join([f"   • {ssl['price']:.5f}" for ssl in recent_ssl])
                        else:
                            ssl_info = "   • یافت نشد"
                        
                        # اطلاعات FVG
                        fvg_info = ""
                        if direction == 'bullish' and len(fvgs_m1['bullish']) > 0:
                            recent_fvgs = fvgs_m1['bullish'][-3:]
                            fvg_info = "\n".join([
                                f"   • Top: {fvg['top']:.5f} | Bottom: {fvg['bottom']:.5f} | Size: {fvg['size']:.5f}"
                                for fvg in recent_fvgs
                            ])
                        elif direction == 'bearish' and len(fvgs_m1['bearish']) > 0:
                            recent_fvgs = fvgs_m1['bearish'][-3:]
                            fvg_info = "\n".join([
                                f"   • Top: {fvg['top']:.5f} | Bottom: {fvg['bottom']:.5f} | Size: {fvg['size']:.5f}"
                                for fvg in recent_fvgs
                            ])
                        else:
                            fvg_info = "   • یافت نشد"
                        
                        # اطلاعات Intermediate Swing
                        intermediate_info = (
                            f"   • قیمت: <code>{intermediate_swing['price']:.5f}</code>\n"
                            f"   • نوع: {swing_type.upper()}\n"
                            f"   • Index: {intermediate_swing['index']}"
                        )
                        
                        # اطلاعات Order Block
                        ob_info = "   • یافت نشد"
                        if ob:
                            ob_info = (
                                f"   • High: <code>{ob['high']:.5f}</code>\n"
                                f"   • Low: <code>{ob['low']:.5f}</code>\n"
                                f"   • Mid: <code>{ob['mid']:.5f}</code>\n"
                                f"   • Index: {ob['index']}"
                            )
                        
                        # پیام اصلی با اطلاعات کامل
                        message = (
                            f"{emoji} <b>═══ سیگنال معاملاتی {order_type_str} ═══</b>\n\n"
                            f"📊 <b>اطلاعات پایه:</b>\n"
                            f"├ سمبل: <code>{self.symbol}</code>\n"
                            f"├ ⏰ زمان: {datetime.now(self.timezone).strftime('%H:%M:%S')}\n"
                            f"├ 📅 تاریخ: {datetime.now(self.timezone).strftime('%Y-%m-%d')}\n"
                            f"└ 🎯 جهت: <b>{direction.upper()}</b>\n\n"
                            
                            f"💰 <b>قیمت‌های کلیدی:</b>\n"
                            f"├ Daily Open: <code>{daily_open:.5f}</code>\n"
                            f"├ قیمت فعلی: <code>{current_price:.5f}</code>\n"
                            f"└ موقعیت: {'🔴 بالای' if current_price > daily_open else '🟢 زیر'} Open\n\n"
                            
                            f"🎯 <b>BSL (Buy Side Liquidity) - M5:</b>\n"
                            f"{bsl_info}\n\n"
                            
                            f"🎯 <b>SSL (Sell Side Liquidity) - M5:</b>\n"
                            f"{ssl_info}\n\n"
                            
                            f"⚡ <b>Liquidity Swept:</b>\n"
                            f"├ نوع: <b>{liquidity_swept['type']}</b> ✅\n"
                            f"├ قیمت: <code>{liquidity_swept['price']:.5f}</code>\n"
                            f"└ زمان: {liquidity_swept['time'].strftime('%H:%M:%S')}\n\n"
                            
                            f"🔄 <b>Intermediate Swing - M1:</b>\n"
                            f"{intermediate_info}\n\n"
                            
                            f"✅ <b>MSS (Market Structure Shift):</b>\n"
                            f"├ وضعیت: <b>تایید شد</b> ✓\n"
                            f"├ Index: {mss_idx}\n"
                            f"└ جهت: <b>{direction.upper()}</b>\n\n"
                            
                            f"📦 <b>FVG (Fair Value Gap) - M1:</b>\n"
                            f"{fvg_info}\n\n"
                            
                            f"🧱 <b>Order Block - M1:</b>\n"
                            f"{ob_info}\n\n"
                            
                            f"🎯 <b>Entry Zone:</b>\n"
                            f"├ نوع: <b>{entry_zone['entry_type']}</b>\n"
                            f"├ بالا: <code>{entry_zone.get('top', entry_zone.get('high')):.5f}</code>\n"
                            f"├ پایین: <code>{entry_zone.get('bottom', entry_zone.get('low')):.5f}</code>\n"
                            f"└ وسط: <code>{entry_zone.get('mid', (entry_zone.get('top', entry_zone.get('high')) + entry_zone.get('bottom', entry_zone.get('low'))) / 2):.5f}</code>\n\n"
                            
                            f"💰 <b>═══ جزئیات معامله ═══</b>\n"
                            f"├ 📍 Entry: <code>{entry:.5f}</code>\n"
                            f"├ 🛑 Stop Loss: <code>{sl:.5f}</code>\n"
                            f"├ 🎯 Take Profit: <code>{tp:.5f}</code>\n"
                            f"├ ⚠️ Risk: <code>{risk:.5f}</code> pips\n"
                            f"├ 💎 Reward: <code>{risk*3:.5f}</code> pips\n"
                            f"├ 📊 R:R: <b>1:3</b> 🎯\n"
                            f"└ 💰 Lot Size: <code>{self.lot_size}</code>\n\n"
                            
                            f"{'🟢' if direction == 'bullish' else '🔴'} <b>Setup Ready!</b>"
                        )
                        
                        self.telegram.send_message(message)
                        logger.info("Telegram message sent successfully")
                        
                        # ارسال نمودار M5
                        logger.info("Generating M5 chart...")
                        chart_m5 = ChartGenerator.create_chart(
                            df_m5[-80:],  # فقط 80 کندل آخر
                            f"{self.symbol} - 5 Min Timeframe",
                            swings=swings_m5,
                            fvgs=self.identify_fvg(df_m5[-80:]),
                            liquidity_swept=liquidity_swept
                        )
                        
                        if chart_m5:
                            self.telegram.send_photo(
                                chart_m5,
                                f"📊 <b>نمودار 5 دقیقه</b>\n"
                                f"🎯 {liquidity_swept['type']} Swept at {liquidity_swept['price']:.5f}"
                            )
                            logger.info("M5 chart sent successfully")
                        
                        # ارسال نمودار M1
                        logger.info("Generating M1 chart...")
                        chart_m1 = ChartGenerator.create_chart(
                            df_m1[-100:],  # فقط 100 کندل آخر
                            f"{self.symbol} - 1 Min Timeframe",
                            swings=swings_m1,
                            fvgs=fvgs_m1,
                            order_blocks=[ob] if ob else [],
                            mss_index=mss_idx if mss_idx >= len(df_m1) - 100 else None,
                            entry_zone=entry_zone
                        )
                        
                        if chart_m1:
                            self.telegram.send_photo(
                                chart_m1,
                                f"📊 <b>نمودار 1 دقیقه</b>\n"
                                f"✅ MSS Confirmed\n"
                                f"🎯 Entry Zone: {entry_zone['entry_type']}\n"
                                f"📍 Entry: {entry:.5f}"
                            )
                            logger.info("M1 chart sent successfully")
                        
                        # پیام نهایی
                        final_message = (
                            f"✅ <b>سیگنال کامل ارسال شد</b>\n\n"
                            f"🔔 برای معامله واقعی، خط place_order را از حالت کامنت خارج کنید\n"
                            f"⚠️ لطفاً قبل از ورود به معامله، تایم فریم و نواحی را بررسی کنید"
                        )
                        self.telegram.send_message(final_message)
                        
                    except Exception as e:
                        logger.error(f"Error sending to Telegram: {e}", exc_info=True)
                
                # ثبت معامله (در حالت کامنت - برای فعال‌سازی کامنت را بردارید)
                # result = self.place_order(order_type_mt5, entry, sl, tp)
                # if result:
                #     logger.info(f"Order placed successfully: Ticket {result.order}")
                #     if self.send_telegram:
                #         self.telegram.send_message(
                #             f"✅ <b>معامله ثبت شد</b>\n\n"
                #             f"🎫 Ticket: <code>{result.order}</code>\n"
                #             f"💰 Volume: <code>{result.volume}</code>\n"
                #             f"📊 Price: <code>{result.price}</code>"
                #         )
                
                # انتظار 5 دقیقه قبل از چک بعدی
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
                        f"⚠️ <b>خطا در ربات</b>\n\n"
                        f"<code>{str(e)}</code>\n\n"
                        f"ربات در حال تلاش مجدد است..."
                    )
                time.sleep(30)
    
    def shutdown(self):
        """بستن اتصال MT5"""
        logger.info("Shutting down bot...")
        mt5.shutdown()
        logger.info("MT5 connection closed")
        
        if self.send_telegram:
            self.telegram.send_message(
                "🔴 <b>ربات متوقف شد</b>\n\n"
                f"⏰ زمان: {datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S')}"
            )


# ======================== MAIN ========================
if __name__ == "__main__":
    # تنظیمات اتصال MT5
    MT5_LOGIN = 5041532543  # شماره اکانت دمو
    MT5_PASSWORD = "@gEhV6Br"
    MT5_SERVER = "MetaQuotes-Demo"

    # تنظیمات تلگرام (از BotFather دریافت کنید)
    TELEGRAM_TOKEN = "7579111522:AAEx1nXSkX6uaC-l3oiWYhcAgzW5NuaJ12I"  # مثال: "1234567890:ABCdefGHIjklMNOpqrsTUVwxyz"
    TELEGRAM_CHAT_ID = "-1002883541378"  # مثال: "123456789" یا "@channel_name"
    
    # تنظیمات معامله
    SYMBOL = "EURUSD"
    LOT_SIZE = 0.01
    
    # ایجاد ربات
    bot = ICTTradingBot(
        symbol=SYMBOL,
        lot_size=LOT_SIZE,
        telegram_token=TELEGRAM_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID
    )
    
    # اتصال به MT5
    if bot.initialize_mt5(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        try:
            logger.info("Starting main strategy loop...")
            bot.run_strategy()
        except KeyboardInterrupt:
            logger.info("\n⏹️  Bot stopped by user")
            bot.shutdown()
        except Exception as e:
            logger.critical(f"Critical error: {e}", exc_info=True)
            bot.shutdown()
    else:
        logger.error("Failed to connect to MT5. Exiting...")
        if bot.send_telegram:
            bot.telegram.send_message(
                "❌ <b>خطا در اتصال به MT5</b>\n\n"
                "ربات نتوانست به MetaTrader 5 متصل شود.\n"
                "لطفاً اطلاعات اتصال را بررسی کنید."
            )