# 📊 راهنمای منابع داده برای بک‌تست - بدون محدودیت!

## 🎯 مشکل Yahoo Finance

Yahoo Finance محدودیت داره:
- ❌ M15/M5: فقط 60 روز اخیر
- ❌ داده‌های قدیمی‌تر موجود نیست

## ✅ راه‌حل: استفاده از فایل‌های CSV

با CSV می‌تونید:
- ✅ هر چقدر بخواید داده داشته باشید (سال‌ها!)
- ✅ همه تایم‌فریم‌ها (M1, M5, M15, H1, H4, D)
- ✅ دقت 100%
- ✅ سرعت بالا

---

## 🚀 روش‌های دانلود داده

### روش 1️⃣: دانلود از HistData.com (رایگان) 🌟 پیشنهادی

**بهترین منبع داده رایگان فارکس!**

#### مراحل:

1. **برو به سایت**:
   ```
   https://www.histdata.com/download-free-forex-data/
   ```

2. **انتخاب جفت‌ارز**:
   - EUR/USD
   - GBP/USD
   - USD/JPY
   - و...

3. **انتخاب تایم‌فریم**:
   - M1 (1 دقیقه)
   - TICK BID/ASK (اگر می‌خوای M5/M15 خودت بسازی)

4. **انتخاب سال و ماه**:
   - می‌تونی چند سال گذشته رو دانلود کنی!

5. **دانلود**:
   - فایل ZIP دانلود می‌شه
   - Extract کن

6. **تبدیل به فرمت مناسب**:
   - فایل معمولاً به فرمت `.csv` هست
   - ستون‌ها: `Date Time, Open, High, Low, Close, Volume`

#### مثال فایل دانلود شده:

```csv
20240101 000000,1.10450,1.10455,1.10445,1.10450,120
20240101 000100,1.10450,1.10460,1.10445,1.10458,95
20240101 000200,1.10458,1.10462,1.10450,1.10455,88
```

#### تبدیل به فرمت مناسب:

اگر فایل به این فرمت هست، باید تبدیل کنی:

```python
import pandas as pd

# خواندن فایل
df = pd.read_csv('DAT_MT_EURUSD_M1_2024.csv',
                 names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])

# تبدیل DateTime
df['time'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H%M%S')

# Rename columns
df = df.rename(columns={
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
})

# Select columns
df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

# Save
df.to_csv('historical_data/EURUSD_M1.csv', index=False)
print(f"✅ Converted: {len(df)} candles")
```

---

### روش 2️⃣: دانلود از MetaTrader 5 (اگر MT5 داری)

#### مراحل:

1. **باز کن MetaTrader 5**

2. **برو به History Center**:
   - Tools → History Center (Ctrl+H)

3. **انتخاب جفت‌ارز**:
   - Forex → EUR/USD

4. **انتخاب تایم‌فریم**:
   - M1, M5, M15, H1, H4, Daily

5. **Download**:
   - کلیک روی Download
   - داده‌ها دانلود می‌شه

6. **Export به CSV**:
   - کلیک راست روی جفت‌ارز
   - Export → CSV file

---

### روش 3️⃣: استفاده از کد Python خودکار (MT5)

اگر MT5 نصب داری، این کد رو اجرا کن:

```python
from csv_data_loader import download_from_mt5

# دانلود EUR/USD - H1 - سال 2024
success = download_from_mt5(
    symbol="EURUSD",
    timeframe="H1",
    start_date="2024-01-01",
    end_date="2024-11-01",
    output_folder="historical_data"
)

if success:
    print("✅ Data downloaded!")
else:
    print("❌ Failed. Make sure MT5 is installed and running.")
```

---

### روش 4️⃣: دانلود از Dukascopy (رایگان)

1. **برو به**:
   ```
   https://www.dukascopy.com/swiss/english/marketwatch/historical/
   ```

2. **انتخاب**:
   - Instrument: EUR/USD
   - Timeframe: H1, M15, etc.
   - Date range: از تاریخ X تا Y

3. **دانلود**:
   - فایل CSV دانلود می‌شه

4. **Rename**:
   - نام فایل رو بذار: `EURUSD_H1.csv`

---

## 📂 ساختار پوشه

بعد از دانلود، فایل‌ها رو اینطوری بذار:

```
htu_claude/
├── historical_data/        ← پوشه داده‌ها
│   ├── EURUSD_M1.csv      ← تایم‌فریم M1
│   ├── EURUSD_M5.csv      ← تایم‌فریم M5
│   ├── EURUSD_M15.csv     ← تایم‌فریم M15
│   ├── EURUSD_H1.csv      ← تایم‌فریم H1 (مهم!)
│   ├── EURUSD_H4.csv      ← تایم‌فریم H4
│   ├── EURUSD_D.csv       ← تایم‌فریم Daily (مهم!)
│   ├── GBPUSD_H1.csv      ← جفت‌ارز دیگه
│   └── ...
├── run_backtest.py        ← اسکریپت بک‌تست
└── csv_data_loader.py     ← بارگذار CSV
```

---

## 📝 فرمت فایل CSV

فایل CSV باید این ستون‌ها رو داشته باشه:

### فرمت استاندارد:
```csv
time,open,high,low,close,volume
2024-01-01 00:00:00,1.10450,1.10455,1.10445,1.10450,120
2024-01-01 01:00:00,1.10450,1.10460,1.10445,1.10458,95
```

### فرمت MetaTrader (هم قبوله):
```csv
Date,Time,Open,High,Low,Close,Volume
2024.01.01,00:00,1.10450,1.10455,1.10445,1.10450,120
2024.01.01,01:00,1.10450,1.10460,1.10445,1.10458,95
```

**نکته**: سیستم خودکار هر دو فرمت رو تشخیص میده! ✅

---

## ⚙️ نحوه استفاده

### 1️⃣ فعال کردن CSV در `run_backtest.py`:

```python
# باز کن: run_backtest.py
# خط 33 رو تغییر بده:

USE_CSV = True  # فعال کردن CSV
CSV_FOLDER = "historical_data"  # پوشه فایل‌ها

SYMBOL = "EURUSD"  # بدون =X
```

### 2️⃣ اجرا:

```bash
python run_backtest.py
```

---

## ✅ تست فایل CSV

برای اطمینان از درستی فایل:

```python
from csv_data_loader import CSVDataLoader

loader = CSVDataLoader("historical_data")

# تست بارگذاری
loader.validate_csv("EURUSD", "H1")
```

خروجی:
```
✅ CSV is valid: EURUSD H1
  - Candles: 5200
  - Date range: 2024-01-01 to 2024-11-01
  - Price range: 1.05234 to 1.12456
```

---

## 🎯 مثال کامل: دانلود و استفاده

### مرحله 1: دانلود از HistData

1. برو به: https://www.histdata.com/download-free-forex-data/
2. انتخاب: EUR/USD, M1, Year: 2024
3. دانلود فایل ZIP
4. Extract: `DAT_MT_EURUSD_M1_202401.csv`

### مرحله 2: تبدیل (اگر نیاز باشه)

ذخیره این کد به عنوان `convert_histdata.py`:

```python
import pandas as pd
import os

def convert_histdata_to_csv(input_file, output_file, timeframe='M1'):
    """
    Convert HistData.com format to our format
    """
    print(f"📥 Reading {input_file}...")

    # Read with specific format
    df = pd.read_csv(input_file,
                     names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'],
                     sep=';' if ';' in open(input_file).readline() else ',')

    print(f"   Loaded {len(df)} rows")

    # Convert datetime
    df['time'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H%M%S')

    # Rename
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    # Select columns
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

    # Remove duplicates
    df = df.drop_duplicates(subset=['time'])

    # Sort
    df = df.sort_values('time')

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"✅ Saved to {output_file}")
    print(f"   {len(df)} candles")
    print(f"   From: {df['time'].min()}")
    print(f"   To: {df['time'].max()}")

# استفاده:
convert_histdata_to_csv(
    'DAT_MT_EURUSD_M1_202401.csv',
    'historical_data/EURUSD_M1.csv',
    'M1'
)
```

اجرا:
```bash
python convert_histdata.py
```

### مرحله 3: Resample به تایم‌فریم‌های بالاتر

اگر فقط M1 داری، می‌تونی M5, M15, H1 بسازی:

```python
import pandas as pd

# خواندن M1
df = pd.read_csv('historical_data/EURUSD_M1.csv')
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')

# Resample به M5
df_m5 = df.resample('5min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

df_m5 = df_m5.reset_index()
df_m5.to_csv('historical_data/EURUSD_M5.csv', index=False)
print(f"✅ M5: {len(df_m5)} candles")

# Resample به M15
df_m15 = df.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

df_m15 = df_m15.reset_index()
df_m15.to_csv('historical_data/EURUSD_M15.csv', index=False)
print(f"✅ M15: {len(df_m15)} candles")

# Resample به H1
df_h1 = df.resample('1h').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

df_h1 = df_h1.reset_index()
df_h1.to_csv('historical_data/EURUSD_H1.csv', index=False)
print(f"✅ H1: {len(df_h1)} candles")

# Daily
df_d = df.resample('1d').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

df_d = df_d.reset_index()
df_d.to_csv('historical_data/EURUSD_D.csv', index=False)
print(f"✅ Daily: {len(df_d)} candles")
```

### مرحله 4: اجرای بک‌تست

```bash
python run_backtest.py
```

---

## 🎓 نکات مهم

### ✅ حداقل فایل‌های لازم:

برای بک‌تست موفق نیاز به:
1. **H1** یا **M15** - برای liquidity sweep
2. **M5** یا **M15** - برای MSS confirmation
3. **D** (Daily) - برای bias determination

مثال:
```
EURUSD_H1.csv  ← برای liquidity
EURUSD_M15.csv ← برای MSS (اگر H1 استفاده می‌کنی)
EURUSD_D.csv   ← برای daily open
```

### ✅ چند داده کافیه؟

- **حداقل**: 3 ماه (برای آمار معتبر)
- **توصیه**: 6-12 ماه
- **عالی**: 2-3 سال!

### ⚠️ خطاهای رایج:

1. **فرمت تاریخ اشتباه**:
   - درست: `2024-01-01 10:00:00`
   - اشتباه: `01/01/2024 10:00`

2. **ستون‌های اضافی**:
   - سیستم خودکار ستون‌های اضافی رو ignore می‌کنه

3. **نام فایل اشتباه**:
   - درست: `EURUSD_H1.csv`
   - اشتباه: `eurusd_h1.csv` یا `EUR-USD_H1.csv`

---

## 🚀 شروع سریع (3 دقیقه)

### گام 1: دانلود نمونه

```bash
# دانلود داده نمونه از HistData
# یا استفاده از داده موجود MT5
```

### گام 2: آماده‌سازی

```bash
# ساخت پوشه
mkdir historical_data

# کپی فایل‌ها
# EURUSD_H1.csv → historical_data/
# EURUSD_D.csv → historical_data/
```

### گام 3: تنظیمات

```python
# run_backtest.py
USE_CSV = True
SYMBOL = "EURUSD"  # بدون =X
START_DATE = "2024-01-01"
END_DATE = "2024-11-01"
```

### گام 4: اجرا!

```bash
python run_backtest.py
```

🎉 **حالا بک‌تست بدون محدودیت!**

---

## 📚 منابع بیشتر

- **HistData.com**: https://www.histdata.com/
- **Dukascopy**: https://www.dukascopy.com/swiss/english/marketwatch/historical/
- **TrueFX**: https://www.truefx.com/
- **FXCM**: https://www.fxcm.com/uk/market-data/

---

**موفق باشید! 🚀**

نیاز به کمک؟ سوال کنید! 😊
