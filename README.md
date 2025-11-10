# 🤖 ICT Trading Bot V2.0

> Complete rewrite of the ICT trading bot with **NO MetaTrader dependency** - Works perfectly on Linux VPS!

[📖 Persian Documentation](README_FA.md) | [📄 Strategy PDF](HTU1st.pdf)

---

## 🎯 What's New in V2?

### ✅ Key Improvements

- **No MetaTrader5 Required** - Pure Python, runs anywhere
- **Linux VPS Ready** - Perfect for cloud deployment
- **Data Provider Abstraction** - OANDA, YFinance, easily extensible
- **Fixed FVG Detection** - Proper algorithm with filtering
- **Professional Charts** - Beautiful visualizations with annotations
- **Modular Architecture** - Clean, maintainable code

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config.json.example config.json
# Edit config.json with your settings
```

### 3. Run

```bash
# Test mode (simulated data)
python ict_bot_complete.py

# Real mode (OANDA - configure credentials first)
python ict_bot_complete.py
```

---

## 📊 Backtesting - Test Your Strategy

**NEW!** Test the bot on historical data to see Win Rate and performance before live trading!

### Quick Backtest:

```bash
# Install dependencies
pip install yfinance pandas numpy matplotlib

# Run backtest
python run_backtest.py
```

### What You Get:

- ✅ **Win Rate** - Percentage of winning trades
- ✅ **Profit Factor** - Ratio of wins to losses
- ✅ **ROI** - Return on investment
- ✅ **Equity Curve** - Visual chart of account balance
- ✅ **Trade History** - Detailed CSV with all trades
- ✅ **Max Drawdown** - Worst decline from peak

### Customize Backtest:

Edit `run_backtest.py` to change:
- Symbol (EUR/USD, GBP/USD, etc.)
- Date range (test any period)
- Initial balance
- Lot size
- Strategy timeframes (M5+M1 or M15+M5)

### Using CSV Files (Unlimited Data):

Yahoo Finance has 60-day limitation. Use CSV files for unlimited testing:

```bash
# 1. Download data from HistData.com or MT5
# 2. Convert to CSV format
python convert_histdata.py your_file.csv EURUSD

# 3. Enable CSV in run_backtest.py
USE_CSV = True
SYMBOL = "EURUSD"

# 4. Run backtest
python run_backtest.py
```

📖 **[Full Backtesting Guide (Persian)](BACKTEST_GUIDE_FA.md)**
📊 **[Data Sources Guide (Persian)](DATA_SOURCES_GUIDE_FA.md)** - Download unlimited historical data!

---

## 📊 Strategy Overview

Based on **HTU Trading Strategy PDF**:

### Trading Flow:

1. **M5 Timeframe** → Detect Liquidity Sweep (BSL/SSL)
2. **M1 Timeframe** → Confirm MSS (Market Structure Shift)
3. **Entry Zone** → FVG (priority) or Order Block
4. **Risk/Reward** → Fixed 1:3

### Trading Sessions (UTC-4):
- London: 02:00 - 05:00
- New York AM: 07:00 - 11:00
- New York PM: 14:00 - 15:00

---

## 📋 Complete Features List

### 🔌 Data Provider System
- ✅ **OANDA Integration** - Connect to OANDA Forex API
  - Fetch real-time M1, M5, M15, H1, H4, Daily candles
  - Place market orders with SL/TP
  - Practice and live account support
- ✅ **Simulated Provider** - Yahoo Finance for testing
  - Free historical data
  - Simulated order placement
- ✅ **Extensible Architecture** - Easy to add new providers (Binance, MT5, etc.)

### 📈 Technical Analysis Components

#### Swing Points Detection
- ✅ **Swing High/Low Identification** (3-candle fractal method)
- ✅ **BSL (Buy-Side Liquidity)** - Above swing highs
- ✅ **SSL (Sell-Side Liquidity)** - Below swing lows
- ✅ **Automatic Filtering** - Remove invalidated swings

#### Fair Value Gaps (FVG)
- ✅ **Bullish FVG Detection** - Gaps showing bullish imbalance
- ✅ **Bearish FVG Detection** - Gaps showing bearish imbalance
- ✅ **Minimum Gap Filter** - Ignore insignificant gaps
- ✅ **Zone Calculation** - Top, bottom, and middle prices

#### Market Structure
- ✅ **Intermediate Swing Detection** - Find key reversal points
  - Inside FVG check
  - Short-term swing identification
- ✅ **MSS (Market Structure Shift)** - Confirm trend change
  - Requires displacement
  - Direction-aware (bullish/bearish)
- ✅ **Displacement Check** - Validate strong moves
  - Consecutive same-color candles (3+)
  - Full-body candles (70%+ body ratio)

#### Order Blocks
- ✅ **Bullish Order Block** - Last bearish candle before MSS
- ✅ **Bearish Order Block** - Last bullish candle before MSS
- ✅ **Fallback Entry Zone** - When no FVG is available

### 🎯 Trading Logic

#### Session Management
- ✅ **Time Zone Support** - Configurable (default: America/New_York)
- ✅ **Trading Windows** - London (2-5), NY AM (7-11), NY PM (14-15)
- ✅ **Auto Pause** - Outside trading hours

#### Bias Determination
- ✅ **Daily Open Reference** - Fetch D timeframe
- ✅ **Price Position Analysis**
  - Above Daily Open → Bearish Bias (look for BSL sweep)
  - Below Daily Open → Bullish Bias (look for SSL sweep)

#### Signal Generation (Step-by-Step)
1. ✅ **Step 1** - M5 Liquidity Sweep Detection
   - Check SSL sweep for buy setups
   - Check BSL sweep for sell setups
   - Based on daily open bias
2. ✅ **Step 2** - M1 MSS Confirmation
   - Find intermediate swing
   - Wait for displacement
   - Confirm market structure shift
3. ✅ **Step 3** - Entry Zone Identification
   - Priority 1: FVG (after MSS)
   - Priority 2: Order Block
4. ✅ **Step 4** - Trade Parameters Calculation
   - Entry: Bottom of bullish zone / Top of bearish zone
   - Stop Loss: 20% outside entry zone
   - Take Profit: 3× risk (1:3 R/R)
5. ✅ **Step 5** - Signal Distribution
   - Send to Telegram with full analysis
   - Generate annotated charts (M5 + M1)
6. ✅ **Step 6** - Order Execution (optional, disabled by default)

### 📊 Chart Generation

- ✅ **Professional Candlestick Charts**
  - Green/red color scheme
  - Real-time price display
  - Configurable candle count (default: 100)
- ✅ **Swing Points Overlay**
  - Red markers for BSL (swing highs)
  - Green markers for SSL (swing lows)
  - Labels for easy identification
- ✅ **FVG Visualization**
  - Green zones for bullish FVG
  - Red zones for bearish FVG
  - Transparent rectangles with labels
- ✅ **Order Block Highlighting**
  - Blue for bullish OB
  - Orange for bearish OB
  - Extended zone display
- ✅ **MSS Marker**
  - Purple vertical line
  - "MSS ✓" label
- ✅ **Liquidity Sweep Indicator**
  - Gold horizontal line
  - Type label (BSL/SSL)
- ✅ **Entry Zone Highlight**
  - Cyan horizontal zone
  - "ENTRY" label
- ✅ **Auto-Resize for Telegram** - Optimized image dimensions

### 📱 Telegram Integration

- ✅ **Rich HTML Messages**
  - Emoji indicators
  - Formatted tables
  - Code blocks for prices
- ✅ **Trade Signals Include:**
  - Direction (BUY/SELL)
  - Symbol and timestamp
  - Daily open and current price
  - Liquidity swept details
  - MSS confirmation
  - Entry zone type and range
  - Entry, SL, TP prices
  - Risk/Reward breakdown
  - Lot size
- ✅ **Chart Attachments**
  - M5 chart with liquidity sweep
  - M1 chart with MSS and entry
  - Captions with key info
- ✅ **Status Notifications**
  - Bot start/stop
  - Error alerts
  - Automatic recovery messages
- ✅ **Image Optimization**
  - Auto compression
  - Resolution adjustment
  - Fast delivery

### 🛡️ Risk Management

- ✅ **Fixed Risk/Reward** - Always 1:3
- ✅ **Stop Loss Calculation** - Based on entry zone size
- ✅ **Take Profit Calculation** - 3× stop loss distance
- ✅ **Position Sizing** - Configurable lot size
- ✅ **No Over-Trading** - 5-minute cooldown between signals

### 📝 Logging System

- ✅ **Multi-Level Logging**
  - DEBUG: Detailed technical info
  - INFO: Important events
  - WARNING: Potential issues
  - ERROR: Failures with stack traces
- ✅ **Dual Output**
  - Console: INFO level and above
  - File: All levels (DEBUG+)
- ✅ **Log Rotation**
  - Max file size: 10 MB
  - Keep last 5 files
  - UTF-8 encoding (Persian support)
- ✅ **Structured Format** - Timestamp | Level | Function | Message
- ✅ **Auto Directory Creation** - `logs/` folder

### 🔧 Configuration Management

- ✅ **JSON Configuration File**
  - Symbol selection
  - Lot size
  - Provider type
  - API credentials
  - Telegram settings
- ✅ **Environment Flexibility** - Test/production switching
- ✅ **Default Fallbacks** - Missing config handled gracefully

### 🔄 Bot Lifecycle

- ✅ **Graceful Startup**
  - Load configuration
  - Initialize providers
  - Connect to Telegram
  - Send start notification
- ✅ **Main Loop**
  - Check trading session
  - Execute strategy steps
  - Handle errors automatically
  - Retry on failures
- ✅ **Clean Shutdown**
  - Keyboard interrupt handling (Ctrl+C)
  - Send stop notification
  - Log final status

### ⚙️ System Requirements

- ✅ **Python 3.7+**
- ✅ **No GUI Required** - CLI only
- ✅ **Low Resource Usage** - Runs on basic VPS
- ✅ **Cross-Platform** - Linux, Windows, macOS

---

## ❓ What Can Be Customized?

The following features can be **modified or removed** based on trader preferences:

### Can Be Disabled/Changed:
- 🔧 Trading sessions (London, NY AM, NY PM)
- 🔧 Risk/Reward ratio (currently 1:3)
- 🔧 Stop loss calculation method
- 🔧 Entry zone selection priority (FVG vs Order Block)
- 🔧 Displacement validation (candle count, body ratio)
- 🔧 Swing point filtering logic
- 🔧 Minimum FVG gap size
- 🔧 Chart visual style and indicators
- 🔧 Telegram message format
- 🔧 Loop delay (5 minutes between checks)

### Core Components (Recommended to Keep):
- ✅ Liquidity sweep detection
- ✅ MSS confirmation
- ✅ Daily open bias
- ✅ Entry zone identification
- ✅ Stop loss (risk management)

---

## 🛠️ Configuration

```json
{
  "symbol": "EUR_USD",
  "lot_size": 1000,
  "provider": "simulated",

  "telegram_token": "YOUR_BOT_TOKEN",
  "telegram_chat_id": "YOUR_CHAT_ID",

  "oanda_api_key": "YOUR_KEY",
  "oanda_account_id": "YOUR_ACCOUNT_ID",
  "oanda_practice": true
}
```

### Providers:

- **`simulated`** - Uses Yahoo Finance (free, for testing)
- **`oanda`** - OANDA API (forex trading, requires account)

---

## 🐧 Linux VPS Deployment

### Install:
```bash
sudo apt update
sudo apt install python3 python3-pip -y
pip3 install -r requirements.txt
```

### Run in Background:
```bash
nohup python3 ict_bot_complete.py > bot.log 2>&1 &
```

### View Logs:
```bash
tail -f bot.log
tail -f logs/bot_full.log
```

### Stop:
```bash
pkill -f ict_bot_complete.py
```

---

## 📱 Telegram Setup

1. Create bot with [@BotFather](https://t.me/BotFather)
2. Get your token
3. Send a message to your bot
4. Get chat ID from: `https://api.telegram.org/bot<TOKEN>/getUpdates`

---

## 📈 Example Output

The bot sends:
- ✅ Trade signals with full analysis
- 📊 M5 chart with liquidity sweep
- 📊 M1 chart with MSS and entry zone

```
📈 ═══ TRADE SIGNAL BUY ═══

📊 Base Info:
├ Symbol: EUR_USD
├ Direction: BULLISH
└ Time: 08:23:15

⚡ Liquidity Swept: SSL ✅
✅ MSS Confirmed
🎯 Entry Zone: FVG

💰 TRADE DETAILS:
├ Entry: 1.07720
├ Stop Loss: 1.07714
├ Take Profit: 1.07738
└ R:R: 1:3 🎯
```

---

## 🔍 Comparison

| Feature | Old (V1) | New (V2) |
|---------|----------|----------|
| **MT5 Dependency** | ✅ Required | ❌ None |
| **Linux Support** | ❌ Difficult | ✅ Easy |
| **Data Provider** | MT5 only | Multiple |
| **FVG Detection** | Basic | Improved |
| **Charts** | Basic | Professional |
| **Code Quality** | Nested | Modular |

---

## 📁 Project Structure

```
htu_claude/
├── ict_bot_complete.py      # Main bot (all-in-one)
├── config.json               # Your settings
├── config.json.example       # Template
├── requirements.txt          # Dependencies
├── README.md                # This file (English)
├── README_FA.md             # Persian docs
├── HTU1st.pdf               # Original strategy
├── logs/                    # Auto-created
└── hamid_ict_v18.py         # Old version (MT5)
```

---

## ⚠️ Important Notes

1. **For Educational Purposes** - Test thoroughly before live trading
2. **Risk Management** - Always use Stop Loss
3. **Monitor Logs** - Check logs regularly

---

## 📄 License

For personal and educational use.

**Author**: Hamid Tabasi (HTU)
**Rewritten by**: Claude AI
**Date**: 2024-11-09

---

**Made with ❤️ for the HTU Trading Community**