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