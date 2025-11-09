# 🤖 ربات معاملاتی ICT - نسخه 2.0

## ✨ ویژگی‌های جدید

### ✅ بدون وابستگی به MetaTrader
- کاملاً مستقل از MT5
- قابل اجرا روی Linux VPS
- پشتیبانی از چندین data provider

### ✅ FVG Detection بهبود یافته
- شناسایی دقیق Fair Value Gaps
- فیلتر کردن FVG های ناچیز
- مطابق با آموزش PDF

### ✅ Chart Visualization حرفه‌ای
- رسم کندل‌ها با کیفیت بالا
- نمایش Swing High/Low (BSL/SSL)
- نمایش FVG ها
- نمایش Order Blocks
- نمایش MSS و Entry Zones

### ✅ ساختار ماژولار
- Data Provider Abstraction
- پشتیبانی از OANDA API
- پشتیبانی از Simulated Mode (YFinance)
- آماده برای اضافه کردن CCXT و سایر API ها

---

## 📚 استراتژی (مطابق PDF HTU)

### جریان کامل:

1. **M5 Timeframe**: بررسی Liquidity Sweep
   - شناسایی BSL/SSL با Fractal سه‌گانه
   - تعیین Bias با Daily Open

2. **M1 Timeframe**: تایید MSS
   - یافتن Intermediate Swing
   - تایید Market Structure Shift
   - بررسی Displacement

3. **Entry Zone**: یافتن ناحیه ورود
   - اولویت اول: FVG
   - اولویت دوم: Order Block

4. **Risk Management**: 1:3 Risk/Reward
   - Entry در ناحیه مناسب
   - SL: 20% خارج از zone
   - TP: 3 برابر ریسک

### تایم‌های معاملاتی (UTC-4):
- **London**: 2:00 - 5:00
- **New York AM**: 7:00 - 11:00
- **New York PM**: 14:00 - 15:00

---

## 🚀 نصب و راه‌اندازی

### 1. نصب Dependencies

```bash
pip install -r requirements.txt
```

### 2. ساخت فایل Configuration

```bash
cp config.json.example config.json
```

سپس `config.json` را ویرایش کنید:

```json
{
  "symbol": "EUR_USD",
  "lot_size": 1000,
  "provider": "simulated",

  "telegram_token": "توکن ربات تلگرام",
  "telegram_chat_id": "آی‌دی چت تلگرام",

  "oanda_api_key": "کلید API اوآندا (اختیاری)",
  "oanda_account_id": "شماره اکانت اوآندا (اختیاری)",
  "oanda_practice": true
}
```

### 3. اجرای ربات

#### حالت Simulated (آزمایشی):
```bash
python ict_bot_complete.py
```

#### حالت OANDA (واقعی/Practice):
```json
{
  "provider": "oanda",
  "oanda_api_key": "YOUR_KEY",
  "oanda_account_id": "YOUR_ACCOUNT_ID",
  "oanda_practice": true
}
```

```bash
python ict_bot_complete.py
```

---

## 🔧 تنظیمات OANDA API

### دریافت API Key:

1. ثبت‌نام در OANDA: https://www.oanda.com
2. ورود به Dashboard
3. Manage API Access
4. Generate API Token
5. کپی کردن Account ID و Token

### نکات مهم:
- برای تست از **Practice Account** استفاده کنید
- `oanda_practice: true` برای حساب تمرینی
- `oanda_practice: false` برای حساب واقعی

---

## 📱 تنظیمات Telegram

### ساخت Bot:

1. در تلگرام به `@BotFather` پیام دهید
2. دستور `/newbot` را بفرستید
3. نام و username برای ربات انتخاب کنید
4. Token دریافتی را کپی کنید

### دریافت Chat ID:

1. به ربات خود پیام بدهید
2. به این آدرس بروید:
   ```
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
3. `chat.id` را پیدا کنید

یا از `@userinfobot` استفاده کنید.

---

## 🐧 اجرا روی Linux VPS

### نصب Python:
```bash
sudo apt update
sudo apt install python3 python3-pip -y
```

### کلون کردن پروژه:
```bash
cd ~
git clone <repository_url>
cd htu_claude
```

### نصب Dependencies:
```bash
pip3 install -r requirements.txt
```

### اجرا در Background:
```bash
nohup python3 ict_bot_complete.py > bot.log 2>&1 &
```

### مشاهده لاگ:
```bash
tail -f bot.log
tail -f logs/bot_full.log
```

### Stop کردن:
```bash
pkill -f ict_bot_complete.py
```

### اجرای خودکار با systemd:

فایل `/etc/systemd/system/ict-bot.service` بسازید:

```ini
[Unit]
Description=ICT Trading Bot
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/htu_claude
ExecStart=/usr/bin/python3 /home/YOUR_USERNAME/htu_claude/ict_bot_complete.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

فعال‌سازی:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ict-bot
sudo systemctl start ict-bot
sudo systemctl status ict-bot
```

---

## 📊 ساختار پروژه

```
htu_claude/
├── ict_bot_complete.py      # فایل اصلی ربات
├── config.json               # تنظیمات (بسازید از example)
├── config.json.example       # نمونه تنظیمات
├── requirements.txt          # Dependencies
├── README_FA.md             # این فایل
├── HTU1st.pdf               # PDF استراتژی اصلی
├── logs/                    # پوشه لاگ‌ها (خودکار ساخته می‌شود)
│   ├── bot_full.log
│   └── bot_errors.log
└── hamid_ict_v18.py         # نسخه قدیمی (با MT5)
```

---

## 🔍 تفاوت‌های نسخه 2 با نسخه 1

| ویژگی | نسخه 1 (v18) | نسخه 2 |
|-------|--------------|---------|
| **وابستگی MT5** | ✅ دارد | ❌ ندارد |
| **اجرا روی Linux** | ❌ سخت | ✅ آسان |
| **Data Provider** | فقط MT5 | OANDA, YFinance, قابل توسعه |
| **FVG Detection** | ساده | بهبود یافته + فیلتر |
| **Chart Quality** | متوسط | حرفه‌ای با رنگ‌بندی بهتر |
| **Code Structure** | تو در تو | ماژولار و تمیز |
| **Configuration** | Hard-coded | JSON file |
| **Testing** | نیاز به MT5 | Simulated mode |

---

## 🎯 مثال خروجی Telegram

```
📈 ═══ TRADE SIGNAL BUY ═══

📊 Base Info:
├ Symbol: EUR_USD
├ ⏰ Time: 08:23:15
├ 📅 Date: 2024-11-09
└ 🎯 Direction: BULLISH

💰 Key Prices:
├ Daily Open: 1.07850
├ Current: 1.07720
└ Position: 🟢 Below Open

⚡ Liquidity Swept:
├ Type: SSL ✅
└ Price: 1.07680

✅ MSS Confirmed at Index 245

🎯 Entry Zone:
├ Type: FVG
├ Top: 1.07750
└ Bottom: 1.07720

💰 ═══ TRADE DETAILS ═══
├ 📍 Entry: 1.07720
├ 🛑 Stop Loss: 1.07714
├ 🎯 Take Profit: 1.07738
├ ⚠️ Risk: 0.00006
├ 💎 Reward: 0.00018
├ 📊 R:R: 1:3 🎯
└ 💰 Lot Size: 1000

🟢 Setup Ready!
```

بعد از آن، دو چارت M5 و M1 با تمام annotations ارسال می‌شود.

---

## 🛠️ عیب‌یابی

### مشکل: ربات شروع نمی‌شود
```bash
# بررسی لاگ
tail -f logs/bot_full.log

# بررسی خطاها
tail -f logs/bot_errors.log
```

### مشکل: Data نمی‌گیرد
- بررسی اتصال به اینترنت
- در حالت OANDA: بررسی API credentials
- در حالت Simulated: ممکن است Yahoo Finance محدودیت داشته باشد

### مشکل: تلگرام کار نمی‌کند
- بررسی صحت Token و Chat ID
- مطمئن شوید به ربات پیام داده‌اید
- فایروال یا Proxy را بررسی کنید

### مشکل: Chart ارسال نمی‌شود
- نصب `matplotlib` و `Pillow`
- بررسی حجم تصویر در لاگ

---

## 📈 بهبودهای آینده

- [ ] اضافه کردن CCXT برای Crypto
- [ ] پشتیبانی از Interactive Brokers
- [ ] Web Dashboard برای مانیتورینگ
- [ ] Backtesting System
- [ ] Multi-symbol support
- [ ] Advanced Risk Management
- [ ] Database logging

---

## ⚠️ هشدارها

1. **این ربات برای آموزش است**
   - قبل از استفاده واقعی، تست کامل انجام دهید
   - با حساب Practice شروع کنید

2. **مدیریت ریسک**
   - همیشه از Stop Loss استفاده کنید
   - سرمایه بیش از حد ریسک نکنید
   - قوانین prop firm را رعایت کنید

3. **لاگ‌ها را بررسی کنید**
   - مرتباً لاگ‌ها را چک کنید
   - در صورت خطا، ربات را متوقف کنید

---

## 📞 پشتیبانی

اگر مشکلی دارید:
1. لاگ‌ها را بررسی کنید
2. PDF استراتژی را دوباره بخوانید
3. کد را debug کنید

---

## 📄 مجوز

این پروژه برای استفاده شخصی و آموزشی است.

**نویسنده**: Hamid Tabasi (HTU)
**بازنویسی شده با**: Claude AI
**تاریخ**: 2024-11-09
