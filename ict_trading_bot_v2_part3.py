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
