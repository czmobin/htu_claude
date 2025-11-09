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
