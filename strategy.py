# ============================================================================
# strategy.py — All indicator calculations and signal logic
# Direct translation of the final backtested strategy.
# No MT5 calls here — receives a DataFrame, returns signals.
# ============================================================================

import pandas as pd
import numpy as np
import config


# ── Hull Moving Average ───────────────────────────────────────────────────────
def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def hull_ma(series: pd.Series, length: int) -> pd.Series:
    half = max(1, length // 2)
    sqrtn = max(1, round(np.sqrt(length)))
    return wma(2 * wma(series, half) - wma(series, length), sqrtn)


# ── Laguerre RSI ──────────────────────────────────────────────────────────────
def laguerre_rsi(close: pd.Series, alpha: float) -> pd.Series:
    gamma = 1.0 - alpha
    n = len(close)
    L0 = np.zeros(n)
    L1 = np.zeros(n)
    L2 = np.zeros(n)
    L3 = np.zeros(n)

    for i in range(1, n):
        L0[i] = (1 - gamma) * close.iloc[i] + gamma * L0[i-1]
        L1[i] = -gamma * L0[i] + L0[i-1] + gamma * L1[i-1]
        L2[i] = -gamma * L1[i] + L1[i-1] + gamma * L2[i-1]
        L3[i] = -gamma * L2[i] + L2[i-1] + gamma * L3[i-1]

    cu = (np.maximum(L0-L1, 0) + np.maximum(L1-L2, 0) + np.maximum(L2-L3, 0))
    cd = (np.maximum(L1-L0, 0) + np.maximum(L2-L1, 0) + np.maximum(L3-L2, 0))
    total = cu + cd
    with np.errstate(invalid="ignore", divide="ignore"):
        rsi = np.where(total == 0, 0.0, cu / total)
    return pd.Series(rsi, index=close.index)


# ── CHoCH Detection ───────────────────────────────────────────────────────────
def detect_choch(df: pd.DataFrame, swing_size: int = 5) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    pivot_high = np.full(n, np.nan)
    pivot_low = np.full(n, np.nan)
    choch_bull = np.zeros(n, dtype=bool)
    choch_bear = np.zeros(n, dtype=bool)

    highs = df['high'].values
    lows = df['low'].values

    ph_level = np.nan
    ph_crossed = True
    pl_level = np.nan
    pl_crossed = True
    struct_bias = 0        # +1 bullish, -1 bearish
    leg = 0        # 0=bearish leg, 1=bullish leg

    for i in range(swing_size, n):
        window_h = highs[i-swing_size:i]
        window_l = lows[i-swing_size:i]

        new_leg_high = highs[i-swing_size] > np.max(window_h[1:])
        new_leg_low = lows[i-swing_size] < np.min(window_l[1:])

        new_leg = 0 if new_leg_high else (1 if new_leg_low else leg)
        if new_leg != leg:
            leg = new_leg
            if leg == 1:   # new bullish leg → record swing low
                pl_level = lows[i-swing_size]
                pl_crossed = False
            else:          # new bearish leg → record swing high
                ph_level = highs[i-swing_size]
                ph_crossed = False

        # Bullish break
        if not ph_crossed and not np.isnan(ph_level) and df['close'].iloc[i] > ph_level:
            choch_bull[i] = (struct_bias == -1)
            ph_crossed = True
            struct_bias = 1

        # Bearish break
        if not pl_crossed and not np.isnan(pl_level) and df['close'].iloc[i] < pl_level:
            choch_bear[i] = (struct_bias == 1)
            pl_crossed = True
            struct_bias = -1

        pivot_high[i] = ph_level
        pivot_low[i] = pl_level

    df['choch_bull'] = choch_bull
    df['choch_bear'] = choch_bear
    df['swing_high'] = pivot_high
    df['swing_low'] = pivot_low
    return df


# ── Fair Value Gap ────────────────────────────────────────────────────────────
def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bar1_high = df['high'].shift(2)
    bar1_low = df['low'].shift(2)
    bar2_high = df['high'].shift(1)
    bar2_low = df['low'].shift(1)

    # Bullish FVG
    fvg_bull = bar1_high < df['low']
    df['fvg_bull'] = fvg_bull
    df['fvg_bull_bottom'] = bar1_high
    df['fvg_bull_top'] = df['low']
    df['fvg_bull_mid'] = (bar1_high + df['low']) / 2
    df['fvg_bull_breakaway'] = fvg_bull & (df['close'] > bar2_high)
    df['fvg_bull_rejection'] = fvg_bull & (df['close'] <= bar1_high)
    df['fvg_bull_perfect'] = fvg_bull & ~df['fvg_bull_breakaway'] & ~df['fvg_bull_rejection']

    # Bearish FVG
    fvg_bear = bar1_low > df['high']
    df['fvg_bear'] = fvg_bear
    df['fvg_bear_top'] = bar1_low
    df['fvg_bear_bottom'] = df['high']
    df['fvg_bear_mid'] = (bar1_low + df['high']) / 2
    df['fvg_bear_breakaway'] = fvg_bear & (df['close'] < bar2_low)
    df['fvg_bear_rejection'] = fvg_bear & (df['close'] >= bar1_low)
    df['fvg_bear_perfect'] = fvg_bear & ~df['fvg_bear_breakaway'] & ~df['fvg_bear_rejection']
    return df


def track_unmitigated_fvgs(df: pd.DataFrame) -> pd.DataFrame:
    """Track most recent unmitigated FVG at each bar. 1 tick = mitigated."""
    df = df.copy()
    n = len(df)
    lows = df['low'].values
    highs = df['high'].values

    last_bull_top = last_bull_bot = last_bull_mid = np.nan
    last_bear_top = last_bear_bot = last_bear_mid = np.nan
    new_bull = np.zeros(n, dtype=bool)
    new_bear = np.zeros(n, dtype=bool)

    arr_bull_top = np.full(n, np.nan)
    arr_bull_bot = np.full(n, np.nan)
    arr_bull_mid = np.full(n, np.nan)
    arr_bear_top = np.full(n, np.nan)
    arr_bear_bot = np.full(n, np.nan)
    arr_bear_mid = np.full(n, np.nan)

    fvg_bull = df['fvg_bull'].values
    fvg_bull_top = df['fvg_bull_top'].values
    fvg_bull_bot = df['fvg_bull_bottom'].values
    fvg_bull_mid_v = df['fvg_bull_mid'].values
    bull_tradeable = (df['fvg_bull_perfect'] | df['fvg_bull_breakaway']).values

    fvg_bear = df['fvg_bear'].values
    fvg_bear_top = df['fvg_bear_top'].values
    fvg_bear_bot = df['fvg_bear_bottom'].values
    fvg_bear_mid_v = df['fvg_bear_mid'].values
    bear_tradeable = (df['fvg_bear_perfect'] | df['fvg_bear_breakaway']).values

    for i in range(n):
        # Mitigation check
        if not np.isnan(last_bull_top) and lows[i] <= last_bull_top:
            last_bull_top = last_bull_bot = last_bull_mid = np.nan
        if not np.isnan(last_bear_bot) and highs[i] >= last_bear_bot:
            last_bear_top = last_bear_bot = last_bear_mid = np.nan

        # Register new FVGs
        if fvg_bull[i] and bull_tradeable[i]:
            last_bull_top = fvg_bull_top[i]
            last_bull_bot = fvg_bull_bot[i]
            last_bull_mid = fvg_bull_mid_v[i]
            new_bull[i] = True
        if fvg_bear[i] and bear_tradeable[i]:
            last_bear_top = fvg_bear_top[i]
            last_bear_bot = fvg_bear_bot[i]
            last_bear_mid = fvg_bear_mid_v[i]
            new_bear[i] = True

        arr_bull_top[i] = last_bull_top
        arr_bull_bot[i] = last_bull_bot
        arr_bull_mid[i] = last_bull_mid
        arr_bear_top[i] = last_bear_top
        arr_bear_bot[i] = last_bear_bot
        arr_bear_mid[i] = last_bear_mid

    df['last_bull_fvg_top'] = arr_bull_top
    df['last_bull_fvg_bottom'] = arr_bull_bot
    df['last_bull_fvg_mid'] = arr_bull_mid
    df['last_bear_fvg_top'] = arr_bear_top
    df['last_bear_fvg_bottom'] = arr_bear_bot
    df['last_bear_fvg_mid'] = arr_bear_mid
    df['new_bull_fvg'] = new_bull
    df['new_bear_fvg'] = new_bear
    return df


# ── Session Filter ────────────────────────────────────────────────────────────
def is_valid_session(dt_utc) -> bool:
    """Return True if the UTC datetime is in a valid trading session."""
    h = dt_utc.hour
    dow = dt_utc.weekday()   # 0=Mon … 4=Fri, 5=Sat, 6=Sun

    if h >= config.BLOCK_START or h < config.BLOCK_END:
        return False
    if config.BLOCK_FRIDAY_PM and dow == 4 and h >= 21:
        return False
    if config.BLOCK_WEEKENDS and dow >= 5:
        return False
    return True


# ── ATR ───────────────────────────────────────────────────────────────────────
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, prev_close = df['high'], df['low'], df['close'].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ── Main: compute all indicators on a DataFrame ───────────────────────────────
def compute_indicators(df: pd.DataFrame, df_htf: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute all indicators in one pass.

    Parameters
    ----------
    df     : 5M OHLCV bars
    df_htf : 15M bars for the HTF Hull filter (forward-filled onto 5M index)
             If None, the 15M Hull filter is disabled.

    Returns
    -------
    pd.DataFrame with all indicator columns added
    """
    df = df.copy()

    # ── 5M Hull ──────────────────────────────────────────────────────────────
    df['hull'] = hull_ma(df['close'], config.HULL_LENGTH)
    df['price_above_hull'] = df['close'] > df['hull']

    # ── 15M HTF Hull (forward-filled) ────────────────────────────────────────
    if df_htf is not None:
        hull_htf = hull_ma(df_htf['close'], config.HULL_LENGTH)
        df['hull_15m'] = hull_htf.reindex(df.index, method='ffill')
        df['above_hull_15m'] = df['close'] > df['hull_15m']
    else:
        df['above_hull_15m'] = True   # disabled — all bars pass

    # ── 200 EMA ───────────────────────────────────────────────────────────────
    df['ema200'] = df['close'].ewm(span=config.EMA_LENGTH, adjust=False).mean()
    df['above_ema200'] = df['close'] > df['ema200']

    # ── ATR ───────────────────────────────────────────────────────────────────
    df['atr'] = atr(df, config.ATR_PERIOD)

    # ── Laguerre RSI ──────────────────────────────────────────────────────────
    df['lrsi'] = laguerre_rsi(df['close'], config.LRSI_ALPHA)

    # ── CHoCH ─────────────────────────────────────────────────────────────────
    df = detect_choch(df, swing_size=config.SWING_SIZE)

    # ── FVG ───────────────────────────────────────────────────────────────────
    df = detect_fvg(df)
    df = track_unmitigated_fvgs(df)

    return df


# ── Signal check on the last closed bar ──────────────────────────────────────
def check_signals(df: pd.DataFrame) -> dict:
    """
    Evaluate signal conditions on the last CLOSED bar (iloc[-2]).
    The current bar (iloc[-1]) may still be forming — never use it.

    Returns dict with keys:
        long_signal, short_signal,
        new_bull_fvg, new_bear_fvg,
        last_bull_fvg_top/bottom/mid,
        last_bear_fvg_top/bottom/mid,
        swing_high, swing_low,
        atr, close, bar_time
    """
    bar = df.iloc[-2]   # last CLOSED bar

    # CHoCH recency
    look = df.iloc[-2-config.CHOCH_WINDOW:-2]
    choch_bull_recent = bar['choch_bull'] or look['choch_bull'].any()
    choch_bear_recent = bar['choch_bear'] or look['choch_bear'].any()

    # LRSI crosses
    prev = df.iloc[-3]
    lrsi_cross_up = bar['lrsi'] > config.LRSI_LONG and prev['lrsi'] <= config.LRSI_LONG
    lrsi_cross_down = bar['lrsi'] < config.LRSI_SHORT and prev['lrsi'] >= config.LRSI_SHORT

    # Session check on bar close time
    valid_session = is_valid_session(bar.name.to_pydatetime())

    # ── Per-bar filter diagnostics (logged every bar at INFO level) ──────────
    # This tells us exactly which condition is blocking signals.
    # Once we see consistent signals, downgrade to DEBUG to reduce log noise.
    import logging as _log
    _diag = _log.getLogger("signals")

    conditions_long = {
        "ema200": bool(bar['above_ema200']),
        "hull_15m": bool(bar['above_hull_15m']),
        "hull_5m": bool(bar['price_above_hull']),
        "choch_bull": bool(choch_bull_recent),
        "lrsi_up": bool(lrsi_cross_up),
        "session": bool(valid_session),
    }
    conditions_short = {
        "ema200": not bool(bar['above_ema200']),
        "hull_15m": not bool(bar['above_hull_15m']),
        "hull_5m": not bool(bar['price_above_hull']),
        "choch_bear": bool(choch_bear_recent),
        "lrsi_down": bool(lrsi_cross_down),
        "session": bool(valid_session),
    }

    # Log failing conditions — reduces noise by only showing what's blocking
    long_fails = [k for k, v in conditions_long.items() if not v]
    short_fails = [k for k, v in conditions_short.items() if not v]

    _diag.info(
        f"Bar {bar.name} | close={bar['close']:.4f} | "
        f"lrsi={bar['lrsi']:.3f} | atr={bar['atr']:.4f} | "
        f"LONG blocked by: {long_fails if long_fails else 'nothing — SIGNAL!'} | "
        f"SHORT blocked by: {short_fails if short_fails else 'nothing — SIGNAL!'}"
    )

    long_signal = all(conditions_long.values())
    short_signal = all(conditions_short.values())

    return {
        'long_signal': long_signal,
        'short_signal': short_signal,
        'new_bull_fvg': bool(bar['new_bull_fvg']),
        'new_bear_fvg': bool(bar['new_bear_fvg']),
        'last_bull_fvg_top': bar['last_bull_fvg_top'],
        'last_bull_fvg_bottom': bar['last_bull_fvg_bottom'],
        'last_bull_fvg_mid': bar['last_bull_fvg_mid'],
        'last_bear_fvg_top': bar['last_bear_fvg_top'],
        'last_bear_fvg_bottom': bar['last_bear_fvg_bottom'],
        'last_bear_fvg_mid': bar['last_bear_fvg_mid'],
        'swing_high': bar['swing_high'],
        'swing_low': bar['swing_low'],
        'atr': bar['atr'],
        'close': bar['close'],
        'bar_time': bar.name,
    }
