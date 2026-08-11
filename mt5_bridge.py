# ============================================================================
# mt5_bridge.py — All MetaTrader 5 API calls
# Isolates every broker interaction so strategy logic stays clean.
# If MT5 API changes, only this file needs updating.
# ============================================================================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import logging
import config

log = logging.getLogger("mt5_bridge")

# Broker server UTC offset — detected once at connect(), cached here
_broker_offset_seconds: int = 0

# ── Timeframe mapping ─────────────────────────────────────────────────────────
TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}


def connect() -> bool:
    """Initialise MT5 connection. Returns True if successful."""
    if not mt5.initialize(
        path     = config.MT5_PATH,
        login    = config.MT5_LOGIN,
        password = config.MT5_PASSWORD,
        server   = config.MT5_SERVER,
    ):
        log.error(f"MT5 initialise failed: {mt5.last_error()}")
        return False

    info = mt5.account_info()
    if info is None:
        log.error(f"Cannot get account info: {mt5.last_error()}")
        return False

    # Detect broker server UTC offset once at connect time
    global _broker_offset_seconds
    import time as _time
    tick = mt5.symbol_info_tick(config.SYMBOL)
    if tick is not None:
        raw_offset = tick.time - int(_time.time())
        _broker_offset_seconds = round(raw_offset / 3600) * 3600
    else:
        _broker_offset_seconds = 0

    log.info(
        f"Connected — Account: {info.login} | "
        f"Balance: {info.balance:.2f} | "
        f"Server: {info.server} | "
        f"UTC offset: {_broker_offset_seconds//3600:+d}h"
    )
    return True


def disconnect():
    mt5.shutdown()
    log.info("MT5 disconnected")


def get_account_balance() -> float:
    info = mt5.account_info()
    return info.balance if info else 0.0


def get_account_equity() -> float:
    info = mt5.account_info()
    return info.equity if info else 0.0


def fetch_bars(symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    """
    Fetch the most recent `count` closed bars for symbol/timeframe.
    Returns DataFrame with columns: open high low close volume
    indexed by UTC datetime.
    The most recent bar (index -1) may still be forming — always work on [-2].
    """
    tf = TF_MAP.get(timeframe)
    if tf is None:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        log.warning(f"No bars returned for {symbol} {timeframe}: {mt5.last_error()}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)

    # MT5 bar timestamps are Unix seconds in broker server time (usually UTC+2/+3).
    # Use the cached offset detected once at connect() time.
    df['time'] = pd.to_datetime(df['time'] - _broker_offset_seconds, unit='s', utc=True)
    df = df.set_index('time').rename(columns={
        'open':      'open',
        'high':      'high',
        'low':       'low',
        'close':     'close',
        'tick_volume': 'volume',
    })[['open','high','low','close','volume']]

    return df


def fetch_htf_bars(symbol: str, htf: str, count: int) -> pd.DataFrame:
    """Fetch higher timeframe bars for Hull MA calculation."""
    return fetch_bars(symbol, htf, count)


def get_open_position(symbol: str):
    """
    Return the open position for symbol, or None if flat.
    Returns mt5.TradePosition namedtuple.
    """
    positions = mt5.positions_get(symbol=symbol)
    if positions and len(positions) > 0:
        return positions[0]
    return None


def get_symbol_info(symbol: str):
    """Return symbol info (point size, digits, etc.)."""
    info = mt5.symbol_info(symbol)
    if info is None:
        log.error(f"Symbol {symbol} not found. Check Market Watch.")
    return info


def calculate_lot_size(symbol: str, risk_usd: float, sl_distance_price: float) -> float:
    """
    Calculate lot size so that hitting the SL costs exactly risk_usd.

    lot_size = risk_usd / (sl_distance_price / point × tick_value)

    Parameters
    ----------
    symbol           : MT5 symbol name
    risk_usd         : fixed dollar risk e.g. 100.0
    sl_distance_price: distance from entry to SL in price units

    Returns
    -------
    float: lot size rounded to symbol's lot step
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        log.error(f"Cannot get symbol info for {symbol}")
        return 0.0

    # tick_value = profit/loss per 1 tick move per 1 lot
    # For crypto CFDs: tick_value = contract_size × tick_size × quote_currency_rate
    # MT5 provides trade_tick_value directly
    tick_value = info.trade_tick_value   # USD per tick per lot
    tick_size  = info.point              # minimum price movement

    if tick_value == 0 or tick_size == 0 or sl_distance_price == 0:
        log.warning(f"Cannot calculate lot size: tick_value={tick_value}, sl_dist={sl_distance_price}")
        return 0.0

    # How many ticks is our SL distance?
    sl_ticks = sl_distance_price / tick_size

    # Loss per lot if SL hit
    loss_per_lot = sl_ticks * tick_value

    if loss_per_lot == 0:
        return 0.0

    raw_lots = risk_usd / loss_per_lot

    # Round to allowed lot step
    lot_step = info.volume_step
    lots = round(raw_lots / lot_step) * lot_step

    # Clamp to broker min/max
    lots = max(info.volume_min, min(info.volume_max, lots))

    log.info(
        f"Lot calc: risk=${risk_usd}, SL_dist={sl_distance_price:.4f}, "
        f"loss_per_lot={loss_per_lot:.2f}, raw={raw_lots:.4f}, final={lots}"
    )
    return lots


def place_market_order(
    symbol:    str,
    direction: str,   # 'long' or 'short'
    lots:      float,
    sl_price:  float,
    tp_price:  float,
    comment:   str = "SMC-B",
) -> dict:
    """
    Place a market order with SL and TP.
    Returns dict with 'success', 'ticket', 'price', 'error'.
    """
    if config.DRY_RUN:
        log.info(f"[DRY RUN] {direction.upper()} {lots} lots {symbol} SL={sl_price:.4f} TP={tp_price:.4f}")
        return {"success": True, "ticket": -1, "price": 0.0, "error": None, "dry_run": True}

    info = mt5.symbol_info(symbol)
    if info is None:
        return {"success": False, "ticket": None, "price": 0, "error": "Symbol not found"}

    # Get current ask/bid
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"success": False, "ticket": None, "price": 0, "error": "No tick data"}

    order_type = mt5.ORDER_TYPE_BUY if direction == 'long' else mt5.ORDER_TYPE_SELL
    price      = tick.ask if direction == 'long' else tick.bid

    # Normalise SL/TP to symbol digits
    digits = info.digits
    sl_price = round(sl_price, digits)
    tp_price = round(tp_price, digits)

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lots,
        "type":         order_type,
        "price":        price,
        "sl":           sl_price,
        "tp":           tp_price,
        "deviation":    20,           # max slippage in points
        "magic":        20250519,     # bot identifier
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result is None:
        err = str(mt5.last_error())
        log.error(f"order_send returned None: {err}")
        return {"success": False, "ticket": None, "price": price, "error": err}

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"Order failed: retcode={result.retcode} comment={result.comment}")
        return {"success": False, "ticket": None, "price": price, "error": result.comment}

    log.info(f"Order filled: ticket={result.order} price={result.price:.4f}")
    return {"success": True, "ticket": result.order, "price": result.price, "error": None}


def modify_sl(symbol: str, ticket: int, new_sl: float) -> bool:
    """Modify the stop loss of an open position (used for breakeven move)."""
    if config.DRY_RUN:
        log.info(f"[DRY RUN] Modify SL ticket={ticket} new_sl={new_sl:.4f}")
        return True

    info = mt5.symbol_info(symbol)
    digits = info.digits if info else 5
    new_sl = round(new_sl, digits)

    position = mt5.positions_get(ticket=ticket)
    if not position:
        log.warning(f"Position {ticket} not found for SL modify")
        return False
    pos = position[0]

    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "symbol":   symbol,
        "sl":       new_sl,
        "tp":       pos.tp,
        "position": ticket,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        err = mt5.last_error()
        log.error(f"SL modify failed: {err}")
        return False

    log.info(f"SL moved to {new_sl:.4f} for ticket {ticket}")
    return True


def close_position(symbol: str, ticket: int) -> bool:
    """Emergency close of an open position."""
    if config.DRY_RUN:
        log.info(f"[DRY RUN] Close position ticket={ticket}")
        return True

    position = mt5.positions_get(ticket=ticket)
    if not position:
        return False
    pos = position[0]

    tick = mt5.symbol_info_tick(symbol)
    price = tick.bid if pos.type == 0 else tick.ask  # 0=BUY close at bid

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       pos.volume,
        "type":         mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
        "position":     ticket,
        "price":        price,
        "deviation":    20,
        "magic":        20250519,
        "comment":      "SMC-close",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
