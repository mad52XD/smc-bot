# ============================================================================
# bot.py — Main trading loop
# Run this script to start the bot:
#   python bot.py
#
# The bot will:
#   1. Connect to MT5
#   2. Every 5M candle close: fetch bars, compute indicators, check signals
#   3. Manage pending Scenario B setups
#   4. Place/modify orders respecting prop firm risk rules
#   5. Log everything to logs/trading_log.csv
#
# To stop: Ctrl+C (graceful shutdown)
# ============================================================================

import time
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

import config
import strategy
import mt5_bridge as mt5b
from risk_manager import RiskManager

# ── Logging setup ─────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/bot.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("bot")


# ── Trade log ─────────────────────────────────────────────────────────────────
def log_trade(event: str, data: dict):
    """Append a row to the CSV trade log."""
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **data}
    df_row = pd.DataFrame([row])
    log_path = config.LOG_FILE
    df_row.to_csv(log_path, mode='a', header=not Path(log_path).exists(), index=False)


# ── State persistence ─────────────────────────────────────────────────────────
# Saves the pending Scenario B setup across bot restarts
def load_state() -> dict:
    if Path(config.STATE_FILE).exists():
        with open(config.STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with open(config.STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def clear_state():
    save_state({})


# ── Time helpers ──────────────────────────────────────────────────────────────
TF_MINUTES = {"M5": 5, "M15": 15, "M30": 30, "H1": 60}

def seconds_to_next_bar(tf: str) -> float:
    """Seconds until the next bar closes."""
    minutes = TF_MINUTES.get(tf, 5)
    now = datetime.now(timezone.utc)
    elapsed = (now.minute % minutes) * 60 + now.second
    remaining = minutes * 60 - elapsed
    return float(remaining)


def last_bar_time_utc(tf: str) -> datetime:
    """Return the UTC open time of the last CLOSED bar.
    Uses local clock floored to bar boundary — broker offset is handled
    in mt5_bridge.fetch_bars, so this just needs to tick on bar boundaries.
    """
    minutes = TF_MINUTES.get(tf, 5)
    now = datetime.now(timezone.utc)
    floored = now.replace(second=0, microsecond=0)
    floored = floored - timedelta(minutes=floored.minute % minutes)
    # Return PREVIOUS bar (the last fully closed one)
    return floored - timedelta(minutes=minutes)


# ── Fetch HTF bars for 15M Hull filter ───────────────────────────────────────
def get_htf_df() -> pd.DataFrame:
    """Fetch 15M bars for the HTF Hull MA filter."""
    try:
        return mt5b.fetch_htf_bars(config.SYMBOL, "M15", 200)
    except Exception as e:
        log.warning(f"Could not fetch 15M HTF bars: {e} — HTF filter disabled")
        return None


# ── Main bot class ────────────────────────────────────────────────────────────
class SMCBot:

    def __init__(self):
        self.risk_mgr      = RiskManager(config.ACCOUNT_SIZE)
        self.state         = load_state()
        self.last_bar_time = None
        self.at_breakeven  = self.state.get('at_breakeven', False)

        log.info("=" * 60)
        log.info("SMC Scenario B Bot starting")
        log.info(f"  Symbol    : {config.SYMBOL}")
        log.info(f"  Timeframe : {config.TIMEFRAME}")
        log.info(f"  Risk/trade: ${config.FIXED_RISK_USD}")
        log.info(f"  DRY RUN   : {config.DRY_RUN}")
        log.info("=" * 60)

    # ── Pending state helpers ─────────────────────────────────────────────────
    @property
    def pending(self) -> bool:
        return self.state.get('pending', False)

    def arm_pending(self, direction: str, signals: dict):
        self.state = {
            'pending'          : True,
            'direction'        : direction,
            'signal_bar'       : str(signals['bar_time']),
            'signal_bar_count' : 0,
            'ref_fvg_bottom'   : signals['last_bull_fvg_bottom'] if direction == 'long'
                                 else signals['last_bear_fvg_top'],
            'ref_fvg_top'      : signals['last_bull_fvg_top'] if direction == 'long'
                                 else signals['last_bear_fvg_top'],
            'at_breakeven'     : False,
        }
        save_state(self.state)
        log.info(f"⏳ Pending {direction.upper()} armed at bar {signals['bar_time']}")
        log_trade("SIGNAL_ARMED", {
            "direction": direction,
            "bar_time": str(signals['bar_time']),
            "ref_fvg_bottom": self.state['ref_fvg_bottom'],
            "ref_fvg_top": self.state['ref_fvg_top'],
        })

    def clear_pending(self):
        self.state = {}
        self.at_breakeven = False
        clear_state()

    # ── On each new closed bar ────────────────────────────────────────────────
    def on_bar(self, df: pd.DataFrame, df_htf: pd.DataFrame):
        """Called once per closed bar. Core strategy logic lives here."""
        now_utc = datetime.now(timezone.utc)

        # ── Update daily drawdown tracker ─────────────────────────────────────
        balance = mt5b.get_account_balance()
        equity  = mt5b.get_account_equity()
        self.risk_mgr.update_daily_start(balance, now_utc.date())

        # ── Compute indicators ────────────────────────────────────────────────
        df = strategy.compute_indicators(df, df_htf)
        if len(df) < config.EMA_LENGTH + 10:
            log.warning("Not enough bars to compute all indicators — skipping")
            return

        signals = strategy.check_signals(df)
        log.debug(
            f"Bar {signals['bar_time']} | "
            f"L={signals['long_signal']} S={signals['short_signal']} | "
            f"new_bull_fvg={signals['new_bull_fvg']} new_bear_fvg={signals['new_bear_fvg']}"
        )

        # ── Get open position ─────────────────────────────────────────────────
        position     = mt5b.get_open_position(config.SYMBOL)
        in_trade     = position is not None
        open_tickets = 1 if in_trade else 0

        # ── MANAGE OPEN TRADE ─────────────────────────────────────────────────
        if in_trade:
            direction  = 'long' if position.type == 0 else 'short'
            entry_px   = position.price_open
            current_sl = position.sl

            # Breakeven check (Arjo page 20)
            at_be = self.state.get('at_breakeven', False)
            should_be = self.risk_mgr.should_move_to_breakeven(
                direction     = direction,
                entry_price   = entry_px,
                sl_price      = current_sl,
                new_bull_fvg  = signals['new_bull_fvg'],
                new_bear_fvg  = signals['new_bear_fvg'],
                last_bull_fvg_bottom = signals['last_bull_fvg_bottom'],
                last_bear_fvg_top    = signals['last_bear_fvg_top'],
                at_breakeven  = at_be,
            )
            if should_be:
                success = mt5b.modify_sl(config.SYMBOL, position.ticket, entry_px)
                if success:
                    self.state['at_breakeven'] = True
                    save_state(self.state)
                    log_trade("BREAKEVEN", {
                        "ticket": position.ticket,
                        "entry": entry_px,
                        "new_sl": entry_px,
                        "balance": balance,
                    })
            return  # don't look for new signals while in a trade

        # ── MANAGE PENDING SETUP ──────────────────────────────────────────────
        if self.pending:
            self.state['signal_bar_count'] = self.state.get('signal_bar_count', 0) + 1
            pending_bars = self.state['signal_bar_count']

            # Cancel if window expired
            if pending_bars > config.FVG_WINDOW:
                log.info(f"⏰ Pending expired after {pending_bars} bars")
                log_trade("PENDING_EXPIRED", {"bars_waited": pending_bars})
                self.clear_pending()
                return

            direction   = self.state['direction']
            ref_bottom  = self.state.get('ref_fvg_bottom', float('nan'))
            ref_top     = self.state.get('ref_fvg_top', float('nan'))

            entered      = False
            entry_price  = None
            sl_price     = None
            tp_price     = None

            # Scenario B: new FVG forming above reference
            if direction == 'long' and signals['new_bull_fvg']:
                fvg_bot = signals['last_bull_fvg_bottom']
                above_ref = np.isnan(ref_bottom) or fvg_bot > ref_bottom
                if above_ref and not np.isnan(signals['last_bull_fvg_mid']):
                    entry_price = signals['last_bull_fvg_mid']
                    sl_price    = (float(ref_bottom) - 0.5 * signals['atr']
                                   if not np.isnan(ref_bottom)
                                   else entry_price - config.SL_ATR_MULT * signals['atr'])
                    tp_price    = self.risk_mgr.calculate_tp(
                        entry_price, sl_price, 'long', signals['swing_high'])
                    entered = True

            elif direction == 'short' and signals['new_bear_fvg']:
                fvg_top = signals['last_bear_fvg_top']
                below_ref = np.isnan(ref_top) or fvg_top < ref_top
                if below_ref and not np.isnan(signals['last_bear_fvg_mid']):
                    entry_price = signals['last_bear_fvg_mid']
                    sl_price    = (float(ref_top) + 0.5 * signals['atr']
                                   if not np.isnan(ref_top)
                                   else entry_price + config.SL_ATR_MULT * signals['atr'])
                    tp_price    = self.risk_mgr.calculate_tp(
                        entry_price, sl_price, 'short', signals['swing_low'])
                    entered = True

            if entered:
                # RR check
                sl_dist = abs(entry_price - sl_price)
                rr = abs(tp_price - entry_price) / sl_dist if sl_dist > 0 else 0
                if rr < config.MIN_RR:
                    log.info(f"RR {rr:.2f} < {config.MIN_RR} — skipping")
                    self.clear_pending()
                    return

                # Risk check
                can_trade, reason = self.risk_mgr.can_trade(balance, open_tickets)
                if not can_trade:
                    log.warning(f"Risk check failed: {reason}")
                    self.clear_pending()
                    return

                # Calculate lot size
                lots = self.risk_mgr.calculate_position_size(
                    entry_price, sl_price, mt5b.calculate_lot_size)
                if lots <= 0:
                    log.error("Lot size calculation returned 0 — skipping")
                    self.clear_pending()
                    return

                # Place order
                result = mt5b.place_market_order(
                    symbol    = config.SYMBOL,
                    direction = direction,
                    lots      = lots,
                    sl_price  = sl_price,
                    tp_price  = tp_price,
                    comment   = f"SMC-B-{direction[0].upper()}",
                )

                if result['success']:
                    log.info(
                        f"✅ {'[DRY] ' if config.DRY_RUN else ''}"
                        f"ENTRY {direction.upper()} "
                        f"lots={lots} entry≈{entry_price:.4f} "
                        f"SL={sl_price:.4f} TP={tp_price:.4f} RR={rr:.2f}"
                    )
                    log_trade("ENTRY", {
                        "direction"   : direction,
                        "entry_price" : entry_price,
                        "sl"          : sl_price,
                        "tp"          : tp_price,
                        "lots"        : lots,
                        "rr"          : round(rr, 3),
                        "ticket"      : result['ticket'],
                        "risk_usd"    : config.FIXED_RISK_USD,
                        "balance"     : balance,
                        "dry_run"     : config.DRY_RUN,
                    })
                    # Update state with trade info
                    self.state['at_breakeven'] = False
                    self.state['ticket']       = result['ticket']
                    save_state(self.state)
                else:
                    log.error(f"Order failed: {result['error']}")
                    log_trade("ORDER_FAILED", {"error": result['error'], "direction": direction})

                self.clear_pending()
            return

        # ── NEW SIGNAL — arm pending ──────────────────────────────────────────
        if signals['long_signal']:
            can_trade, reason = self.risk_mgr.can_trade(balance, open_tickets)
            if can_trade:
                self.arm_pending('long', signals)

        elif signals['short_signal']:
            can_trade, reason = self.risk_mgr.can_trade(balance, open_tickets)
            if can_trade:
                self.arm_pending('short', signals)

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        if not mt5b.connect():
            log.critical("Cannot connect to MT5 — exiting")
            sys.exit(1)

        log.info(f"Bot running. Waiting for {'DRY RUN ' if config.DRY_RUN else ''}bar closes...")

        processed_bar = None
        stale_since   = None   # wall-clock time we first saw the current bar repeated
        stale_warned  = False  # fire the warning only once per stale episode

        # Genuine stale = no new bar for 1.5× the bar period
        tf_secs      = TF_MINUTES.get(config.TIMEFRAME, 5) * 60
        stale_thresh = tf_secs * 1.5

        try:
            while True:
                try:
                        # Always fetch fresh data
                        df = mt5b.fetch_bars(config.SYMBOL, config.TIMEFRAME, config.BAR_COUNT)
                        if df.empty:
                            log.warning("Empty bar data — skipping")
                            time.sleep(config.POLL_INTERVAL_SEC)
                            continue

                        # Use the ACTUAL last closed bar time from the data
                        # iloc[-1] = still-forming bar, iloc[-2] = last closed
                        actual_bar_time = df.index[-2]

                        # Only process once per new closed bar
                        if actual_bar_time != processed_bar:
                            processed_bar = actual_bar_time
                            stale_since   = None
                            stale_warned  = False
                            log.info(f"── New bar: {actual_bar_time} ──────────────────────────────")

                            # Fetch 15M bars for HTF Hull
                            df_htf = get_htf_df()

                            # Run strategy
                            self.on_bar(df, df_htf)
                        else:
                            now = datetime.now(timezone.utc)
                            if stale_since is None:
                                stale_since = now
                            stale_secs = (now - stale_since).total_seconds()
                            if not stale_warned and stale_secs > stale_thresh:
                                log.warning(
                                    f"⚠️  MT5 data may be stalled — no new bar for "
                                    f"{stale_secs:.0f}s (expected ≤{stale_thresh:.0f}s) "
                                    f"last bar: {actual_bar_time}"
                                )
                                stale_warned = True
                            # Do not call on_bar — wait for the next new bar

                except Exception as e:
                    log.error(f"Error in on_bar: {e}", exc_info=True)

                # Sleep until near next bar close
                wait = min(seconds_to_next_bar(config.TIMEFRAME) - 2,
                           config.POLL_INTERVAL_SEC)
                time.sleep(max(1.0, wait))

        except KeyboardInterrupt:
            log.info("Keyboard interrupt — shutting down gracefully")
        finally:
            mt5b.disconnect()
            log.info("Bot stopped")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot = SMCBot()
    bot.run()
