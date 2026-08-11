# ============================================================================
# risk_manager.py — Prop firm rules and position sizing
# All risk checks in one place. Bot asks this before every order.
# ============================================================================

import logging
import config

log = logging.getLogger("risk_manager")


class RiskManager:
    """
    Enforces The 5ers High Stakes rules:
      - Max daily loss: 5% of starting balance
      - Max total loss: 10% of starting balance
      - Fixed dollar risk per trade
      - No more than MAX_OPEN_TRADES simultaneously
    """

    def __init__(self, starting_balance: float):
        self.starting_balance = starting_balance
        self.daily_start_balance = starting_balance  # reset each day
        self.current_day = None

    def update_daily_start(self, balance: float, today):
        """Call once per day with the opening balance."""
        if today != self.current_day:
            self.daily_start_balance = balance
            self.current_day         = today
            log.info(f"New trading day: daily start balance = {balance:.2f}")

    def can_trade(self, current_balance: float, open_positions: int) -> tuple:
        """
        Check all prop firm rules before opening a new trade.

        Returns
        -------
        (bool, str) — (allowed, reason)
        """
        # ── Max total drawdown ────────────────────────────────────────────────
        total_dd = (current_balance - self.starting_balance) / self.starting_balance
        if total_dd <= -config.MAX_TOTAL_LOSS:
            reason = (
                f"TOTAL DRAWDOWN LIMIT HIT: "
                f"{total_dd*100:.2f}% ≤ -{config.MAX_TOTAL_LOSS*100:.0f}%"
            )
            log.critical(reason)
            return False, reason

        # ── Max daily drawdown ────────────────────────────────────────────────
        daily_dd = (current_balance - self.daily_start_balance) / self.starting_balance
        if daily_dd <= -config.MAX_DAILY_LOSS:
            reason = (
                f"DAILY DRAWDOWN LIMIT HIT: "
                f"{daily_dd*100:.2f}% ≤ -{config.MAX_DAILY_LOSS*100:.0f}%"
            )
            log.warning(reason)
            return False, reason

        # ── No pyramiding ─────────────────────────────────────────────────────
        if open_positions >= config.MAX_OPEN_TRADES:
            reason = f"Already in {open_positions} trade(s) — max is {config.MAX_OPEN_TRADES}"
            log.info(reason)
            return False, reason

        return True, "OK"

    def calculate_position_size(
        self,
        entry_price: float,
        sl_price:    float,
        lot_calculator,   # callable: (symbol, risk_usd, sl_dist) → lots
    ) -> float:
        """
        Calculate lot size for fixed dollar risk.
        Delegates to mt5_bridge.calculate_lot_size for broker-specific math.
        """
        sl_distance = abs(entry_price - sl_price)
        if sl_distance == 0:
            log.warning("SL distance is zero — cannot size position")
            return 0.0

        lots = lot_calculator(config.SYMBOL, config.FIXED_RISK_USD, sl_distance)

        log.info(
            f"Position sizing: entry={entry_price:.4f} sl={sl_price:.4f} "
            f"dist={sl_distance:.4f} risk=${config.FIXED_RISK_USD} → {lots} lots"
        )
        return lots

    def calculate_tp(
        self,
        entry_price: float,
        sl_price:    float,
        direction:   str,
        swing_level: float,
    ) -> float:
        """
        Calculate TP: use swing high/low if it gives min_rr, else fixed RR.
        Mirrors the backtest TP logic exactly.
        """
        import numpy as np
        sl_dist = abs(entry_price - sl_price)
        min_tp_dist = config.MIN_RR * sl_dist

        if direction == 'long':
            if (not np.isnan(swing_level) and
                    swing_level > entry_price and
                    (swing_level - entry_price) >= min_tp_dist):
                return swing_level
            return entry_price + min_tp_dist
        else:
            if (not np.isnan(swing_level) and
                    swing_level < entry_price and
                    (entry_price - swing_level) >= min_tp_dist):
                return swing_level
            return entry_price - min_tp_dist

    def should_move_to_breakeven(
        self,
        direction:    str,
        entry_price:  float,
        sl_price:     float,
        new_bull_fvg: bool,
        new_bear_fvg: bool,
        last_bull_fvg_bottom: float,
        last_bear_fvg_top:    float,
        at_breakeven: bool,
    ) -> bool:
        """
        Arjo page 20 rule: move SL to entry when a new FVG forms
        in our direction above/below entry price.
        """
        if at_breakeven:
            return False

        import numpy as np
        if direction == 'long' and new_bull_fvg:
            if not np.isnan(last_bull_fvg_bottom) and last_bull_fvg_bottom > entry_price:
                log.info(f"BE trigger: new bull FVG at {last_bull_fvg_bottom:.4f} > entry {entry_price:.4f}")
                return True
        elif direction == 'short' and new_bear_fvg:
            if not np.isnan(last_bear_fvg_top) and last_bear_fvg_top < entry_price:
                log.info(f"BE trigger: new bear FVG at {last_bear_fvg_top:.4f} < entry {entry_price:.4f}")
                return True
        return False
