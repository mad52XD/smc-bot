# ============================================================================
# config.py — SMC Strategy Bot Configuration
# Edit this file to change instruments, risk, and strategy parameters.
# Never hardcode credentials here — use environment variables or MT5 login.
# ============================================================================

import os
from dotenv import load_dotenv

load_dotenv()

# ── MT5 Connection ────────────────────────────────────────────────────────────
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))  # ← your MT5 login number
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")  # ← your MT5 password
# ← your broker server e.g. "TheX-Server"
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"  # ← MT5 path

# ── Instrument ────────────────────────────────────────────────────────────────
SYMBOL = "SOLUSD"    # ← exact MT5 symbol name (check Market Watch)
TIMEFRAME = "M5"         # M5 or M15
# bars to fetch on each cycle (must cover all lookbacks)
BAR_COUNT = 600

# ── Strategy Parameters (SOL 5M optimised) ───────────────────────────────────
HULL_LENGTH = 55
EMA_LENGTH = 200
LRSI_ALPHA = 0.2
LRSI_LONG = 0.25     # LaRSI cross-above threshold for longs
LRSI_SHORT = 0.65     # LaRSI cross-below threshold for shorts
SWING_SIZE = 5        # CHoCH pivot detection window
CHOCH_WINDOW = 3        # bars CHoCH signal stays valid
FVG_WINDOW = 10       # bars to wait for Scenario B FVG
MIN_RR = 2.0      # minimum reward:risk to take a trade
SL_ATR_MULT = 1.0      # SL = SL_ATR_MULT × ATR(14)
BE_TRIGGER_R = 0.75     # move SL to BE after BE_TRIGGER_R × SL distance
ATR_PERIOD = 14

# ── Session Filter ────────────────────────────────────────────────────────────
BLOCK_START = 22     # UTC hour to block entries
BLOCK_END = 1      # UTC hour to resume entries
BLOCK_FRIDAY_PM = True   # block Friday after 15:00 Mexico City (≈21:00 UTC)
BLOCK_WEEKENDS = False  # block Saturday and Sunday

# ── Risk Management (The 5ers High Stakes) ───────────────────────────────────
ACCOUNT_SIZE = 5000  # challenge starting balance
FIXED_RISK_USD = 12.0   # fixed dollar risk per trade
MAX_DAILY_LOSS = 0.05    # 5% daily drawdown limit
MAX_TOTAL_LOSS = 0.10    # 10% total drawdown hard limit
MAX_OPEN_TRADES = 1       # no pyramiding — one trade at a time

# ── Bot Behaviour ─────────────────────────────────────────────────────────────
POLL_INTERVAL_SEC = 10    # seconds between MT5 checks within a bar
LOG_FILE = "logs/trading_log.csv"
STATE_FILE = "logs/bot_state.json"  # persists pending setup across restarts
DRY_RUN = True  # ← SET FALSE to place real orders. True = log only.
