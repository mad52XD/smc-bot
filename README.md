# SMC Scenario B Trading Bot

An algorithmic trading bot built on Smart Money Concepts (SMC) — 
specifically the Scenario B entry pattern. Designed for SOL/USD 5M 
on The 5ers prop firm challenge ($10K, +8% target).

## How It Works

Every 5 minutes on bar close:

```
1. Fetch 600 bars from MT5
2. Compute indicators — Hull MA, EMA200, Laguerre RSI, CHoCH, FVG
3. Check 6 entry conditions simultaneously
4. If signal fires → arm Scenario B pending setup
5. Wait up to 10 bars for a new unmitigated FVG above reference
6. Apply Market Regime Filter — block RANGING, reduce size in VOLATILE
7. Place order with fixed dollar risk and prop firm guardrails
8. Manage breakeven trigger on new FVG formation
9. Log entry, exit, P&L and R realised to CSV
```

## Market Regime Filter

Integrated with a trained CatBoost model that classifies the current 
market condition before every entry:

| Regime   | Action                        |
|----------|-------------------------------|
| TRENDING | Trade normally — full risk    |
| RANGING  | Block entry, wait for new setup |
| VOLATILE | Reduce position size by 50%   |

This filter was built to address zero-return months identified 
during backtesting — all correlating with ranging market conditions.

## Validated Results (BTC/USD 5M — 3 years)

| Instrument | Win Rate | Profit Factor | Max DD  | EV/trade |
|------------|----------|---------------|---------|----------|
| BTC 5M     | 51.22%   | 5.12          | -7.73%  | +0.54R   |
| ETH 5M     | 56.69%   | 3.20          | -3.94%  | +0.49R   |
| SOL 5M     | 55.97%   | 4.95          | -2.02%  | +0.71R   |
| BTC 15M    | 55.45%   | 5.59          | -3.97%  | +0.73R   |

**Deployment: SOL/USD 5M at $100 fixed risk**

## Strategy Parameters (SOL 5M optimised)

| Parameter      | Value | Description                        |
|----------------|-------|------------------------------------|
| Hull length    | 55    | Trend direction filter             |
| EMA length     | 200   | Macro trend filter                 |
| LRSI alpha     | 0.2   | Laguerre RSI smoothing             |
| LRSI long      | 0.25  | Long momentum threshold            |
| LRSI short     | 0.65  | Short momentum threshold           |
| Swing size     | 5     | CHoCH pivot detection window       |
| FVG window     | 10    | Bars to wait for Scenario B FVG    |
| Min RR         | 2.0   | Minimum reward:risk ratio          |
| Fixed risk     | $100  | Fixed dollar risk per trade        |

## File Structure

```
smc-bot/
├── bot.py            ← main loop (run this)
├── strategy.py       ← indicators and signal logic
├── mt5_bridge.py     ← all MT5 API calls
├── risk_manager.py   ← prop firm rules and position sizing
├── regime_filter.py  ← market regime ML filter
├── config.py         ← parameters (edit this first)
├── models/
│   ├── regime_model.pkl   ← trained CatBoost model
│   └── feature_cols.pkl   ← model feature list
├── requirements.txt
└── logs/
    ├── bot.log            ← full activity log
    ├── trading_log.csv    ← entries, exits, P&L, R realised
    └── bot_state.json     ← pending setup state
```

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/mad52XD/smc-bot.git
cd smc-bot
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure credentials**
```bash
# Create .env file with your MT5 credentials
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server
```

**4. Edit config.py**
```python
SYMBOL     = "SOLUSD"   # exact MT5 symbol name
DRY_RUN    = True       # set False for live trading
FIXED_RISK_USD = 100.0  # risk per trade in USD
```

**5. Run in DRY RUN mode first**
```bash
python bot.py
```

**6. Check logs/**
- `bot.log` — everything the bot did
- `trading_log.csv` — every signal, entry, exit, P&L, R realised

## Risk Management — The 5ers High Stakes Rules

```
Max daily loss:   5% of account
Max total loss:  10% of account
Max open trades:  1 (no pyramiding)
Fixed risk:      $100 per trade
```

## Key Design Decisions

**Why Python over MT5 Expert Advisor?**
Shared codebase between backtester and bot. Full data science 
tooling available. Easier debugging and testing.

**Why fixed dollar risk over % equity?**
Losses don't compound during drawdown. Protects prop firm 
daily and total limits more predictably.

**Why Scenario B over breakout entries?**
Avoids chasing breakouts. Enters on the pullback FVG which 
gives better RR and reduces slippage.

**Why SOL over BTC?**
Higher signal quality and lower drawdown at same parameters — 
confirmed across 3-year backtest.

**Why Market Regime Filter?**
Backtesting identified months of zero returns correlating with 
ranging market conditions. The ML filter automates regime 
detection and blocks entries in unfavorable conditions.

## Stopping the Bot

`Ctrl+C` — graceful shutdown, disconnects MT5, saves pending state.
