"""
Trading Configuration

All constants, IBKR connection defaults, cost parameters, and risk limits
in one place.  CLI args override these at runtime.
"""

from __future__ import annotations

# ── IBKR Connection ─────────────────────────────────────────────────────
PAPER_PORT = 7497          # TWS / Gateway paper-trading port
LIVE_PORT  = 7496          # TWS / Gateway live-trading port
DEFAULT_HOST = "127.0.0.1"
DEFAULT_CLIENT_ID = 10     # avoid colliding with other scripts

# ── Capital & Risk ──────────────────────────────────────────────────────
CAPITAL          = 1_000_000   # $1M
RISK_PER_TRADE   = 0.01        # 1% of capital risked per trade
PROB_THRESHOLD   = 0.50        # minimum model probability to enter
PROB_SCALE_MIN   = 0.30        # probability-scaled share floor (30%)
MAX_CONCURRENT   = 1           # max simultaneous positions
MAX_POSITION_PCT = 0.50        # max notional as % of capital (50% = $500K at $1M capital)

# ── Costs ───────────────────────────────────────────────────────────────
COMMISSION_PER_SHARE = 0.005   # IBKR Pro US equities
SLIPPAGE_PER_SHARE   = 0.01    # assumption used in backtest (MKT orders saw $0.10-0.46 live!)
#                               # With LMT entries (ENTRY_LIMIT_BUFFER=$0.10) live slippage
#                               # should stay within $0.02-0.05/share on normal opens.

# ── Entry order ──────────────────────────────────────────────────────────
# Limit-order buffer added to signal close in the entry direction.
# LONG  entry: limit = close + buffer  (buy if price stays ≤ close+$0.10)
# SHORT entry: limit = close - buffer  (sell if price stays ≥ close-$0.10)
# Trades where the bar gaps more than $0.10 against us are SKIPPED — better
# to miss a trade than to chase with a market order and absorb $0.20-0.46/share.
ENTRY_LIMIT_BUFFER = 0.10      # $ beyond signal close to still accept a fill

# ── Strategy ────────────────────────────────────────────────────────────
DEFAULT_STOP_ATR  = 0.75       # stop distance in ATR multiples
DEFAULT_SYMBOL    = "TSLA"
DEFAULT_BAR_SIZE  = "5 mins"

# ── Entry quality filters ────────────────────────────────────────────────────
# Disabled: model is retrained with --slippage 0.18 so it learns to avoid
# low-target-distance setups internally — no post-hoc filter needed.
MIN_TARGET_DIST  = 0.0         # $/share minimum |entry - VWAP|  (0 = disabled)
MIN_REWARD_RISK  = 0.0         # minimum R:R ratio               (0 = disabled)

# ── Bar-size string → ib_insync realtime bar size mapping ───────────────
# ib_insync reqRealTimeBars only supports 5-second bars.
# For anything else we use reqHistoricalData with keepUpToDate=True.
# This map converts user-friendly names to IB API format.
IB_BAR_SIZES = {
    "tick":    None,           # reqMktData / reqTickByTickData
    "1 sec":   None,          # reqTickByTickData (AllLast)
    "5 secs":  "5 secs",
    "10 secs": "10 secs",
    "15 secs": "15 secs",
    "30 secs": "30 secs",
    "1 min":   "1 min",
    "2 mins":  "2 mins",
    "3 mins":  "3 mins",
    "5 mins":  "5 mins",
    "10 mins": "10 mins",
    "15 mins": "15 mins",
    "30 mins": "30 mins",
    "1 hour":  "1 hour",
}

# Aliases for CLI convenience
BAR_SIZE_ALIASES = {
    "tick":  "tick",
    "1s":    "1 sec",
    "5s":    "5 secs",
    "10s":   "10 secs",
    "15s":   "15 secs",
    "30s":   "30 secs",
    "1m":    "1 min",
    "2m":    "2 mins",
    "3m":    "3 mins",
    "5m":    "5 mins",
    "10m":   "10 mins",
    "15m":   "15 mins",
    "30m":   "30 mins",
    "1h":    "1 hour",
}
