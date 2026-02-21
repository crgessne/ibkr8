"""Quick smoke test: verify all trading modules import and basic logic works."""
import sys
import time

t0 = time.time()
print("Testing trading module imports...", flush=True)

# 1) Config
from trading.config import CAPITAL, PAPER_PORT, LIVE_PORT, BAR_SIZE_ALIASES
print(f"  [OK] config        capital=${CAPITAL:,.0f}  paper={PAPER_PORT}  live={LIVE_PORT}", flush=True)

# 2) Risk
from trading.risk import compute_shares, ibkr_margin_cost, round_trip_cost
shares = compute_shares(prob=0.7, threshold=0.5, atr=5.0, stop_atr=0.75,
                        risk_pct=0.01, min_frac=0.3, capital=1_000_000, entry_price=250.0)
margin = ibkr_margin_cost(borrowed=50_000, hold_hours=2.0)
cost = round_trip_cost(shares)
print(f"  [OK] risk          shares={shares}  margin=${margin:.2f}  rt_cost=${cost:.2f}", flush=True)

# 3) Orders (import only, no IB connection)
from trading.orders import OrderManager, OrderTicket, Side
print(f"  [OK] orders        Side.BUY={Side.BUY}  Side.SELL={Side.SELL}", flush=True)

# 4) Strategy
from trading.strategy import VWAPReversionStrategy, Signal
print(f"  [OK] strategy      Signal fields: {[f.name for f in Signal.__dataclass_fields__.values()]}", flush=True)

# 5) Indicators (lazy import — doesn't load master_pipeline yet)
from trading.indicators import LiveIndicators
li = LiveIndicators(lookback=100)
print(f"  [OK] indicators    lookback={li.lookback}  bars=0", flush=True)

# 6) Engine (import only)
from trading.engine import TradingEngine
print(f"  [OK] engine        TradingEngine imported", flush=True)

# 7) Runner
from trading.runner import parse_args, resolve_bar_size
bar = resolve_bar_size("5m")
print(f"  [OK] runner        resolve_bar_size('5m') = '{bar}'", flush=True)

# 8) __main__ — skip (it calls main() on import)

elapsed = time.time() - t0
print(f"\nAll imports OK in {elapsed:.1f}s", flush=True)
