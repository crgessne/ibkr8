"""Quick smoke test for all trading module imports."""
import sys, time
t0 = time.time()

def elapsed():
    return f"{time.time()-t0:.1f}s"

print(f"[{elapsed()}] Starting...", flush=True)

from trading.config import CAPITAL, PAPER_PORT, LIVE_PORT
print(f"[{elapsed()}] config OK  capital=${CAPITAL:,.0f}", flush=True)

from trading.risk import compute_shares, ibkr_margin_cost
shares = compute_shares(0.7, 0.5, 5.0, 0.75, 0.01, 0.3, 1_000_000, 250.0)
margin = ibkr_margin_cost(50_000, 2.0)
print(f"[{elapsed()}] risk OK    shares={shares}  margin=${margin:.2f}", flush=True)

from trading.orders import OrderManager, Side
print(f"[{elapsed()}] orders OK  Side.BUY={Side.BUY}", flush=True)

from trading.strategy import VWAPReversionStrategy, Signal
print(f"[{elapsed()}] strategy OK", flush=True)

from trading.indicators import LiveIndicators
li = LiveIndicators(lookback=100)
print(f"[{elapsed()}] indicators OK (lazy, no master_pipeline yet)", flush=True)

from trading.engine import TradingEngine
print(f"[{elapsed()}] engine OK", flush=True)

from trading.runner import resolve_bar_size
print(f"[{elapsed()}] runner OK  resolve('5m')='{resolve_bar_size('5m')}'", flush=True)

# Test CLI help
sys.argv = ['test', '--help']
try:
    from trading.runner import parse_args
    parse_args()
except SystemExit:
    pass
print(f"\n[{elapsed()}] ALL IMPORTS OK", flush=True)
