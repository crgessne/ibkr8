import sys, time
t0 = time.time()
def ts(): return f"{time.time()-t0:.1f}s"

print(f"[{ts()}] 1. config...", flush=True)
from trading.config import CAPITAL
print(f"[{ts()}] 1. OK capital={CAPITAL}", flush=True)

print(f"[{ts()}] 2. risk...", flush=True)
from trading.risk import compute_shares, ibkr_margin_cost
s = compute_shares(0.7, 0.5, 5.0, 0.75, 0.01, 0.3, 1_000_000, 250.0)
m = ibkr_margin_cost(50_000, 2.0)
print(f"[{ts()}] 2. OK shares={s} margin=${m:.2f}", flush=True)

print(f"[{ts()}] 3. ib_insync...", flush=True)
from ib_insync import IB, Stock, Contract, MarketOrder, LimitOrder, StopOrder, Trade
print(f"[{ts()}] 3. OK", flush=True)

print(f"[{ts()}] 4. orders...", flush=True)
from trading.orders import OrderManager, Side
print(f"[{ts()}] 4. OK", flush=True)

print(f"[{ts()}] 5. strategy...", flush=True)
from trading.strategy import VWAPReversionStrategy, Signal
print(f"[{ts()}] 5. OK", flush=True)

print(f"[{ts()}] 6. indicators...", flush=True)
from trading.indicators import LiveIndicators
print(f"[{ts()}] 6. OK (lazy)", flush=True)

print(f"[{ts()}] 7. engine...", flush=True)
from trading.engine import TradingEngine
print(f"[{ts()}] 7. OK", flush=True)

print(f"[{ts()}] 8. runner...", flush=True)
from trading.runner import resolve_bar_size
b = resolve_bar_size("5m")
print(f"[{ts()}] 8. OK bar='{b}'", flush=True)

print(f"\n[{ts()}] ALL OK", flush=True)
