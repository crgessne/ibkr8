import sys, os
sys.path.insert(0, r"C:\Users\Administrator\ibkr8")
os.chdir(r"C:\Users\Administrator\ibkr8")

import time, traceback
OUTF = r"C:\Users\Administrator\ibkr8\_test_result.txt"
lines = []
def log(msg):
    lines.append(msg)
    print(msg, flush=True)

try:
    t0 = time.time()
    def ts(): return f"{time.time()-t0:.1f}s"

    log(f"[{ts()}] 1. config")
    from trading.config import CAPITAL
    log(f"[{ts()}] OK capital={CAPITAL}")

    log(f"[{ts()}] 2. risk")
    from trading.risk import compute_shares, ibkr_margin_cost
    s = compute_shares(0.7, 0.5, 5.0, 0.75, 0.01, 0.3, 1_000_000, 250.0)
    m = ibkr_margin_cost(50_000, 2.0)
    log(f"[{ts()}] OK shares={s} margin=${m:.2f}")

    log(f"[{ts()}] 3. ib_insync")
    from ib_insync import IB, Stock, Contract, MarketOrder, LimitOrder, StopOrder, Trade
    log(f"[{ts()}] OK")

    log(f"[{ts()}] 4. orders")
    from trading.orders import OrderManager, Side
    log(f"[{ts()}] OK")

    log(f"[{ts()}] 5. strategy")
    from trading.strategy import VWAPReversionStrategy, Signal
    log(f"[{ts()}] OK")

    log(f"[{ts()}] 6. indicators")
    from trading.indicators import LiveIndicators
    log(f"[{ts()}] OK (lazy)")

    log(f"[{ts()}] 7. engine")
    from trading.engine import TradingEngine
    log(f"[{ts()}] OK")

    log(f"[{ts()}] 8. runner")
    from trading.runner import resolve_bar_size
    b = resolve_bar_size("5m")
    log(f"[{ts()}] OK bar='{b}'")

    log(f"[{ts()}] ALL IMPORTS PASSED")
except Exception as e:
    log(f"ERROR: {e}")
    log(traceback.format_exc())
finally:
    with open(OUTF, "w") as f:
        f.write("\n".join(lines))
