"""Launch trading engine in paper mode, capture output, auto-stop after timeout."""
import sys, os, time, traceback, signal, threading

sys.path.insert(0, r"C:\Users\Administrator\ibkr8")
os.chdir(r"C:\Users\Administrator\ibkr8")

OUTF = r"C:\Users\Administrator\ibkr8\_paper_output.txt"
TIMEOUT = 60  # seconds — enough to connect, warm up, receive a few bars

lines = []
def log(msg):
    lines.append(msg)
    print(msg, flush=True)

def save():
    with open(OUTF, "w") as f:
        f.write("\n".join(lines))

engine_ref = [None]

def timeout_stop():
    time.sleep(TIMEOUT)
    log(f"\n[TIMEOUT] {TIMEOUT}s reached — stopping engine")
    if engine_ref[0]:
        engine_ref[0]._running = False
    save()

try:
    t0 = time.time()
    def ts(): return f"{time.time()-t0:.1f}s"

    # Start timeout thread
    t = threading.Thread(target=timeout_stop, daemon=True)
    t.start()

    log(f"[{ts()}] Loading model...")
    from src.model_persistence import load_model
    model, metadata = load_model("models/rf_vwap_stop0.75_20260218_192628.pkl")
    features = metadata.get("features", [])
    stop_atr = float(metadata.get("stop_atr", 0.75))
    log(f"[{ts()}] Model loaded: {len(features)} features, stop_atr={stop_atr}")

    log(f"[{ts()}] Creating engine...")
    from trading.engine import TradingEngine
    engine = TradingEngine(
        mode="paper",
        symbol="TSLA",
        bar_size="5 mins",
        model=model,
        features=features,
        stop_atr=stop_atr,
        threshold=0.50,
        risk_pct=0.01,
        prob_scale_min=0.30,
        capital=1_000_000,
        max_concurrent=1,
        host="127.0.0.1",
        client_id=10,
        lookback=200,
    )
    engine_ref[0] = engine
    log(f"[{ts()}] Engine created, starting...")

    engine.start()

    log(f"[{ts()}] Engine stopped normally")
except Exception as e:
    log(f"ERROR: {e}")
    log(traceback.format_exc())
finally:
    save()
