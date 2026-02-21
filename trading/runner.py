"""
CLI Runner — entry point for live / paper trading.

Usage
-----
  # Paper trading, TSLA, 5-min bars (default):
  python -m trading.runner --model models/nn_pnl_stop0.75.pkl

  # Live trading, 1-min bars:
  python -m trading.runner --live --model models/nn_pnl_stop0.75.pkl --bar-size 1m

  # Paper, tick data, custom risk:
  python -m trading.runner --model models/nn_pnl_stop0.50.pkl --bar-size tick --risk-pct 0.005

  # Show help:
  python -m trading.runner --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trading.config import (
    BAR_SIZE_ALIASES,
    CAPITAL,
    DEFAULT_BAR_SIZE,
    DEFAULT_CLIENT_ID,
    DEFAULT_HOST,
    DEFAULT_STOP_ATR,
    DEFAULT_SYMBOL,
    MAX_CONCURRENT,
    MIN_REWARD_RISK,
    PROB_SCALE_MIN,
    PROB_THRESHOLD,
    RISK_PER_TRADE,
)
from trading.engine import TradingEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IBKR Trading Platform — paper & live",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Bar size shortcuts:
  tick, 1s, 5s, 10s, 15s, 30s, 1m, 2m, 3m, 5m, 10m, 15m, 30m, 1h

Examples:
  %(prog)s --model models/nn_pnl_stop0.75.pkl
  %(prog)s --live --model models/nn_pnl_stop0.75.pkl --bar-size 1m
  %(prog)s --model models/nn_pnl_stop0.50.pkl --bar-size tick --risk-pct 0.005
""",
    )

    # ── Mode ────────────────────────────────────────────────────────
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--paper", action="store_true", default=True,
        help="Paper trading (default). Connects to port 7497.",
    )
    mode_group.add_argument(
        "--live", action="store_true",
        help="LIVE trading. Connects to port 7496. Use with extreme caution.",
    )

    # ── Model ───────────────────────────────────────────────────────
    p.add_argument(
        "--model", type=str, required=True,
        help="Path to .pkl model file (e.g. models/nn_pnl_stop0.75.pkl)",
    )

    # ── Symbol & bars ───────────────────────────────────────────────
    p.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL,
                   help=f"Ticker symbol (default: {DEFAULT_SYMBOL})")
    p.add_argument("--bar-size", type=str, default="5m",
                   help="Bar size: tick, 1s, 5s, 10s, 15s, 30s, 1m, 2m, 3m, "
                        "5m (default), 10m, 15m, 30m, 1h")

    # ── Risk & sizing ──────────────────────────────────────────────
    p.add_argument("--capital", type=float, default=CAPITAL,
                   help=f"Trading capital (default: ${CAPITAL:,.0f})")
    p.add_argument("--risk-pct", type=float, default=RISK_PER_TRADE,
                   help=f"Fraction of capital risked per trade (default: {RISK_PER_TRADE})")
    p.add_argument("--stop-atr", type=float, default=DEFAULT_STOP_ATR,
                   help=f"Stop distance in ATR multiples (default: {DEFAULT_STOP_ATR})")
    p.add_argument("--threshold", type=float, default=PROB_THRESHOLD,
                   help=f"Minimum model probability to trade (default: {PROB_THRESHOLD})")
    p.add_argument("--prob-scale-min", type=float, default=PROB_SCALE_MIN,
                   help=f"Minimum probability scaling fraction (default: {PROB_SCALE_MIN})")
    p.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT,
                   help=f"Max simultaneous positions (default: {MAX_CONCURRENT})")
    p.add_argument("--min-rr", type=float, default=MIN_REWARD_RISK,
                   help=f"Minimum reward:risk ratio — skips trades where "
                        f"|entry-VWAP| / (stop_atr * ATR) < this value. "
                        f"Matches backtest min_rr. (default: {MIN_REWARD_RISK})")

    # ── Connection ─────────────────────────────────────────────────
    p.add_argument("--host", type=str, default=DEFAULT_HOST,
                   help=f"IBKR TWS/Gateway host (default: {DEFAULT_HOST})")
    p.add_argument("--client-id", type=int, default=DEFAULT_CLIENT_ID,
                   help=f"IBKR client ID (default: {DEFAULT_CLIENT_ID})")

    # ── Indicator tuning ───────────────────────────────────────────
    p.add_argument("--lookback", type=int, default=200,
                   help="Number of bars for indicator warm-up (default: 200)")

    # ── Logging ────────────────────────────────────────────────────
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging level (default: INFO)")

    return p.parse_args()


def resolve_bar_size(raw: str) -> str:
    """Convert CLI shorthand to canonical IB bar-size string."""
    raw = raw.strip().lower()
    if raw in BAR_SIZE_ALIASES:
        return BAR_SIZE_ALIASES[raw]
    # Already canonical?
    from trading.config import IB_BAR_SIZES
    if raw in IB_BAR_SIZES:
        return raw
    # Try as-is
    return raw


def main() -> None:
    args = parse_args()

    # ── Logging ─────────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Mode ────────────────────────────────────────────────────────
    mode = "live" if args.live else "paper"
    if mode == "live":
        print("\n" + "!" * 72, flush=True)
        print("  !!  LIVE TRADING MODE -- REAL MONEY AT RISK  !!", flush=True)
        print("!" * 72, flush=True)
        confirm = input("  Type 'YES' to confirm: ").strip()
        if confirm != "YES":
            print("  Aborted.", flush=True)
            sys.exit(0)

    # ── Load model ──────────────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"  [ERROR] Model file not found: {model_path}", flush=True)
        sys.exit(1)

    from src.model_persistence import load_model
    model, metadata = load_model(str(model_path))
    features = metadata.get("features", [])
    model_stop_atr = metadata.get("stop_atr", args.stop_atr)

    # Use the model's stop_atr unless user explicitly overrides
    stop_atr = args.stop_atr
    if abs(stop_atr - DEFAULT_STOP_ATR) < 0.001 and model_stop_atr:
        stop_atr = float(model_stop_atr)
        print(f"  Using model's stop_atr: {stop_atr}", flush=True)

    # ── Resolve bar size ────────────────────────────────────────────
    bar_size = resolve_bar_size(args.bar_size)

    # ── Build & run engine ──────────────────────────────────────────
    engine = TradingEngine(
        mode=mode,
        symbol=args.symbol,
        bar_size=bar_size,
        model=model,
        features=features,
        stop_atr=stop_atr,
        threshold=args.threshold,
        risk_pct=args.risk_pct,
        prob_scale_min=args.prob_scale_min,
        capital=args.capital,
        max_concurrent=args.max_concurrent,
        host=args.host,
        client_id=args.client_id,
        lookback=args.lookback,
        min_rr=args.min_rr,
    )

    engine.start()


if __name__ == "__main__":
    main()
