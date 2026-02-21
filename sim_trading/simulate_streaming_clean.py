"""\
Streaming Simulation — Live Paper Trading Simulator

PURPOSE: Simulate EXACTLY what would happen in live paper trading.
This is NOT a pipeline replay. There is no label filtering, no train/test
split awareness. Every bar is processed as it arrives, indicators are
computed in a streaming fashion, the model predicts, and we trade if the
signal fires. This is the ground truth for what a deployed model would
actually produce.

Setup filter (--setup-filter):
  Must match how the model was trained in master_pipeline.py.
  - If the pipeline was run with --setup-filter, the model only saw bars
    passing the filter during training. Use --setup-filter here too so
    the simulator only trades on the same kind of bars the model learned.
  - If the pipeline was run WITHOUT --setup-filter (the default), the model
    saw ALL bars. Do NOT use --setup-filter here — the model expects to
    score every bar, and filtering would skip bars the model is designed to
    handle (by predicting low probability on non-setup bars).
  In both cases the simulator matches the pipeline: same filter at training
  time = same filter at inference time.

Key design principles (live paper trading):
  - Bar-by-bar processing: each bar is seen once, in order, no future data
  - Streaming indicators: day-window computation matching live inference
    (vol_pct_complete=1.0, no end-of-day lookahead)
  - NO label validity filter: in live trading there are no labels
  - Entry at close of signal bar
  - Fixed costs: 2 × (commission + slippage) × shares per round-trip
  - Stop/target evaluation starts on the bar AFTER entry
  - Conservative ordering: stop checked before target within a bar
  - IBKR margin interest on borrowed capital
  - Flat-to-flat: skip signals while a position is open
  - EOD exit: close at close of last bar of entry day (intraday only)
  - Risk-based position sizing: shares = (capital × risk_pct) / (stop_atr × ATR)
    with probability scaling (30% at threshold → 100% at P=1.0)

Trade mechanics match master_pipeline.py's _simulate_trade_realized_path()
so that backtest P&L is comparable, but the SIGNAL GENERATION is what would
happen live — no filtering to bars that have valid labels or pass setup criteria.

Example:
  python sim_trading\\simulate_streaming_clean.py --year 2024 --stop-atr 0.5 --risk-pct 0.01
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd

from sim_trading.streaming_indicators_aligned import StreamingIndicatorsAligned
from sim_trading.streaming_simulator import StreamingSimulator
from src.model_selector import load_model_for_stop

# ── Cost constants (must match master_pipeline.py exactly) ──────────────
COMMISSION_PER_SHARE = 0.005   # $0.005/share
SLIPPAGE_PER_SHARE = 0.01      # $0.01/share
CAPITAL_CAP = 1_000_000        # $1M
DEFAULT_RISK_PER_TRADE = 0.01  # 1% of capital risked per trade
SHARES_PER_TRADE = 100         # fallback if risk-based sizing disabled

# IBKR margin tiers (same as master_pipeline.py)
IBKR_MARGIN_RATE_TIERS = [
    (100_000,       0.0683),
    (1_000_000,     0.0633),
    (50_000_000,    0.0608),
    (200_000_000,   0.0583),
    (float('inf'),  0.0558),
]
IBKR_DAYS_PER_YEAR = 360


def ibkr_margin_cost(borrowed: float, hold_hours: float) -> float:
    """Compute IBKR tiered margin interest for a given borrowed amount and hold duration.

    Identical to master_pipeline.py — prorates each tier's annual rate to the
    actual hold duration: tier_amount * rate * hold_hours / (360 * 24).

    Args:
        borrowed: Dollar amount borrowed on margin (notional - cash). Must be >= 0.
        hold_hours: Duration the position is held, in hours.

    Returns:
        Interest cost in dollars (always >= 0).
    """
    if borrowed <= 0 or hold_hours <= 0:
        return 0.0
    remaining = borrowed
    total_interest = 0.0
    prev_bound = 0.0
    for upper_bound, rate in IBKR_MARGIN_RATE_TIERS:
        tier_capacity = upper_bound - prev_bound
        tier_amount = min(remaining, tier_capacity)
        if tier_amount <= 0:
            break
        # annual_rate / 360 days / 24 hours * hold_hours
        total_interest += tier_amount * rate * hold_hours / (IBKR_DAYS_PER_YEAR * 24.0)
        remaining -= tier_amount
        prev_bound = upper_bound
    return total_interest


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if "time" in df.columns and "datetime" not in df.columns:
        df["datetime"] = pd.to_datetime(df["time"], utc=True)
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


# ── SETUP_DEFAULTS (identical to master_pipeline.py) ────────────────────
SETUP_DEFAULTS = {
    'min_rr': 0.3,
    'min_minutes_session': 15,
    'max_minutes_session': 360,
}


def passes_setup_filter(indicators: dict, stop_atr: float) -> bool:
    """Check if a bar passes the setup filter (matches master_pipeline exactly)."""
    vwap_width_atr = float(indicators.get("vwap_width_atr", 0.0) or 0.0)
    minutes_into_session = float(indicators.get("minutes_into_session", 0.0) or 0.0)

    if minutes_into_session < SETUP_DEFAULTS['min_minutes_session']:
        return False
    if minutes_into_session > SETUP_DEFAULTS['max_minutes_session']:
        return False
    if stop_atr > 0 and (vwap_width_atr / stop_atr) < SETUP_DEFAULTS['min_rr']:
        return False
    return True


def compute_shares(prob: float, threshold: float, atr: float, stop_atr: float,
                   risk_pct: float, min_frac: float, capital: float,
                   entry_price: float = 0.0) -> int:
    """Compute risk-based, probability-scaled shares with notional cap.

    Formula from master_pipeline.py prob_weighted mode:
      raw_scale = clip((prob - threshold) / (1 - threshold), 0, 1)
      scale_frac = min_frac + raw_scale * (1 - min_frac)
      risk_dollars = capital * risk_pct
      stop_risk_per_share = stop_atr * atr
      full_shares = floor(risk_dollars / stop_risk_per_share)
      shares = round(full_shares * scale_frac), clipped [1, 9999]

    Capital constraint (live-realistic):
      max_shares = floor(capital / entry_price)
      shares = min(shares, max_shares)
      Ensures notional (shares × price) never exceeds available capital.
      The pipeline omits this, but in live trading you can't buy more than
      your capital allows (without margin — and we charge margin separately).
    """
    prob_range = 1.0 - threshold
    if prob_range <= 0:
        prob_range = 0.01
    raw_scale = (prob - threshold) / prob_range
    raw_scale = max(0.0, min(1.0, raw_scale))

    scale_frac = min_frac + raw_scale * (1.0 - min_frac)

    risk_dollars = capital * risk_pct
    stop_risk_per_share = stop_atr * atr
    if stop_risk_per_share <= 0:
        stop_risk_per_share = 1.0
    full_shares = int(np.floor(risk_dollars / stop_risk_per_share))
    shares = int(np.round(full_shares * scale_frac))

    # Cap notional to capital: shares × entry_price ≤ capital
    if entry_price > 0:
        max_shares_by_capital = int(np.floor(capital / entry_price))
        shares = min(shares, max_shares_by_capital)

    shares = max(1, min(9999, shares))
    return shares


def main() -> None:
    print("STARTING SIMULATION...", flush=True)
    p = argparse.ArgumentParser(description="Streaming simulation (pipeline-identical logic)")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--rf-threshold", type=float, default=0.5)
    p.add_argument("--stop-atr", type=float, default=0.5)
    p.add_argument("--warmup-bars", type=int, default=60,
                   help="Warmup bars before prior-day open (day-window mode, matches precompute)")
    p.add_argument("--no-day-window", action="store_true",
                   help="Disable day-aware windowing; use legacy deque(maxlen=lookback)")
    p.add_argument("--lookback", type=int, default=200,
                   help="Legacy rolling-deque lookback (only used if --no-day-window)")
    p.add_argument("--data", type=str, default="data/tsla_5min_10years.csv")
    p.add_argument("--models-dir", type=str, default="models")
    p.add_argument("--model-file", type=str, default=None,
                   help="Path to a specific .pkl model file. Skips slow model scanning.")
    # ── Risk-based sizing args (match master_pipeline prob_weighted mode) ──
    p.add_argument("--risk-pct", type=float, default=0.01,
                   help="Fraction of capital risked per trade (default: 0.01 = 1%%)")
    p.add_argument("--prob-min-frac", type=float, default=0.3,
                   help="Minimum scaling fraction at threshold probability (default: 0.3)")
    p.add_argument("--fixed-shares", type=int, default=None,
                   help="Use fixed shares instead of risk-based sizing (overrides --risk-pct)")
    p.add_argument("--setup-filter", action="store_true",
                   help="Enable setup filter (must match pipeline training: use only if "
                        "model was trained with --setup-filter)")
    p.add_argument("--indicators-file", type=str, default=None,
                   help="Pre-computed indicators parquet (same file used by pipeline). "
                        "Bypasses on-the-fly indicator computation for exact match.")
    args = p.parse_args()

    use_day_window = not args.no_day_window
    stop_atr = args.stop_atr
    use_risk_sizing = args.fixed_shares is None
    fixed_shares = args.fixed_shares
    use_setup_filter = args.setup_filter
    use_precomputed = args.indicators_file is not None

    # ── Load data ───────────────────────────────────────────────────────
    df = load_data(args.data)
    df["year"] = df["datetime"].dt.year

    if use_day_window:
        df_full = df[df["year"].isin([args.year - 1, args.year])].reset_index(drop=True)
        target_mask = df_full["year"] == args.year
        if not target_mask.any():
            raise SystemExit(f"No data found for year {args.year}")
        target_start_iloc = int(target_mask.idxmax())
        print(f"  Day-window mode: {len(df_full):,} total bars, "
              f"target starts at iloc {target_start_iloc} "
              f"({df_full.iloc[target_start_iloc]['datetime']})",
              file=sys.stderr, flush=True)
    else:
        df_full = df[df["year"] == args.year].reset_index(drop=True)
        target_start_iloc = 0
        if df_full.empty:
            raise SystemExit(f"No data found for year {args.year}")

    # ── Load model ──────────────────────────────────────────────────────
    if args.model_file:
        from src.model_persistence import load_model
        model, metadata = load_model(args.model_file)
    else:
        model, metadata = load_model_for_stop(stop_atr=stop_atr, models_dir=args.models_dir, latest=True)
    feature_cols = metadata["features"]
    print(f"  Model: {len(feature_cols)} features, stop_atr={metadata.get('stop_atr','?')}",
          file=sys.stderr, flush=True)

    # ── Build simulator (for indicator computation only) ────────────────
    sim = StreamingSimulator(
        initial_capital=CAPITAL_CAP,
        bar_interval_minutes=5,
        lookback_bars=args.lookback,
        warmup_bars=args.warmup_bars,
        use_day_window=use_day_window,
        verbose=False,  # we do our own logging
    )
    indicator_calc = StreamingIndicatorsAligned(verbose=False)

    # ── Prepare data ────────────────────────────────────────────────────
    if "date" not in df_full.columns:
        df_full["date"] = pd.to_datetime(df_full["datetime"]).dt.date
    df_full = df_full.reset_index(drop=True)

    # Build day index for day-window indicator computation
    if use_day_window:
        unique_dates, day_ranges, date_to_order = sim._build_day_index(df_full)
        if hasattr(indicator_calc, "calculate_core_indicators"):
            _core_calc_fn = indicator_calc.calculate_core_indicators
        else:
            _core_calc_fn = None

    # Day indicator cache (lazy, same as simulator's run())
    day_indicator_cache = {}   # date -> pd.DataFrame
    day_window_offset_cache = {}  # date -> window_start

    def _ensure_day_cached(bar_date):
        if bar_date in day_indicator_cache:
            return
        if _core_calc_fn is None:
            return
        order = date_to_order[bar_date]
        if order >= 1:
            prior_date = unique_dates[order - 1]
            prior_start, _ = day_ranges[prior_date]
            w_start = max(0, prior_start - args.warmup_bars)
        else:
            cur_start, _ = day_ranges[bar_date]
            w_start = max(0, cur_start - args.warmup_bars)
        _, target_day_end = day_ranges[bar_date]
        w_end = target_day_end
        if w_end - w_start < 20:
            return
        window_df = df_full.iloc[w_start:w_end].copy().reset_index(drop=True)
        if "date" not in window_df.columns:
            window_df["date"] = pd.to_datetime(window_df["datetime"]).dt.date
        try:
            result_df = _core_calc_fn(window_df, verbose=False)
        except Exception:
            return
        day_indicator_cache[bar_date] = result_df
        day_window_offset_cache[bar_date] = w_start
        print(f"  Day {len(day_indicator_cache):>3d}: computed indicators for {bar_date} "
              f"(window={len(window_df)} bars)",
              file=sys.stderr, flush=True)

    def get_indicators(iloc_idx, bar_date):
        """Get indicators for a specific bar using cached day computation."""
        _ensure_day_cached(bar_date)
        cached_df = day_indicator_cache.get(bar_date)
        if cached_df is None:
            return {}
        window_start = day_window_offset_cache[bar_date]
        row_offset = iloc_idx - window_start
        if row_offset < 0 or row_offset >= len(cached_df):
            return {}
        ind_row = cached_df.iloc[row_offset]
        indicators = {}
        for col in cached_df.columns:
            val = ind_row[col]
            if isinstance(val, (np.integer,)):
                indicators[col] = int(val)
            elif isinstance(val, (np.floating,)):
                indicators[col] = float(val)
            else:
                indicators[col] = val
        indicators['vol_pct_complete'] = 1.0
        return indicators    # ── Simulation state ────────────────────────────────────────────────
    trades = []        # completed trade dicts
    position = None    # None or dict with entry info
    # position = {
    #   'entry_price': float, 'entry_atr': float, 'entry_vwap': float,
    #   'is_long': bool, 'entry_date': date, 'entry_dt': Timestamp,
    #   'stop_price': float, 'entry_iloc': int, 'shares': int,
    #   'prob': float,
    # }

    # flat_until_dt: after a trade exits, no new entry allowed on any bar
    # with datetime <= flat_until_dt.  Matches pipeline's flat_time logic in
    # simulate_trade_level_pnl():  ``if flat_time is not None and entry_dt <= flat_time: continue``
    flat_until_dt = None

    stats = {
        'bars_checked': 0,
        'missing_indicators': 0,
        'not_setup': 0,
        'not_is_long_setup': 0,
        'missing_features': 0,
        'low_prob': 0,
        'in_position': 0,
        'entries': 0,
    }

    total_target_bars = len(df_full) - target_start_iloc

    # Sizing info for header
    if use_risk_sizing:
        risk_dollars = CAPITAL_CAP * args.risk_pct
        sizing_desc = (f"Risk-based: {args.risk_pct:.1%} of ${CAPITAL_CAP:,.0f} = "
                       f"${risk_dollars:,.0f}/trade, prob-scaled [{args.prob_min_frac:.0%}–100%]")
    else:
        sizing_desc = f"Fixed: {fixed_shares} shares/trade"

    print(f"\n{'='*80}", file=sys.stderr, flush=True)
    print(f"LIVE PAPER TRADING SIMULATOR (bar-by-bar, no lookahead)", file=sys.stderr, flush=True)
    print(f"{'='*80}", file=sys.stderr, flush=True)
    print(f"  Capital: ${CAPITAL_CAP:,.0f}", file=sys.stderr, flush=True)
    print(f"  Sizing: {sizing_desc}", file=sys.stderr, flush=True)
    print(f"  Stop: {stop_atr} ATR", file=sys.stderr, flush=True)
    print(f"  RF threshold: {args.rf_threshold}", file=sys.stderr, flush=True)
    print(f"  Setup filter: {'ON' if use_setup_filter else 'OFF (live mode)'}", file=sys.stderr, flush=True)
    print(f"  Target bars: {total_target_bars:,}", file=sys.stderr, flush=True)
    print(f"{'='*80}\n", file=sys.stderr, flush=True)

    # ── Build last-bar-of-day index for proper EOD exit ────────────────
    # Pipeline exits at close of last bar of entry day (not open of next day)
    last_bar_of_day = {}  # date -> iloc index of last bar
    for i in range(len(df_full)):
        d = df_full.iloc[i]["date"]
        last_bar_of_day[d] = i  # keeps overwriting, so final value is last bar

    # ── Main bar loop ───────────────────────────────────────────────────
    for iloc_idx in range(target_start_iloc, len(df_full)):
        row = df_full.iloc[iloc_idx]
        bar = row.to_dict()
        bar_dt = pd.Timestamp(bar["datetime"])
        bar_date = row["date"]
        stats['bars_checked'] += 1        # Get indicators
        indicators = get_indicators(iloc_idx, bar_date)
        if not indicators or "atr" not in indicators:
            stats['missing_indicators'] += 1
            # Still need to check EOD exit even without indicators
            if position is not None and position['entry_date'] != bar_date:
                # Exit at close of last bar of entry day
                eod_iloc = last_bar_of_day.get(position['entry_date'], position['entry_iloc'])
                eod_row = df_full.iloc[eod_iloc]
                exit_price = float(eod_row["close"])
                exit_dt = pd.Timestamp(eod_row["datetime"])
                _close_trade(position, exit_price, exit_dt, 'eod', trades)
                flat_until_dt = exit_dt
                position = None
            continue

        atr = float(indicators.get("atr", 0.0) or 0.0)        # ── EOD EXIT: close position if entry_date != today ─────────
        if position is not None and position['entry_date'] != bar_date:
            # Pipeline: exit at close of last bar of entry day
            eod_iloc = last_bar_of_day.get(position['entry_date'], position['entry_iloc'])
            eod_row = df_full.iloc[eod_iloc]
            exit_price = float(eod_row["close"])
            exit_dt = pd.Timestamp(eod_row["datetime"])
            _close_trade(position, exit_price, exit_dt, 'eod', trades)
            flat_until_dt = exit_dt
            position = None

        # ── STOP / TARGET EXIT (starts bar AFTER entry) ─────────────
        if position is not None and iloc_idx > position['entry_iloc']:
            hi = float(bar["high"])
            lo = float(bar["low"])
            is_long = position['is_long']
            stop_price = position['stop_price']
            target_price = position['entry_vwap']            # Conservative: stop checked FIRST (matches pipeline)
            if is_long:
                if lo <= stop_price:
                    _close_trade(position, stop_price, bar_dt, 'stop', trades)
                    flat_until_dt = bar_dt
                    position = None
                elif hi >= target_price:
                    _close_trade(position, target_price, bar_dt, 'vwap', trades)
                    flat_until_dt = bar_dt
                    position = None
            else:
                if hi >= stop_price:
                    _close_trade(position, stop_price, bar_dt, 'stop', trades)
                    flat_until_dt = bar_dt
                    position = None
                elif lo <= target_price:
                    _close_trade(position, target_price, bar_dt, 'vwap', trades)
                    flat_until_dt = bar_dt
                    position = None        # ── ENTRY LOGIC (only when flat) ────────────────────────────
        if position is not None:
            stats['in_position'] += 1
            continue  # flat-to-flat: skip signals while in a trade

        # flat_until_dt gate: pipeline skips signals where entry_dt <= flat_time
        # This prevents re-entry on the same bar where a trade just exited
        if flat_until_dt is not None and bar_dt <= flat_until_dt:
            stats['in_position'] += 1
            continue

        # Direction: is_long_setup determines trade direction (NOT a filter)
        # Pipeline trades both longs AND shorts — is_long_setup sets direction only
        is_long_setup = indicators.get("is_long_setup", None)
        if is_long_setup is None:
            stats['not_is_long_setup'] += 1
            continue

        # Setup filter: must match pipeline training. If model was trained with
        # --setup-filter, enable here too. If not (default), leave OFF.
        if use_setup_filter and not passes_setup_filter(indicators, stop_atr):
            stats['not_setup'] += 1
            continue

        # Feature vector
        feature_vector = [indicators.get(c) for c in feature_cols]
        if any(pd.isna(x) for x in feature_vector):
            stats['missing_features'] += 1
            continue

        # Model prediction
        prob = model.predict_proba([feature_vector])[0, 1]
        if prob < args.rf_threshold:
            stats['low_prob'] += 1
            continue        # ── ENTER TRADE (at close of this bar — matching pipeline) ──
        stats['entries'] += 1
        entry_price = float(bar["close"])

        # ── COMPUTE SHARES (needs entry_price for notional cap) ─────
        if use_risk_sizing:
            trade_shares = compute_shares(
                prob=prob,
                threshold=args.rf_threshold,
                atr=atr,
                stop_atr=stop_atr,
                risk_pct=args.risk_pct,
                min_frac=args.prob_min_frac,
                capital=float(CAPITAL_CAP),
                entry_price=entry_price,
            )
        else:
            trade_shares = fixed_shares
        entry_atr = atr
        entry_vwap = float(indicators.get("vwap", entry_price))

        if is_long_setup:
            stop_price = entry_price - (stop_atr * entry_atr)
        else:
            stop_price = entry_price + (stop_atr * entry_atr)

        position = {
            'entry_price': entry_price,
            'entry_atr': entry_atr,
            'entry_vwap': entry_vwap,
            'is_long': bool(is_long_setup),
            'entry_date': bar_date,
            'entry_dt': bar_dt,
            'stop_price': stop_price,
            'entry_iloc': iloc_idx,
            'shares': trade_shares,
            'prob': prob,
        }

        if stats['entries'] % 100 == 1:
            print(f"  [ENTRY #{stats['entries']}] {bar_dt} | "
                  f"{'LONG' if is_long_setup else 'SHORT'} @ ${entry_price:.2f} | "
                  f"stop=${stop_price:.2f} | target_vwap=${entry_vwap:.2f} | "
                  f"prob={prob:.3f} | shares={trade_shares}",
                  file=sys.stderr, flush=True)

        # ── Progress ────────────────────────────────────────────────
        target_processed = iloc_idx - target_start_iloc + 1
        if target_processed % 1000 == 0:
            n_trades = len(trades)
            total_pnl = sum(t['net_pnl'] for t in trades)
            print(f"  Bar {target_processed:,}/{total_target_bars:,} | "
                  f"Trades: {n_trades} | P&L: ${total_pnl:+,.2f} | "
                  f"Days cached: {len(day_indicator_cache)}",
                  file=sys.stderr, flush=True)

        # Evict old days from cache
        if len(day_indicator_cache) > 2:
            for cached_date in list(day_indicator_cache.keys()):
                if cached_date < bar_date:
                    order_cur = date_to_order.get(bar_date, 0)
                    order_cached = date_to_order.get(cached_date, 0)
                    if order_cur - order_cached > 1:
                        del day_indicator_cache[cached_date]
                        del day_window_offset_cache[cached_date]

    # ── Close any remaining position at simulation end ──────────────
    if position is not None:
        last_bar = df_full.iloc[-1]
        exit_price = float(last_bar["close"])
        _close_trade(position, exit_price, pd.Timestamp(last_bar["datetime"]), 'eod_final', trades)
        position = None

    # ── Results ─────────────────────────────────────────────────────
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    print(f"\n{'='*80}", file=sys.stderr, flush=True)
    print(f"SIMULATION COMPLETE", file=sys.stderr, flush=True)
    print(f"{'='*80}", file=sys.stderr, flush=True)
    print(f"  Bars checked: {stats['bars_checked']:,}", file=sys.stderr, flush=True)
    print(f"  Missing indicators: {stats['missing_indicators']:,}", file=sys.stderr, flush=True)
    print(f"  Missing is_long_setup: {stats['not_is_long_setup']:,}", file=sys.stderr, flush=True)
    print(f"  Failed setup filter: {stats['not_setup']:,}", file=sys.stderr, flush=True)
    print(f"  Missing features: {stats['missing_features']:,}", file=sys.stderr, flush=True)
    print(f"  Low probability: {stats['low_prob']:,}", file=sys.stderr, flush=True)
    print(f"  In position (skipped): {stats['in_position']:,}", file=sys.stderr, flush=True)
    print(f"  Entries: {stats['entries']:,}", file=sys.stderr, flush=True)

    if len(trades_df) > 0:
        total_gross = trades_df['gross_pnl'].sum()
        total_net = trades_df['net_pnl'].sum()
        total_costs = trades_df['costs'].sum()
        total_margin = trades_df['margin_cost'].sum()
        n_trades = len(trades_df)
        n_wins = (trades_df['net_pnl'] > 0).sum()
        win_rate = n_wins / n_trades
        avg_net = trades_df['net_pnl'].mean()
        vwap_rate = (trades_df['exit_reason'] == 'vwap').mean()
        avg_shares = trades_df['shares'].mean()
        avg_risk = trades_df['risk_dollars'].mean()

        print(f"\n  Total trades: {n_trades:,}", file=sys.stderr, flush=True)
        print(f"  Win rate: {win_rate*100:.1f}%", file=sys.stderr, flush=True)
        print(f"  VWAP touch rate: {vwap_rate*100:.1f}%", file=sys.stderr, flush=True)
        print(f"  Avg shares/trade: {avg_shares:,.0f} "
              f"(range: {int(trades_df['shares'].min())}-{int(trades_df['shares'].max())})",
              file=sys.stderr, flush=True)
        print(f"  Avg risk$/trade: ${avg_risk:,.2f}", file=sys.stderr, flush=True)
        print(f"  Gross P&L: ${total_gross:+,.2f}", file=sys.stderr, flush=True)
        print(f"  Total costs: ${total_costs:,.2f}", file=sys.stderr, flush=True)
        print(f"  Total margin interest: ${total_margin:,.2f}", file=sys.stderr, flush=True)
        print(f"  Net P&L: ${total_net:+,.2f}", file=sys.stderr, flush=True)
        print(f"  Avg net P&L/trade: ${avg_net:+,.2f}", file=sys.stderr, flush=True)
        print(f"  Return on capital: {total_net/CAPITAL_CAP*100:.2f}%", file=sys.stderr, flush=True)

        # Exit reason breakdown
        print(f"\n  Exit reasons:", file=sys.stderr, flush=True)
        for reason, grp in trades_df.groupby('exit_reason'):
            print(f"    {reason}: {len(grp)} trades, ${grp['net_pnl'].sum():+,.2f}",
                  file=sys.stderr, flush=True)

        # Save trade log
        out_path = Path("data") / "streaming_sim_trades.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(out_path, index=False)
        print(f"\n  Trade log saved: {out_path}", file=sys.stderr, flush=True)
    else:
        print("  NO TRADES EXECUTED", file=sys.stderr, flush=True)

    print(f"{'='*80}\n", file=sys.stderr, flush=True)


def _close_trade(position: dict, exit_price: float, exit_dt, exit_reason: str,
                 trades: list, available_capital: float = CAPITAL_CAP):
    """Close a trade and append to trades list (matches pipeline exactly).

    Uses variable shares from position dict (risk-based or fixed).
    Costs computed per-trade: 2 × (commission + slippage) × shares.

    Args:
        available_capital: Cash capital available at entry time for this trade.
            Notional exceeding this incurs IBKR margin interest.
            Matches pipeline's remaining_capital = capital - open_notional.
    """
    entry_price = position['entry_price']
    is_long = position['is_long']
    shares = position['shares']

    # Gross P&L
    if is_long:
        gross = (exit_price - entry_price) * shares
    else:
        gross = (entry_price - exit_price) * shares

    # Per-trade costs (variable shares)
    costs = 2.0 * (COMMISSION_PER_SHARE + SLIPPAGE_PER_SHARE) * shares

    # IBKR margin cost (matches pipeline: borrowed = notional - available_capital)
    notional = entry_price * shares
    borrowed = max(0.0, notional - float(available_capital))
    entry_dt = position['entry_dt']
    hold_td = exit_dt - entry_dt
    hold_hours = max(hold_td.total_seconds() / 3600.0, 0.0)
    # Minimum 5 minutes (1 bar) even when entry == exit bar (matches pipeline)
    if hold_hours < (5.0 / 60.0):
        hold_hours = 5.0 / 60.0
    margin_cost = ibkr_margin_cost(borrowed, hold_hours)

    # Risk dollars (same as pipeline: stop_dist × shares)
    stop_dist = abs(entry_price - position['stop_price'])
    risk_dollars = stop_dist * shares

    net = gross - costs - margin_cost

    trades.append({
        'entry_datetime': entry_dt,
        'exit_datetime': exit_dt,
        'is_long': int(is_long),
        'entry_price': entry_price,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'gross_pnl': gross,
        'costs': costs,
        'margin_cost': margin_cost,
        'net_pnl': net,
        'entry_atr': position['entry_atr'],
        'entry_vwap': position['entry_vwap'],
        'stop_price': position['stop_price'],        'notional': notional,
        'margin_borrowed': borrowed,
        'hold_hours': hold_hours,
        'shares': shares,
        'risk_dollars': risk_dollars,
        'prob': position.get('prob', float('nan')),
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        raise
