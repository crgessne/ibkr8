"""
Compare pipeline (vectorized) trades vs streaming simulation trades.

Generates the exact 446 trades from the pipeline, then runs the streaming sim
with identical parameters, and compares them side-by-side.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from model_persistence import load_model

# ── Pipeline constants (must match master_pipeline.py) ──────────────────
STOP_ATR = 0.75
TEST_YEAR = 2023
CAPITAL = 1_000_000
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_PER_SHARE = 0.01
PROB_THRESHOLD = 0.50
MIN_PROB_SCALE = 0.30
RISK_PCT = 0.01
SETUP_MIN_DIST = 0.5
SETUP_MIN_MINUTES = 15
SETUP_MAX_MINUTES = 360
SETUP_MIN_RR = 1.0          # training setup filter used min_rr_setup=1.0

MODEL_PATH = "models/rf_vwap_stop0.75_20260216_114723.pkl"
DATA_PATH = "data/tsla_5min_10years.csv"

import math

def ibkr_margin_cost(borrowed, hold_hours):
    if borrowed <= 0 or hold_hours <= 0:
        return 0.0
    tiers = [(100_000, 0.0683), (1_000_000, 0.0633), (50_000_000, 0.0608),
             (200_000_000, 0.0583), (float('inf'), 0.0558)]
    remaining = borrowed
    total = 0.0
    prev = 0.0
    for ub, rate in tiers:
        amt = min(remaining, ub - prev)
        if amt <= 0:
            break
        total += amt * rate * hold_hours / (360.0 * 24.0)
        remaining -= amt
        prev = ub
    return total


def compute_shares_risk(prob, atr, stop_atr, risk_pct, capital):
    prob_range = 1.0 - PROB_THRESHOLD
    raw_scale = min(max((prob - PROB_THRESHOLD) / prob_range, 0.0), 1.0)
    risk_dollars = capital * risk_pct
    stop_risk = stop_atr * atr
    if stop_risk <= 0:
        return 0
    full_shares = int(math.floor(risk_dollars / stop_risk))
    if full_shares <= 0:
        return 0
    scale_frac = MIN_PROB_SCALE + raw_scale * (1.0 - MIN_PROB_SCALE)
    shares = max(1, int(round(full_shares * scale_frac)))
    return min(shares, 9999)


def main():
    print("=" * 80)
    print("PIPELINE vs STREAMING SIMULATION COMPARISON")
    print("=" * 80)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    df = pd.read_csv(DATA_PATH)
    if 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['time'], utc=True)
    df['date'] = df['datetime'].dt.date
    df['year'] = df['datetime'].dt.year
    print(f"    Total bars: {len(df):,}")

    # ── Calculate indicators (full vectorized, same as pipeline) ────────
    print("\n[2] Calculating indicators (vectorized, full dataset)...")
    from master_pipeline import calculate_core_indicators
    df = calculate_core_indicators(df, verbose=False)
    print(f"    Indicators computed. Columns: {len(df.columns)}")

    # ── Generate labels ─────────────────────────────────────────────────
    print("\n[3] Generating labels for stop={STOP_ATR}...")
    from label_generator import LabelConfig, generate_labels
    config = LabelConfig(stop_atrs=[STOP_ATR])
    df = generate_labels(df, config)
    label_col = f"label_s{STOP_ATR}".replace(".", "_")
    n_valid = df[label_col].notna().sum()
    print(f"    Valid labels: {n_valid:,}")

    # ── Apply setup filter (training filter) ────────────────────────────
    print("\n[4] Applying setup filter (matches training)...")
    from master_pipeline import apply_setup_filter
    valid_mask = df[label_col].notna()
    df_valid = df[valid_mask].copy()
    setup_mask = apply_setup_filter(
        df_valid, stop_atr=STOP_ATR,
        min_dist_atr=SETUP_MIN_DIST,
        min_minutes_session=SETUP_MIN_MINUTES,
        max_minutes_session=SETUP_MAX_MINUTES,
        min_rr_setup=SETUP_MIN_RR,
    )
    df_setup = df_valid[setup_mask].copy()
    df_test_setup = df_setup[df_setup['year'] == TEST_YEAR].copy()
    print(f"    Setup bars (all years): {len(df_setup):,}")
    print(f"    Setup bars (test {TEST_YEAR}): {len(df_test_setup):,}")

    # ── Load model & score ──────────────────────────────────────────────
    print("\n[5] Loading model and scoring...")
    model, metadata = load_model(MODEL_PATH)
    feature_cols = metadata['features']
    print(f"    Features: {len(feature_cols)}")

    X_test = df_test_setup[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    probs = model.predict_proba(X_test.values)[:, 1]
    df_test_setup = df_test_setup.copy()
    df_test_setup['prob'] = probs
    df_test_setup['bar_index'] = df_test_setup.index  # original df index

    above_thresh = df_test_setup['prob'] >= PROB_THRESHOLD
    print(f"    Bars with P(win) >= {PROB_THRESHOLD}: {above_thresh.sum():,} / {len(df_test_setup):,}")

    # ── Compute per-signal shares ───────────────────────────────────────
    df_signals = df_test_setup[above_thresh].copy()
    shares_list = []
    for _, row in df_signals.iterrows():
        s = compute_shares_risk(row['prob'], row['atr'], STOP_ATR, RISK_PCT, CAPITAL)
        shares_list.append(s)
    df_signals['shares'] = shares_list

    # ── Simulate flat-to-flat (vectorized, pipeline logic) ──────────────
    print("\n[6] Simulating flat-to-flat trades (pipeline logic)...")
    from master_pipeline import _simulate_trade_realized_path

    df_signals_sorted = df_signals.sort_values('datetime')
    trades_pipeline = []
    flat_time = None

    for _, sig in df_signals_sorted.iterrows():
        entry_idx = int(sig['bar_index'])
        entry_dt = pd.to_datetime(sig['datetime'])
        trade_shares = int(sig['shares'])

        if flat_time is not None and entry_dt <= flat_time:
            continue

        t = _simulate_trade_realized_path(
            df_full=df,
            entry_bar_index=entry_idx,
            stop_atr=STOP_ATR,
            shares=trade_shares,
            slippage_per_share=SLIPPAGE_PER_SHARE,
            capital=CAPITAL,
        )
        trades_pipeline.append(t)
        flat_time = t['exit_datetime']

    df_pipe = pd.DataFrame(trades_pipeline)
    print(f"    Pipeline trades: {len(df_pipe):,}")
    print(f"    Pipeline net P&L: ${df_pipe['net_pnl'].sum():+,.0f}")
    wr = (df_pipe['net_pnl'] > 0).mean() * 100
    print(f"    Pipeline win rate: {wr:.1f}%")
    print(f"    Pipeline avg shares: {df_pipe['shares'].mean():.0f}")

    # Save pipeline trades
    pipe_path = Path("data/pipeline_trades_stop0.75_2023.csv")
    pipe_path.parent.mkdir(exist_ok=True)
    df_pipe.to_csv(pipe_path, index=False)
    print(f"    Saved: {pipe_path}")

    # ── Print first 30 pipeline trades ──────────────────────────────────
    print(f"\n{'='*100}")
    print("FIRST 30 PIPELINE TRADES")
    print(f"{'='*100}")
    cum = 0
    for i, (_, t) in enumerate(df_pipe.head(30).iterrows()):
        cum += t['net_pnl']
        side = 'LONG' if t['is_long'] else 'SHORT'
        print(f"  #{i+1:>3}: {t['entry_datetime']}  {side:5}  {int(t['shares']):>5} sh  "
              f"@ ${t['entry_price']:.2f} -> ${t['exit_price']:.2f} ({t['exit_reason']:>4})  "
              f"Net=${t['net_pnl']:>+9,.0f}  Cum=${cum:>+10,.0f}  "
              f"vwap_dist={t['vwap_dist_atr']:.2f}")

    # ── Now compare with streaming sim ──────────────────────────────────
    # Check if a sim output exists
    sim_path = Path("data/sim_nn_pnl_2023_stop0.75_risk0.01.csv")
    if sim_path.exists():
        df_sim = pd.read_csv(sim_path)
        print(f"\n{'='*100}")
        print("FIRST 30 SIMULATION TRADES")
        print(f"{'='*100}")
        cum = 0
        for i, (_, t) in enumerate(df_sim.head(30).iterrows()):
            cum += t['net_pnl']
            side = 'LONG' if t['is_long'] else 'SHORT'
            print(f"  #{i+1:>3}: {t['entry_datetime']}  {side:5}  {int(t['shares']):>5} sh  "
                  f"@ ${t['entry_price']:.2f} -> ${t['exit_price']:.2f} ({t['exit_reason']:>4})  "
                  f"Net=${t['net_pnl']:>+9,.0f}  Cum=${cum:>+10,.0f}")

        # Compare matching trades by entry_datetime
        print(f"\n{'='*100}")
        print("COMPARISON SUMMARY")
        print(f"{'='*100}")
        print(f"  Pipeline: {len(df_pipe):>5} trades, ${df_pipe['net_pnl'].sum():>+12,.0f}")
        print(f"  Sim:      {len(df_sim):>5} trades, ${df_sim['net_pnl'].sum():>+12,.0f}")

        # Match by entry_datetime
        pipe_entries = set(pd.to_datetime(df_pipe['entry_datetime']).dt.strftime('%Y-%m-%d %H:%M'))
        sim_entries = set(pd.to_datetime(df_sim['entry_datetime']).dt.strftime('%Y-%m-%d %H:%M'))
        matched = pipe_entries & sim_entries
        pipe_only = pipe_entries - sim_entries
        sim_only = sim_entries - pipe_entries
        print(f"\n  Matched entries:     {len(matched):>5}")
        print(f"  Pipeline-only:       {len(pipe_only):>5}")
        print(f"  Sim-only:            {len(sim_only):>5}")

        if pipe_only:
            print(f"\n  First 10 pipeline-only entries:")
            for e in sorted(pipe_only)[:10]:
                row = df_pipe[pd.to_datetime(df_pipe['entry_datetime']).dt.strftime('%Y-%m-%d %H:%M') == e].iloc[0]
                side = 'LONG' if row['is_long'] else 'SHORT'
                print(f"    {e}  {side:5}  {int(row['shares']):>5} sh  "
                      f"Net=${row['net_pnl']:>+9,.0f}  vwap_dist={row['vwap_dist_atr']:.2f}")

        if sim_only:
            print(f"\n  First 10 sim-only entries:")
            for e in sorted(sim_only)[:10]:
                row = df_sim[pd.to_datetime(df_sim['entry_datetime']).dt.strftime('%Y-%m-%d %H:%M') == e].iloc[0]
                side = 'LONG' if row['is_long'] else 'SHORT'
                print(f"    {e}  {side:5}  {int(row['shares']):>5} sh  "
                      f"Net=${row['net_pnl']:>+9,.0f}")

        # For matched trades, compare direction and exit
        if matched:
            print(f"\n  Direction mismatches in matched trades:")
            n_dir_mismatch = 0
            n_exit_mismatch = 0
            n_price_mismatch = 0
            for e in sorted(matched)[:50]:
                p = df_pipe[pd.to_datetime(df_pipe['entry_datetime']).dt.strftime('%Y-%m-%d %H:%M') == e].iloc[0]
                s = df_sim[pd.to_datetime(df_sim['entry_datetime']).dt.strftime('%Y-%m-%d %H:%M') == e].iloc[0]
                if p['is_long'] != s['is_long']:
                    n_dir_mismatch += 1
                    print(f"    {e}: pipeline={('LONG' if p['is_long'] else 'SHORT'):5} vs sim={('LONG' if s['is_long'] else 'SHORT'):5}")
                if p['exit_reason'] != s['exit_reason']:
                    n_exit_mismatch += 1
                if abs(p['entry_price'] - s['entry_price']) > 0.01:
                    n_price_mismatch += 1
            print(f"    Direction mismatches: {n_dir_mismatch} / {len(matched)}")
            print(f"    Exit reason mismatches: {n_exit_mismatch} / {len(matched)}")
            print(f"    Entry price mismatches: {n_price_mismatch} / {len(matched)}")
    else:
        print(f"\n  [INFO] No simulation output found at {sim_path}")
        print(f"  Run the simulation first with:")
        print(f"    python sim_trading/simulate_nn_pnl.py --year 2023 --stop-atr 0.75 --risk-pct 0.01 --min-rr 1.0 --model-file {MODEL_PATH}")

    # ── Diagnostic: how many bars pass each filter stage ────────────────
    print(f"\n{'='*100}")
    print("FILTER FUNNEL (test year 2023)")
    print(f"{'='*100}")
    df_2023 = df[df['year'] == TEST_YEAR].copy()
    n_total = len(df_2023)
    n_valid_label = df_2023[label_col].notna().sum()
    n_min_dist = (df_2023['vwap_width_atr'] >= SETUP_MIN_DIST).sum()
    n_min_rr = (df_2023['vwap_width_atr'] / STOP_ATR >= SETUP_MIN_RR).sum()
    n_minutes_ok = ((df_2023['minutes_into_session'] >= SETUP_MIN_MINUTES) &
                    (df_2023['minutes_into_session'] <= SETUP_MAX_MINUTES)).sum()
    # Combined
    all_filters = (
        df_2023[label_col].notna() &
        (df_2023['vwap_width_atr'] >= SETUP_MIN_DIST) &
        (df_2023['vwap_width_atr'] / STOP_ATR >= SETUP_MIN_RR) &
        (df_2023['minutes_into_session'] >= SETUP_MIN_MINUTES) &
        (df_2023['minutes_into_session'] <= SETUP_MAX_MINUTES)
    )
    n_all_filters = all_filters.sum()

    print(f"  Total bars:                   {n_total:>6,}")
    print(f"  Valid labels:                 {n_valid_label:>6,}  ({n_valid_label/n_total*100:.1f}%)")
    print(f"  vwap_width >= {SETUP_MIN_DIST}:          {n_min_dist:>6,}  ({n_min_dist/n_total*100:.1f}%)")
    print(f"  vwap_width/stop >= {SETUP_MIN_RR}:       {n_min_rr:>6,}  ({n_min_rr/n_total*100:.1f}%)")
    print(f"  minutes [{SETUP_MIN_MINUTES}..{SETUP_MAX_MINUTES}]:           {n_minutes_ok:>6,}  ({n_minutes_ok/n_total*100:.1f}%)")
    print(f"  ALL filters combined:         {n_all_filters:>6,}  ({n_all_filters/n_total*100:.1f}%)")
    print(f"  (Pipeline reported n_test:     2,877)")

    # Without label filter (what streaming sim sees)
    no_label = (
        (df_2023['vwap_width_atr'] >= SETUP_MIN_DIST) &
        (df_2023['vwap_width_atr'] / STOP_ATR >= SETUP_MIN_RR) &
        (df_2023['minutes_into_session'] >= SETUP_MIN_MINUTES) &
        (df_2023['minutes_into_session'] <= SETUP_MAX_MINUTES)
    )
    n_no_label = no_label.sum()
    print(f"  Filters WITHOUT label check:  {n_no_label:>6,}  ({n_no_label/n_total*100:.1f}%)")
    print(f"  (This is what streaming sim sees)")

    # Score all bars that pass filter (no label requirement)
    df_2023_filtered = df_2023[no_label].copy()
    X_filtered = df_2023_filtered[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Check for NaN features
    has_nan = X_filtered.isna().any(axis=1)
    n_nan = has_nan.sum()
    print(f"  Bars with NaN features:       {n_nan:>6,}")

    probs_all = model.predict_proba(X_filtered.values)[:, 1]
    n_above_thresh = (probs_all >= PROB_THRESHOLD).sum()
    print(f"  P(win) >= {PROB_THRESHOLD} (no label):    {n_above_thresh:>6,}")
    print(f"  P(win) >= {PROB_THRESHOLD} (w/ label):    {above_thresh.sum():>6,}  (pipeline's signal count)")

    # ── Distribution of probabilities ───────────────────────────────────
    print(f"\n  Probability distribution (all filtered bars, no label req):")
    for lo, hi in [(0, 0.3), (0.3, 0.4), (0.4, 0.45), (0.45, 0.5),
                   (0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
        n = ((probs_all >= lo) & (probs_all < hi)).sum()
        print(f"    [{lo:.2f}, {hi:.2f}): {n:>5,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
