"""
NN PnL Paper Trading Simulation

Streaming (bar-by-bar) simulation using the NN PnL model with:
  - Long AND short VWAP reversion setups
  - VWAP touch / stop / EOD exit logic (matches master_pipeline exactly)
  - Prob-scaled fixed-share sizing (default, matches master_pipeline training)
  - Optional risk-based position sizing (--risk-pct > 0)
  - IBKR margin cost per trade
  - Setup filter (min_dist, min/max minutes, min R:R)
  - Flat-to-flat (one position at a time)

Usage:
  python sim_trading/simulate_nn_pnl.py --year 2024 --stop-atr 0.40
  python sim_trading/simulate_nn_pnl.py --year 2024 --stop-atr 0.40 --min-shares 100 --max-shares 500
  python sim_trading/simulate_nn_pnl.py --year 2024 --stop-atr 0.40 --risk-pct 0.01  # risk-based mode
"""

import sys
from pathlib import Path

# Ensure imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import math
import numpy as np
import pandas as pd
from datetime import datetime

from sim_trading.streaming_indicators_aligned import StreamingIndicatorsAligned
from model_persistence import load_model
from model_selector import load_model_for_stop

# ---------------------------------------------------------------------------
# Constants (aligned with master_pipeline.py)
# ---------------------------------------------------------------------------
CAPITAL = 1_000_000
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_PER_SHARE = 0.01
PROB_THRESHOLD = 0.50
MIN_PROB_SCALE = 0.30          # at threshold prob -> 30% of full size
# Default fixed-share sizing (matches master_pipeline defaults)
DEFAULT_MIN_SHARES = 100       # shares at P(win) = threshold
DEFAULT_MAX_SHARES = 500       # shares at P(win) = 1.0

# IBKR margin rate tiers (same as master_pipeline.py)
IBKR_MARGIN_RATE_TIERS = [
    (100_000,       0.0683),
    (1_000_000,     0.0633),
    (50_000_000,    0.0608),
    (200_000_000,   0.0583),
    (float('inf'),  0.0558),
]
IBKR_DAYS_PER_YEAR = 360


def ibkr_margin_cost(borrowed: float, hold_hours: float) -> float:
    """Compute IBKR tiered margin interest."""
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
        total_interest += tier_amount * rate * hold_hours / (IBKR_DAYS_PER_YEAR * 24.0)
        remaining -= tier_amount
        prev_bound = upper_bound
    return total_interest


# ---------------------------------------------------------------------------
# Setup filter (same criteria as master_pipeline)
# ---------------------------------------------------------------------------
def passes_setup_filter(indicators: dict, stop_atr: float,
                        min_dist_atr: float = 0.5,
                        min_minutes: int = 15,
                        max_minutes: int = 360,
                        min_rr: float = 1.0) -> bool:
    """Check if the current bar qualifies as a valid reversion setup."""
    vwap_width = indicators.get('vwap_width_atr', 0)
    mins = indicators.get('minutes_into_session', 0)
    if pd.isna(vwap_width) or pd.isna(mins):
        return False
    if vwap_width < min_dist_atr:
        return False
    if mins < min_minutes or mins > max_minutes:
        return False
    if stop_atr > 0 and min_rr > 0:
        rr = vwap_width / stop_atr
        if rr < min_rr:
            return False
    return True


# ---------------------------------------------------------------------------
# Position sizing (aligned with master_pipeline.py)
# ---------------------------------------------------------------------------
def compute_shares(prob: float, atr: float, stop_atr: float,
                   risk_pct: float, capital: float,
                   min_shares: int = DEFAULT_MIN_SHARES,
                   max_shares: int = DEFAULT_MAX_SHARES) -> int:
    """Compute shares scaled by model probability.

    Two modes (matching master_pipeline.py):
      1. Fixed-share mode (risk_pct <= 0, DEFAULT):
         Linearly scale between min_shares and max_shares based on prob.
         At P=threshold -> min_shares, at P=1.0 -> max_shares.
      2. Risk-based mode (risk_pct > 0):
         shares = (capital * risk_pct) / (stop_atr * ATR), then scale by prob.
         Scale: 30% of full size at threshold, 100% at P=1.0, capped at 9999.
    """
    # Probability scaling factor: 0.0 at threshold, 1.0 at P=1.0
    prob_range = 1.0 - PROB_THRESHOLD
    if prob_range <= 0:
        prob_range = 0.01
    raw_scale = min(max((prob - PROB_THRESHOLD) / prob_range, 0.0), 1.0)

    if risk_pct > 0:
        # Risk-based sizing
        if atr <= 0 or stop_atr <= 0:
            return 0
        risk_dollars = capital * risk_pct
        stop_risk_per_share = stop_atr * atr
        full_shares = int(math.floor(risk_dollars / stop_risk_per_share))
        if full_shares <= 0:
            return 0
        scale_frac = MIN_PROB_SCALE + raw_scale * (1.0 - MIN_PROB_SCALE)
        shares = max(1, int(round(full_shares * scale_frac)))
        return min(shares, 9999)
    else:
        # Fixed-share mode (master_pipeline default)
        shares = int(round(min_shares + raw_scale * (max_shares - min_shares)))
        return max(1, min(shares, max_shares))


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if 'time' in df.columns and 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['time'], utc=True)
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    if 'date' not in df.columns:
        df['date'] = df['datetime'].dt.date
    return df


def run_simulation(
    df_test: pd.DataFrame,
    model,
    feature_cols: list,
    stop_atr: float,
    risk_pct: float,
    capital: float,
    lookback: int = 200,
    verbose: bool = True,
    min_shares: int = DEFAULT_MIN_SHARES,
    max_shares: int = DEFAULT_MAX_SHARES,
    warmup_bars: int = 60,
    min_rr: float = 0.0,
) -> pd.DataFrame:
    """Run bar-by-bar paper trading simulation.

    Bar history management (mimics live IBKR paper trading):
      - Keeps ALL bars from the prior trading day's open onward so that
        groupby('date') operations (VWAP, day_high/low, opening range,
        volume profile, prior-day stats, etc.) are computed correctly.
      - Also prepends `warmup_bars` extra bars before the prior day's
        open so rolling indicators (ATR-14, EMA-20, rolling-60, etc.)
        are fully warmed up.
      - On each new day, the oldest complete day is pruned, keeping the
        buffer at ~2-3 days + warmup.

    Returns DataFrame of closed trades with full P&L breakdown.
    """
    indicator_calc = StreamingIndicatorsAligned(verbose=False)

    trades = []          # completed trades
    position = None      # dict describing open position, or None
    last_exit_dt = None  # timestamp of last exit; no new entry until bar_dt > this

    # --- Bar history: keeps full days + warmup for rolling indicators ---
    # all_bars accumulates every bar seen.  We build the indicator window
    # from it by keeping: warmup_bars + all bars from prior-day-open onward.
    all_bars = []        # every bar appended (list of dicts)
    day_start_indices = {}   # date -> index in all_bars where that day starts
    prior_day_date = None
    current_day_date = None

    n_bars = len(df_test)
    n_signals = 0
    n_setups = 0
    n_entries = 0

    for i, (_, row) in enumerate(df_test.iterrows()):
        if i == 0 and verbose:
            print("  Starting bar-by-bar simulation...", flush=True)
        bar = row.to_dict()
        all_bars.append(bar)

        bar_dt = pd.to_datetime(bar['datetime'])
        bar_date = bar.get('date', bar_dt.date())

        # Track day boundaries
        if bar_date != current_day_date:
            prior_day_date = current_day_date
            current_day_date = bar_date
            day_start_indices[bar_date] = len(all_bars) - 1

        # Build indicator window: keep from (prior-day-open - warmup) onward
        # so groupby('date') sees complete prior day + current day,
        # and rolling indicators have warmup history.
        if prior_day_date is not None and prior_day_date in day_start_indices:
            window_start = max(0, day_start_indices[prior_day_date] - warmup_bars)
        elif current_day_date in day_start_indices:
            window_start = max(0, day_start_indices[current_day_date] - warmup_bars)
        else:
            window_start = max(0, len(all_bars) - lookback)

        window_bars = all_bars[window_start:]

        # Calculate indicators — need enough bars for ATR(14) at minimum
        indicators = {}
        if len(window_bars) >= 20:
            bars_df = pd.DataFrame(window_bars)
            indicators = indicator_calc.calculate(bars_df)        # Prune memory: once we have 3+ completed days, drop the oldest.
        # Keep: warmup_bars before prior_day_start + prior day + current day.
        if prior_day_date is not None and len(day_start_indices) > 2:
            dates_sorted = sorted(day_start_indices.keys())
            # Keep only the two most recent days (prior + current)
            keep_from_date = dates_sorted[-2]  # prior day
            prune_to = max(0, day_start_indices[keep_from_date] - warmup_bars)
            if prune_to > 0:
                all_bars = all_bars[prune_to:]
                # Rebase indices
                day_start_indices = {
                    k: v - prune_to
                    for k, v in day_start_indices.items()
                    if v >= prune_to
                }
                # Remove dates that were fully pruned
                for old_date in dates_sorted[:-2]:
                    day_start_indices.pop(old_date, None)

        # ----- EXIT LOGIC (check before entry) -----
        if position is not None:
            hi = float(bar.get('high', bar['close']))
            lo = float(bar.get('low', bar['close']))
            close = float(bar['close'])

            exited = False
            exit_price = None
            exit_reason = None

            # Only check stop/VWAP starting from the bar AFTER entry
            if bar_dt > position['entry_dt']:
                is_long = position['is_long']
                stop_price = position['stop_price']
                target_vwap = position['target_vwap']

                # Conservative: check stop first
                if is_long:
                    if lo <= stop_price:
                        exit_price = stop_price
                        exit_reason = 'stop'
                        exited = True
                    elif hi >= target_vwap:
                        exit_price = target_vwap
                        exit_reason = 'vwap'
                        exited = True
                else:
                    if hi >= stop_price:
                        exit_price = stop_price
                        exit_reason = 'stop'
                        exited = True
                    elif lo <= target_vwap:
                        exit_price = target_vwap
                        exit_reason = 'vwap'
                        exited = True

            # EOD exit: close at end of day
            is_last_bar_of_day = False
            if i + 1 < n_bars:
                next_date = df_test.iloc[i + 1].get('date',
                    pd.to_datetime(df_test.iloc[i + 1]['datetime']).date())
                if next_date != bar_date:
                    is_last_bar_of_day = True
            else:
                is_last_bar_of_day = True

            if not exited and is_last_bar_of_day:
                exit_price = close
                exit_reason = 'eod'
                exited = True

            if exited:
                # Compute P&L
                shares = position['shares']
                is_long = position['is_long']
                entry_price = position['entry_price']

                if is_long:
                    gross = (exit_price - entry_price) * shares
                else:
                    gross = (entry_price - exit_price) * shares

                costs = 2.0 * (COMMISSION_PER_SHARE + SLIPPAGE_PER_SHARE) * shares

                # Margin cost
                notional = entry_price * shares
                borrowed = max(0.0, notional - capital)
                hold_td = bar_dt - position['entry_dt']
                hold_hours = max(hold_td.total_seconds() / 3600.0, 5.0 / 60.0)
                margin_cost = ibkr_margin_cost(borrowed, hold_hours)

                net = gross - costs - margin_cost

                trade = {
                    'entry_datetime': position['entry_dt'],
                    'exit_datetime': bar_dt,
                    'is_long': int(is_long),
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'shares': shares,
                    'prob': position['prob'],
                    'gross_pnl': gross,
                    'costs': costs,
                    'margin_cost': margin_cost,
                    'net_pnl': net,
                    'notional': notional,
                    'margin_borrowed': borrowed,
                    'hold_hours': hold_hours,
                    'risk_dollars': position['risk_dollars'],
                    'vwap_dist_atr': position['vwap_dist_atr'],
                }
                trades.append(trade)
                position = None
                last_exit_dt = bar_dt  # prevent re-entry on this same bar

                if verbose and (len(trades) <= 20 or len(trades) % 50 == 0):
                    side = 'LONG' if trade['is_long'] else 'SHORT'
                    print(
                        f"  TRADE #{len(trades):>4}: {trade['entry_datetime']} {side} "
                        f"{shares:>5} sh @ ${entry_price:.2f} -> "
                        f"${exit_price:.2f} ({exit_reason}) "
                        f"Net=${net:+,.0f}  Cum=${sum(t['net_pnl'] for t in trades):+,.0f}",
                        flush=True,
                    )

        # ----- ENTRY LOGIC -----
        if position is None and indicators:
            # Don't enter on the same bar we just exited (matches pipeline flat_time)
            if last_exit_dt is not None and bar_dt <= last_exit_dt:
                continue

            atr = indicators.get('atr', 0)
            if pd.isna(atr) or atr <= 0:
                continue

            if not passes_setup_filter(indicators, stop_atr, min_rr=min_rr):
                continue
            n_setups += 1

            # Check both long AND short setups
            is_long_setup = bool(indicators.get('is_long_setup', False))
            # Note: master pipeline labels both long (close<vwap) and short (close>vwap)
            # The model was trained on both via the is_long_setup feature

            # Get model probability
            feature_vector = [indicators.get(c) for c in feature_cols]
            if any(v is None or (isinstance(v, float) and pd.isna(v)) for v in feature_vector):
                continue
            n_signals += 1

            prob = model.predict_proba([feature_vector])[0, 1]
            if prob < PROB_THRESHOLD:
                continue

            # Compute shares
            shares = compute_shares(prob, atr, stop_atr, risk_pct, capital,
                                    min_shares=min_shares, max_shares=max_shares)
            if shares <= 0:
                continue

            entry_price = float(bar['close'])
            vwap = float(indicators.get('vwap', entry_price))
            stop_dist = stop_atr * atr

            if is_long_setup:
                stop_price = entry_price - stop_dist
            else:
                stop_price = entry_price + stop_dist

            # Target = entry bar's VWAP (fixed, matches master_pipeline)
            target_vwap = vwap

            position = {
                'entry_dt': bar_dt,
                'entry_date': bar_date,
                'entry_price': entry_price,
                'is_long': is_long_setup,
                'stop_price': stop_price,
                'target_vwap': target_vwap,
                'shares': shares,
                'prob': prob,
                'risk_dollars': stop_dist * shares,
                'vwap_dist_atr': float(indicators.get('vwap_width_atr', 0)),
            }
            n_entries += 1        # Progress
        if verbose and (i + 1) % 1000 == 0:
            cum_pnl = sum(t['net_pnl'] for t in trades) if trades else 0
            print(f"  ... {i+1:>6}/{n_bars} bars | {len(trades)} trades | P&L=${cum_pnl:+,.0f}", flush=True)

    # Force close if still holding at end of data
    if position is not None:
        last_bar = df_test.iloc[-1]
        exit_price = float(last_bar['close'])
        shares = position['shares']
        is_long = position['is_long']
        entry_price = position['entry_price']

        if is_long:
            gross = (exit_price - entry_price) * shares
        else:
            gross = (entry_price - exit_price) * shares

        costs = 2.0 * (COMMISSION_PER_SHARE + SLIPPAGE_PER_SHARE) * shares
        notional = entry_price * shares
        borrowed = max(0.0, notional - capital)
        hold_hours = 5.0 / 60.0  # minimal
        margin_cost = ibkr_margin_cost(borrowed, hold_hours)
        net = gross - costs - margin_cost

        trades.append({
            'entry_datetime': position['entry_dt'],
            'exit_datetime': pd.to_datetime(last_bar['datetime']),
            'is_long': int(is_long),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': 'end_of_data',
            'shares': shares,
            'prob': position['prob'],
            'gross_pnl': gross,
            'costs': costs,
            'margin_cost': margin_cost,
            'net_pnl': net,
            'notional': notional,
            'margin_borrowed': borrowed,
            'hold_hours': hold_hours,
            'risk_dollars': position['risk_dollars'],
            'vwap_dist_atr': position.get('vwap_dist_atr', 0),
        })
        position = None

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # Print summary
    if verbose:
        print(f"\n{'='*70}", flush=True)
        print(f"SIMULATION RESULTS", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  Bars processed:    {n_bars:,}", flush=True)
        print(f"  Setups seen:       {n_setups:,}", flush=True)
        print(f"  Signals (features OK): {n_signals:,}", flush=True)
        print(f"  Entries taken:     {n_entries:,}", flush=True)
        print(f"  Trades closed:     {len(trades):,}", flush=True)

        if len(trades_df) > 0:
            total_net = trades_df['net_pnl'].sum()
            total_gross = trades_df['gross_pnl'].sum()
            total_costs = trades_df['costs'].sum()
            total_margin = trades_df['margin_cost'].sum()
            n_wins = (trades_df['net_pnl'] > 0).sum()
            n_losses = (trades_df['net_pnl'] <= 0).sum()
            win_rate = n_wins / len(trades_df) * 100
            avg_win = trades_df.loc[trades_df['net_pnl'] > 0, 'net_pnl'].mean() if n_wins > 0 else 0
            avg_loss = trades_df.loc[trades_df['net_pnl'] <= 0, 'net_pnl'].mean() if n_losses > 0 else 0
            avg_shares = trades_df['shares'].mean()
            avg_notional = trades_df['notional'].mean()
            avg_hold = trades_df['hold_hours'].mean()

            # Exit reason breakdown
            exit_counts = trades_df['exit_reason'].value_counts()

            # Long vs short
            n_long = (trades_df['is_long'] == 1).sum()
            n_short = (trades_df['is_long'] == 0).sum()
            long_pnl = trades_df.loc[trades_df['is_long'] == 1, 'net_pnl'].sum() if n_long > 0 else 0
            short_pnl = trades_df.loc[trades_df['is_long'] == 0, 'net_pnl'].sum() if n_short > 0 else 0

            print(f"\n  Total Net P&L:     ${total_net:+,.0f}", flush=True)
            print(f"  Total Gross P&L:   ${total_gross:+,.0f}", flush=True)
            print(f"  Total Costs:       ${total_costs:,.0f}", flush=True)
            print(f"  Total Margin Cost: ${total_margin:,.0f}", flush=True)
            print(f"  Return on Capital: {total_net / capital * 100:+.1f}%", flush=True)
            print(f"\n  Win Rate:          {win_rate:.1f}% ({n_wins}W / {n_losses}L)", flush=True)
            print(f"  Avg Win:           ${avg_win:+,.0f}", flush=True)
            print(f"  Avg Loss:          ${avg_loss:+,.0f}", flush=True)
            print(f"  Avg Shares/Trade:  {avg_shares:,.0f}", flush=True)
            print(f"  Avg Notional:      ${avg_notional:,.0f}", flush=True)
            print(f"  Avg Hold (hours):  {avg_hold:.2f}", flush=True)
            print(f"\n  Long:  {n_long:>4} trades  ${long_pnl:+,.0f}", flush=True)
            print(f"  Short: {n_short:>4} trades  ${short_pnl:+,.0f}", flush=True)
            print(f"\n  Exit reasons:", flush=True)
            for reason, count in exit_counts.items():
                pnl = trades_df.loc[trades_df['exit_reason'] == reason, 'net_pnl'].sum()
                print(f"    {reason:>6}: {count:>4} trades  ${pnl:+,.0f}", flush=True)

            # Monthly breakdown
            trades_df['_month'] = pd.to_datetime(trades_df['entry_datetime']).dt.to_period('M')
            print(f"\n  Monthly P&L:", flush=True)
            for month, grp in trades_df.groupby('_month'):
                m_net = grp['net_pnl'].sum()
                m_n = len(grp)
                m_wr = (grp['net_pnl'] > 0).mean() * 100
                print(f"    {month}: ${m_net:>+10,.0f}  ({m_n:>3} trades, {m_wr:.0f}% WR)", flush=True)
            trades_df.drop(columns=['_month'], inplace=True)

        print(f"\n{'='*70}", flush=True)

    return trades_df


def main():
    p = argparse.ArgumentParser(description="NN PnL paper trading simulation")
    p.add_argument("--year", type=int, default=2024, help="Test year")
    p.add_argument("--stop-atr", type=float, default=0.40, help="Stop width in ATR (default: 0.40)")
    p.add_argument("--risk-pct", type=float, default=0.0,
                   help="Risk per trade as fraction of capital (default: 0.0 = disabled, use fixed shares). "
                        "When > 0, overrides --min-shares/--max-shares.")
    p.add_argument("--min-shares", type=int, default=DEFAULT_MIN_SHARES,
                   help=f"Min shares at P(win)=threshold (default: {DEFAULT_MIN_SHARES})")
    p.add_argument("--max-shares", type=int, default=DEFAULT_MAX_SHARES,
                   help=f"Max shares at P(win)=1.0 (default: {DEFAULT_MAX_SHARES})")
    p.add_argument("--capital", type=float, default=CAPITAL, help="Capital (default: $1M)")
    p.add_argument("--lookback", type=int, default=200, help="Lookback bars for indicators")
    p.add_argument("--data", type=str, default="data/tsla_5min_10years.csv", help="Data file")
    p.add_argument("--models-dir", type=str, default="models", help="Models directory")
    p.add_argument("--model-file", type=str, default=None,
                   help="Path to a specific model .pkl file (overrides --models-dir auto-selection)")
    p.add_argument("--model-type", type=str, default="PnLModelWrapper",
                   help="Filter models by model_type metadata (default: PnLModelWrapper for nn_pnl). "
                        "Use 'MLPClassifier' for sklearn NN, 'RandomForestClassifier' for RF, "
                        "or 'any' to disable filtering.")
    p.add_argument("--output", type=str, default=None, help="Output CSV path (default: auto)")
    p.add_argument("--quiet", action="store_true", help="Suppress per-trade output")
    p.add_argument("--min-rr", type=float, default=1.0, help="Minimum R:R ratio for setups (default: 1.0)")
    args = p.parse_args()

    # Sizing mode label
    if args.risk_pct > 0:
        sizing_str = f"Risk-based: {args.risk_pct:.1%} of ${args.capital:,.0f}"
    else:
        sizing_str = f"Fixed shares: [{args.min_shares}..{args.max_shares}] (prob-scaled)"

    print(f"\n{'='*70}", flush=True)
    print(f"NN PnL PAPER TRADING SIMULATION", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Year:       {args.year}", flush=True)
    print(f"  Stop ATR:   {args.stop_atr}", flush=True)
    print(f"  Min R:R:    {args.min_rr}", flush=True)
    print(f"  Sizing:     {sizing_str}", flush=True)
    print(f"  Capital:    ${args.capital:,.0f}", flush=True)
    print(f"  Data:       {args.data}", flush=True)
    print(f"  Models:     {args.models_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Load data
    df = load_data(args.data)
    df['year'] = df['datetime'].dt.year
    df_test = df[df['year'] == args.year].reset_index(drop=True)
    if df_test.empty:
        raise SystemExit(f"No data found for year {args.year}")
    print(f"  Loaded {len(df_test):,} bars for {args.year}", flush=True)
    print(f"  Date range: {df_test['datetime'].min()} to {df_test['datetime'].max()}", flush=True)

    # Load model
    if args.model_file:
        from model_persistence import load_model as _load_model
        print(f"\n  Loading model from: {args.model_file}", flush=True)
        model, metadata = _load_model(args.model_file)
    else:
        model_type_filter = None if args.model_type == 'any' else args.model_type
        print(f"\n  Loading model for stop={args.stop_atr} ATR (type={args.model_type})...", flush=True)
        model, metadata = load_model_for_stop(
            stop_atr=args.stop_atr,
            models_dir=args.models_dir,
            latest=True,
            model_type=model_type_filter,
        )
    feature_cols = metadata['features']
    print(f"  Model type: {metadata.get('model_type', 'unknown')}", flush=True)
    print(f"  Features:   {len(feature_cols)}", flush=True)
    print(f"  Saved at:   {metadata.get('saved_at', 'unknown')}", flush=True)
    print(flush=True)

    # Run simulation
    trades_df = run_simulation(
        df_test=df_test,
        model=model,
        feature_cols=feature_cols,
        stop_atr=args.stop_atr,
        risk_pct=args.risk_pct,
        capital=args.capital,
        lookback=args.lookback,
        verbose=not args.quiet,
        min_shares=args.min_shares,
        max_shares=args.max_shares,
        min_rr=args.min_rr,
    )

    # Save results
    if args.output:
        out_path = Path(args.output)
    else:
        if args.risk_pct > 0:
            out_path = Path("data") / f"sim_nn_pnl_{args.year}_stop{args.stop_atr}_risk{args.risk_pct}.csv"
        else:
            out_path = Path("data") / f"sim_nn_pnl_{args.year}_stop{args.stop_atr}_shares{args.min_shares}-{args.max_shares}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(trades_df) > 0:
        trades_df.to_csv(out_path, index=False)
        print(f"\n  Trades saved to: {out_path}", flush=True)
    else:
        print("\n  No trades to save.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
