"""
BRAND NEW CONCURRENT BACKTESTER - Built from scratch

This is a completely new implementation that doesn't use StreamingSimulator.
It processes bars sequentially and manages multiple concurrent positions directly.
"""

import os
os.environ['PYTHONWARNINGS'] = 'ignore'  # Suppress all Python warnings at OS level
os.environ['LOKY_MAX_CPU_COUNT'] = '1'  # Prevent sklearn parallel warnings

import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings

# Suppress specific sklearn warnings
import logging
logging.getLogger('sklearn').setLevel(logging.ERROR)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

# Configure sklearn to not use joblib parallelism (fixes warnings)
import sklearn
sklearn.set_config(assume_finite=True)

from src.model_selector import load_model_for_stop
from sim_trading.streaming_indicators_aligned import StreamingIndicatorsAligned


class ConcurrentBacktester:
    """Simple backtester that allows multiple concurrent positions"""
    
    def __init__(self, initial_capital, position_size, stop_atr, rr, rf_threshold,
                 commission_per_share: float = 0.0, slippage_per_share: float = 0.0):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.stop_atr = stop_atr
        self.rr = rr
        self.rf_threshold = rf_threshold
        self.commission_per_share = commission_per_share
        self.slippage_per_share = slippage_per_share

        # Portfolio state
        self.cash = initial_capital
        self.positions = {}  # symbol -> {entry_price, quantity, entry_time, stop, target, is_long}
        self.next_position_id = 1
        
        # Trade history
        self.closed_trades = []
        self.equity_curve = []
    
    def calculate_portfolio_value(self, current_price):
        """Calculate total portfolio value"""
        position_value = sum(pos['quantity'] * current_price for pos in self.positions.values())
        return self.cash + position_value
    
    def open_position(self, bar, atr):
        """Open a new position"""
        symbol = f"TSLA_{self.next_position_id}"
        self.next_position_id += 1
        
        entry_price = bar['close']  # Use close price for entry
        vwap = bar.get('vwap', np.nan)
        
        # Check if VWAP is valid
        if np.isnan(vwap):
            return None
        
        quantity = self.position_size
        cost = entry_price * quantity
        
        # Check if we have enough cash
        if cost > self.cash:
            return None
        
        # Deduct cash
        self.cash -= cost
          # MATCH LABEL GENERATOR: Determine if long or short based on price vs VWAP
        is_long = entry_price < vwap
        stop_dist = self.stop_atr * atr
        
        if is_long:
            # Long position: stop below entry, target at VWAP
            stop_price = entry_price - stop_dist
            target_price = vwap
        else:
            # Short position: stop above entry, target at VWAP
            stop_price = entry_price + stop_dist
            target_price = vwap
        
        # Store position
        self.positions[symbol] = {
            'entry_price': entry_price,
            'quantity': quantity,
            'entry_time': bar['datetime'],
            'stop': stop_price,
            'target': target_price,
            'is_long': is_long,  # Track direction
        }
        
        return symbol
    
    def check_exits(self, bar):
        """Check all positions for stop/target hits - MATCH LABEL GENERATOR LOGIC"""
        symbols_to_close = []
        
        for symbol, pos in self.positions.items():
            is_long = pos.get('is_long', True)  # Default to long for backward compatibility
            
            if is_long:
                # Long position: check low for stop, high for target
                if bar['low'] <= pos['stop']:
                    symbols_to_close.append((symbol, pos['stop'], 'stop'))
                elif bar['high'] >= pos['target']:
                    symbols_to_close.append((symbol, pos['target'], 'target'))
            else:
                # Short position: check high for stop, low for target
                if bar['high'] >= pos['stop']:
                    symbols_to_close.append((symbol, pos['stop'], 'stop'))
                elif bar['low'] <= pos['target']:
                    symbols_to_close.append((symbol, pos['target'], 'target'))
          # Close positions
        for symbol, exit_price, exit_reason in symbols_to_close:
            self.close_position(symbol, exit_price, bar['datetime'], exit_reason)
    
    def close_position(self, symbol, exit_price, exit_time, reason):
        """Close a position"""
        pos = self.positions[symbol]
        
        # Calculate P&L - MUST account for position direction!
        is_long = pos.get('is_long', True)
        if is_long:
            # Long: profit when price goes up
            gross_pnl = (exit_price - pos['entry_price']) * pos['quantity']
        else:
            # Short: profit when price goes down
            gross_pnl = (pos['entry_price'] - exit_price) * pos['quantity']

        # Costs (round trip)
        costs = 2.0 * (self.commission_per_share + self.slippage_per_share) * pos['quantity']
        net_pnl = gross_pnl - costs

        # Add cash back
        proceeds = exit_price * pos['quantity']
        self.cash += proceeds

        # Execution outcome (matches label semantics): win iff target hit
        outcome = 1 if reason == 'target' else 0

        # Record trade
        self.closed_trades.append({
            'symbol': symbol,
            'entry_time': pos['entry_time'],
            'exit_time': exit_time,
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'quantity': pos['quantity'],
            'gross_pnl': gross_pnl,
            'costs': costs,
            'pnl': net_pnl,
            'reason': reason,
            'outcome': outcome,
            'is_long': is_long,
            'stop': pos.get('stop', np.nan),
            'target': pos.get('target', np.nan),
        })
        
        # Remove position
        del self.positions[symbol]
    
    def close_all_positions(self, bar):
        """Close all remaining positions at end"""
        for symbol in list(self.positions.keys()):
            self.close_position(symbol, bar['close'], bar['datetime'], 'eod')
    
    def record_equity(self, timestamp, current_price):
        """Record current equity"""
        equity = self.calculate_portfolio_value(current_price)
        self.equity_curve.append({
            'datetime': timestamp,
            'equity': equity,
            'cash': self.cash,
            'positions': len(self.positions),
        })


def _compute_forward_label_and_exit(df_day: pd.DataFrame, j: int, stop_atr: float):
    """Match src/label_generator.py forward-looking logic for ONE entry bar.

    Uses:
    - entry_price = close[j]
    - vwap[j] as target
    - stop = +/- stop_atr * atr[j]
    - scan k=j+1..day_end-1
    - stop check has priority over target check

    Returns: (label, exit_reason, exit_price, exit_time, is_long)
    """
    row_j = df_day.iloc[j]
    entry_price = float(row_j['close'])
    vwap = float(row_j['vwap'])
    atr = float(row_j['atr'])

    if not np.isfinite(entry_price) or not np.isfinite(vwap) or not np.isfinite(atr) or atr <= 0:
        return np.nan, None, np.nan, pd.NaT, None

    is_long = entry_price < vwap
    stop_dist = stop_atr * atr

    if is_long:
        stop_price = entry_price - stop_dist
        target_price = vwap
    else:
        stop_price = entry_price + stop_dist
        target_price = vwap

    # Last bar => cannot evaluate (match label generator)
    if j + 1 >= len(df_day):
        return np.nan, None, np.nan, pd.NaT, is_long

    # Scan forward within the day
    for k in range(j + 1, len(df_day)):
        row_k = df_day.iloc[k]
        hi = float(row_k['high'])
        lo = float(row_k['low'])

        if is_long:
            if lo <= stop_price:
                return 0, 'stop', stop_price, row_k['datetime'], is_long
            if hi >= target_price:
                return 1, 'target', target_price, row_k['datetime'], is_long
        else:
            if hi >= stop_price:
                return 0, 'stop', stop_price, row_k['datetime'], is_long
            if lo <= target_price:
                return 1, 'target', target_price, row_k['datetime'], is_long

    # If neither hit by EOD => loss (match label_generator docstring)
    return 0, 'eod', float(df_day.iloc[-1]['close']), df_day.iloc[-1]['datetime'], is_long


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--stop-atr", type=float, default=1.5)
    parser.add_argument("--rf-threshold", type=float, default=0.5)
    parser.add_argument("--position-size", type=int, default=100)
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--concurrent", action="store_true")
    parser.add_argument(
        "--pnl-mode",
        choices=["simulate", "label"],
        default="simulate",
        help="'simulate' = bar-by-bar stop/target execution. 'label' = forward-looking label-based outcomes (master-parity).",
    )
    parser.add_argument("--commission-per-share", type=float, default=0.005)
    parser.add_argument("--slippage-per-share", type=float, default=0.02)

    # New: allow explicit maximum concurrent positions (apples-to-apples mode)
    parser.add_argument("--max-positions", type=int, default=None,
                        help="Explicit max concurrent positions. If not set: 1 for single-mode, large number when --concurrent is used.")
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"NEW CONCURRENT BACKTESTER")
    print(f"{'='*80}")
    print(f"Mode: {'CONCURRENT' if args.concurrent else 'SINGLE'}")
    print(f"Year: {args.year}")
    print(f"Stop ATR: {args.stop_atr}")
    print(f"RF Threshold: {args.rf_threshold}\n")
      # Load data
    print("Loading data...")
    df = pd.read_csv("data/tsla_5min_10years.csv")
    if "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"], utc=True)
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    
    df["year"] = df["datetime"].dt.year
    df_year = df[df["year"] == args.year].reset_index(drop=True)
    print(f"Loaded {len(df_year):,} bars\n")
      # Load model
    print("Loading model...")
    model, metadata = load_model_for_stop(stop_atr=args.stop_atr, models_dir="models", latest=True)
    feature_cols = metadata["features"]
    rr = float(metadata.get("rr", 1.2))
    
    # CRITICAL FIX: Disable parallel execution in RandomForest to avoid sklearn warnings
    # This prevents the joblib parallel warnings that slow down execution
    model.n_jobs = 1  # Force single-threaded predictions
    
    print(f"Model loaded. RR: {rr}\n")
    
    # PRE-CALCULATE ALL INDICATORS (MASSIVE SPEED IMPROVEMENT!)
    print("Pre-calculating indicators for entire year...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from master_pipeline import calculate_core_indicators
    
    # Add date column if needed
    if 'date' not in df_year.columns:
        df_year['date'] = df_year['datetime'].dt.date
    
    # Calculate all indicators ONCE
    df_year = calculate_core_indicators(df_year, verbose=False)
    print(f"Indicators calculated for {len(df_year):,} bars\n")
    
    # Initialize backtester
    bt = ConcurrentBacktester(
        initial_capital=args.capital,
        position_size=args.position_size,
        stop_atr=args.stop_atr,
        rr=rr,
        rf_threshold=args.rf_threshold,
        commission_per_share=float(args.commission_per_share),
        slippage_per_share=float(args.slippage_per_share),
    )
    
    # Determine max positions for this run (backwards compatible)
    if args.max_positions is None:
        max_positions = 9999 if args.concurrent else 1
    else:
        max_positions = args.max_positions
    
    # Prepare per-bar feature/prob export for detailed diffs against master
    per_bar_records = []
    
    # Statistics
    stats = {
        'bars_processed': 0,
        'signals_generated': 0,
        'positions_opened': 0,
        'max_concurrent': 0,
    }    # Process bars
    print("Processing bars...", flush=True)
    
    # Open output file for real-time progress
    with open('concurrent_progress.txt', 'w') as f:
        f.write(f"Starting simulation at {datetime.now()}\n")
      # Debug counters
    debug_counts = {
        'total_bars': 0,
        'no_indicators': 0,
        'no_atr': 0,
        'atr_zero': 0,
        'not_long_setup': 0,
        'missing_features': 0,
        'rf_filtered': 0,
        'signals_taken': 0,
        'cant_enter': 0,
    }
    
    lookback = 200
    
    # Track current trading day for EOD closes (to match label generator)
    current_date = None

    # Group day boundaries for label mode (fast day slicing)
    if args.pnl_mode == "label":
        df_year = df_year.copy()
        if 'date' not in df_year.columns:
            df_year['date'] = df_year['datetime'].dt.date
    
    for i in range(lookback, len(df_year)):
        bar = df_year.iloc[i]
        stats['bars_processed'] += 1
        debug_counts['total_bars'] += 1
        
        # Get pre-calculated indicators for current bar
        # No need to recalculate - just read from dataframe!
        atr = bar.get('atr', 0.0)
        
        # ALWAYS EXPORT ALL BARS (for apples-to-apples comparison with master)
        # Extract features and calculate probability for every bar
        is_setup = bar.get('is_long_setup', False)
        feature_vector = [bar.get(col, np.nan) for col in feature_cols]
        has_all_features = not any(pd.isna(x) for x in feature_vector)
        
        # Calculate probability if we have all features
        prob = np.nan
        if has_all_features:
            prob = model.predict_proba([feature_vector])[0, 1]
        
        # Record this bar (ALL BARS, not just setups)
        record = {
            'datetime': bar['datetime'],
            'is_setup': is_setup,
            'has_all_features': has_all_features,
            'prob': prob,
        }
        # Add all features
        for col in feature_cols:
            record[col] = bar.get(col, np.nan)
        per_bar_records.append(record)
        
        # Now continue with execution logic
        if pd.isna(atr) or atr == 0.0:
            debug_counts['atr_zero'] += 1
            continue

        # In label mode, we do NOT manage concurrent open positions via check_exits/open_position.
        # We emulate master: each selected bar becomes a forward-looking labeled trade outcome.
        if args.pnl_mode == "label":
            if not has_all_features:
                debug_counts['missing_features'] += 1
                continue
            if prob < bt.rf_threshold:
                debug_counts['rf_filtered'] += 1
                continue

            # SIGNAL GENERATED
            stats['signals_generated'] += 1
            debug_counts['signals_taken'] += 1

            # Slice current day
            bar_date = bar['datetime'].date()
            day_mask = df_year['date'] == bar_date
            df_day = df_year.loc[day_mask].reset_index(drop=True)

            # Map global index i to day-relative z
            # (safe because we reset_index in df_day)
            day_indices = np.flatnonzero(day_mask.to_numpy())
            j_rel = int(np.where(day_indices == i)[0][0])

            label, exit_reason, exit_price, exit_time, is_long = _compute_forward_label_and_exit(
                df_day=df_day,
                j=j_rel,
                stop_atr=args.stop_atr,
            )
            if np.isnan(label):
                # Can't evaluate (last bar / missing data)
                continue

            entry_price = float(bar['close'])
            risk_per_share = args.stop_atr * float(bar['atr'])
            reward_per_share = risk_per_share * rr

            quantity = args.position_size
            gross_pnl = (reward_per_share * quantity) if label == 1 else (-risk_per_share * quantity)
            costs = 2.0 * (args.commission_per_share + args.slippage_per_share) * quantity
            net_pnl = gross_pnl - costs

            bt.closed_trades.append({
                'symbol': f"TSLA_LABEL_{stats['signals_generated']}",
                'entry_time': bar['datetime'],
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': float(exit_price),
                'quantity': quantity,
                'gross_pnl': float(gross_pnl),
                'costs': float(costs),
                'pnl': float(net_pnl),
                'reason': str(exit_reason),
                'is_long': bool(is_long),
                'label': int(label),
                'stop_atr': float(args.stop_atr),
            })

            continue

        # =========================
        # simulate mode (existing)
        # =========================

        # MATCH MASTER: Close all positions at end of day (like label generator)
        bar_date = bar['datetime'].date()
        if current_date is not None and bar_date != current_date:
            # New day started - close all positions from yesterday
            if len(bt.positions) > 0:
                for symbol in list(bt.positions.keys()):
                    bt.close_position(symbol, bar['close'], bar['datetime'], 'eod')
        current_date = bar_date
        
        # CHECK EXITS FIRST
        bt.check_exits(bar)
          # CHECK ENTRY: allow entry only if we have free position slots
        can_enter = len(bt.positions) < max_positions
        
        if not can_enter:
            debug_counts['cant_enter'] += 1
            # Count signals even if we can't enter
            if has_all_features and prob >= bt.rf_threshold:
                stats['signals_generated'] += 1
            continue
        
        # MASTER PIPELINE APPROACH: No is_long_setup check!
        # Master generates signals on ANY bar with prob >= threshold
        # (Removed is_long_setup filter to match master pipeline)
        
        # Check if we have all features
        if not has_all_features:
            debug_counts['missing_features'] += 1
            continue
        
        # Check RF probability
        if prob < bt.rf_threshold:
            debug_counts['rf_filtered'] += 1
            continue
        
        # SIGNAL GENERATED
        stats['signals_generated'] += 1
        debug_counts['signals_taken'] += 1
        
        # Open position
        symbol = bt.open_position(bar, atr)
        if symbol:
            stats['positions_opened'] += 1
            
            # Track max concurrent
            current_pos = len(bt.positions)
            if current_pos > stats['max_concurrent']:
                stats['max_concurrent'] = current_pos
              # Progress
            if stats['positions_opened'] % 100 == 0:
                msg = f"  Opened {stats['positions_opened']} positions, max concurrent: {stats['max_concurrent']}"
                print(msg, flush=True)
                with open('concurrent_progress.txt', 'a') as f:
                    f.write(f"{msg}\n")
        
        # Record equity every bar
        bt.record_equity(bar['datetime'], bar['close'])
          # Progress
        if (i - lookback) % 10000 == 0:
            pct = ((i - lookback) / (len(df_year) - lookback)) * 100
            msg = f"  Progress: {pct:.1f}% (bar {i}/{len(df_year)})"
            print(msg, flush=True)
            with open('concurrent_progress.txt', 'a') as f:
                f.write(f"{msg}\n")      # Close all remaining positions
    print("\nClosing remaining positions...", flush=True)
    if args.pnl_mode == "simulate":
        bt.close_all_positions(df_year.iloc[-1])
    
    # Debug output
    print(f"\n{'='*80}", flush=True)
    print("DEBUG: SIGNAL FILTERING BREAKDOWN", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Total bars processed: {debug_counts['total_bars']:,}", flush=True)
    print(f"  - No indicators: {debug_counts['no_indicators']:,}", flush=True)
    print(f"  - No ATR: {debug_counts['no_atr']:,}", flush=True)
    print(f"  - ATR zero/NaN: {debug_counts['atr_zero']:,}", flush=True)
    print(f"  - Can't enter (position full): {debug_counts['cant_enter']:,}", flush=True)
    print(f"  - Not long setup: {debug_counts['not_long_setup']:,}", flush=True)
    print(f"  - Missing features: {debug_counts['missing_features']:,}", flush=True)
    print(f"  - RF filtered: {debug_counts['rf_filtered']:,}", flush=True)
    print(f"  - Signals taken: {debug_counts['signals_taken']:,}", flush=True)
    
    # Results
    print(f"\n{'='*80}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Bars processed: {stats['bars_processed']:,}", flush=True)
    print(f"Signals generated: {stats['signals_generated']:,}", flush=True)
    print(f"Positions opened: {stats['positions_opened']:,}", flush=True)
    print(f"Max concurrent: {stats['max_concurrent']}", flush=True)
    
    if len(bt.closed_trades) > 0:
        trades_df = pd.DataFrame(bt.closed_trades)
        total_pnl = trades_df['pnl'].sum()

        # Win rate should be based on execution outcome (target vs stop/eod),
        # not whether net P&L stayed positive after costs.
        if 'outcome' in trades_df.columns:
            win_rate = trades_df['outcome'].mean()
        else:
            win_rate = (trades_df['pnl'] > 0).mean()

        print(f"\nTotal trades: {len(trades_df):,}")
        print(f"Total P&L: ${total_pnl:,.2f}")
        print(f"Win rate: {win_rate:.1%}")
        
        if (trades_df['pnl'] > 0).any():
            print(f"Avg win: ${trades_df[trades_df['pnl'] > 0]['pnl'].mean():.2f}")
        if (trades_df['pnl'] < 0).any():
            print(f"Avg loss: ${trades_df[trades_df['pnl'] < 0]['pnl'].mean():.2f}")
        
        # Save results
        suffix = "_concurrent" if args.concurrent else "_single"
        trades_df.to_csv(f"data/concurrent_backtest_trades{suffix}.csv", index=False)
        
        equity_df = pd.DataFrame(bt.equity_curve)
        equity_df.to_csv(f"data/concurrent_backtest_equity{suffix}.csv", index=False)
        # Save per-bar feature + RF probability export for reconciliation
        if len(per_bar_records) > 0:
            per_bar_df = pd.DataFrame(per_bar_records)
            per_bar_df.to_csv(f"data/concurrent_per_bar_features{suffix}.csv", index=False)
        
        print(f"\nResults saved to data/concurrent_backtest_*{suffix}.csv")
    else:
        print("\nWARNING: No trades executed!")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
