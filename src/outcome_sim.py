"""
Outcome simulation for reversal trades.
Simulates forward-looking price action to determine if target or stop is hit first.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class TradeOutcome:
    """Result of a simulated trade."""
    entry_time: pd.Timestamp
    entry_price: float
    direction: str  # 'LONG' or 'SHORT'
    target: float
    stop: float
    exit_time: Optional[pd.Timestamp]
    exit_price: Optional[float]
    outcome: str  # 'TARGET', 'STOP', 'EOD' (end of day)
    bars_held: int
    pnl_per_share: float
    rr_achieved: float  # Actual R:R achieved (negative if loss)


def simulate_trade_outcome(
    df: pd.DataFrame,
    entry_idx: int,
    direction: str,
    entry_price: float,
    target: float,
    stop: float,
    max_bars: int = 78  # ~6.5 hours at 5-min bars
) -> TradeOutcome:
    """
    Simulate a single trade forward from entry to determine outcome.
    
    Args:
        df: DataFrame with OHLC data
        entry_idx: Integer index of entry bar
        direction: 'LONG' or 'SHORT'
        entry_price: Entry price
        target: Target price
        stop: Stop price
        max_bars: Maximum bars to hold before forcing exit
        
    Returns:
        TradeOutcome with results
    """
    entry_time = df.index[entry_idx]
    entry_date = entry_time.date() if hasattr(entry_time, 'date') else pd.Timestamp(entry_time).date()
    
    risk = abs(entry_price - stop)
    
    exit_time = None
    exit_price = None
    outcome = 'EOD'
    bars_held = 0
    
    # Look forward from entry
    for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(df))):
        bars_held += 1
        bar = df.iloc[i]
        bar_time = df.index[i]
        bar_date = bar_time.date() if hasattr(bar_time, 'date') else pd.Timestamp(bar_time).date()
        
        # Check if we've crossed into a new day
        if bar_date != entry_date:
            # Exit at previous bar's close (EOD)
            exit_time = df.index[i - 1]
            exit_price = df.iloc[i - 1]['close']
            outcome = 'EOD'
            bars_held -= 1
            break
        
        high = bar['high']
        low = bar['low']
        
        if direction == 'LONG':
            # Check stop first (conservative - assume worst case)
            if low <= stop:
                exit_time = bar_time
                exit_price = stop
                outcome = 'STOP'
                break
            # Check target
            if high >= target:
                exit_time = bar_time
                exit_price = target
                outcome = 'TARGET'
                break
        else:  # SHORT
            # Check stop first
            if high >= stop:
                exit_time = bar_time
                exit_price = stop
                outcome = 'STOP'
                break
            # Check target
            if low <= target:
                exit_time = bar_time
                exit_price = target
                outcome = 'TARGET'
                break
    
    # If we exited the loop without hitting target/stop, it's EOD
    if exit_time is None:
        exit_time = df.index[min(entry_idx + bars_held, len(df) - 1)]
        exit_price = df.iloc[min(entry_idx + bars_held, len(df) - 1)]['close']
        outcome = 'EOD'
    
    # Calculate P&L
    if direction == 'LONG':
        pnl_per_share = exit_price - entry_price
    else:
        pnl_per_share = entry_price - exit_price
    
    # R:R achieved (positive = profit in terms of risk units)
    rr_achieved = pnl_per_share / risk if risk > 0 else 0
    
    return TradeOutcome(
        entry_time=entry_time,
        entry_price=entry_price,
        direction=direction,
        target=target,
        stop=stop,
        exit_time=exit_time,
        exit_price=exit_price,
        outcome=outcome,
        bars_held=bars_held,
        pnl_per_share=pnl_per_share,
        rr_achieved=rr_achieved
    )


def simulate_all_setups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate outcomes for all reversal setups in the dataframe.
    
    Args:
        df: DataFrame with indicators and setup signals
            Must have: long_setup, short_setup, long_target, long_stop,
                      short_target, short_stop columns
    
    Returns:
        DataFrame with one row per trade and outcome details
    """
    results = []
    
    # Get integer positions for iteration
    df_reset = df.reset_index()
    
    for i, row in df_reset.iterrows():
        # Long setups
        if row.get('long_setup', False) and pd.notna(row.get('long_target')) and pd.notna(row.get('long_stop')):
            outcome = simulate_trade_outcome(
                df=df,
                entry_idx=i,
                direction='LONG',
                entry_price=row['close'],
                target=row['long_target'],
                stop=row['long_stop']
            )
            results.append({
                'entry_time': outcome.entry_time,
                'direction': outcome.direction,
                'entry_price': outcome.entry_price,
                'target': outcome.target,
                'stop': outcome.stop,
                'rsi': row.get('rsi'),
                'vwap': row.get('vwap'),
                'vwap_dist_pct': row.get('vwap_dist_pct'),
                'bb_pct': row.get('bb_pct'),
                'rel_vol': row.get('rel_vol'),
                'theo_rr': row.get('long_rr'),
                'exit_time': outcome.exit_time,
                'exit_price': outcome.exit_price,
                'outcome': outcome.outcome,
                'bars_held': outcome.bars_held,
                'pnl_per_share': outcome.pnl_per_share,
                'rr_achieved': outcome.rr_achieved,
                'is_winner': outcome.pnl_per_share > 0
            })
        
        # Short setups
        if row.get('short_setup', False) and pd.notna(row.get('short_target')) and pd.notna(row.get('short_stop')):
            outcome = simulate_trade_outcome(
                df=df,
                entry_idx=i,
                direction='SHORT',
                entry_price=row['close'],
                target=row['short_target'],
                stop=row['short_stop']
            )
            results.append({
                'entry_time': outcome.entry_time,
                'direction': outcome.direction,
                'entry_price': outcome.entry_price,
                'target': outcome.target,
                'stop': outcome.stop,
                'rsi': row.get('rsi'),
                'vwap': row.get('vwap'),
                'vwap_dist_pct': row.get('vwap_dist_pct'),
                'bb_pct': row.get('bb_pct'),
                'rel_vol': row.get('rel_vol'),
                'theo_rr': row.get('short_rr'),
                'exit_time': outcome.exit_time,
                'exit_price': outcome.exit_price,
                'outcome': outcome.outcome,
                'bars_held': outcome.bars_held,
                'pnl_per_share': outcome.pnl_per_share,
                'rr_achieved': outcome.rr_achieved,
                'is_winner': outcome.pnl_per_share > 0
            })
    
    return pd.DataFrame(results)


def print_outcome_summary(trades_df: pd.DataFrame):
    """Print summary statistics for trade outcomes."""
    
    if len(trades_df) == 0:
        print("No trades to analyze")
        return
    
    print("=" * 70)
    print("TRADE OUTCOME SUMMARY")
    print("=" * 70)
    
    total = len(trades_df)
    winners = trades_df['is_winner'].sum()
    win_rate = winners / total * 100
    
    print(f"\nTotal Trades: {total}")
    print(f"Winners: {winners} ({win_rate:.1f}%)")
    print(f"Losers: {total - winners} ({100 - win_rate:.1f}%)")
    
    # By outcome type
    print("\n--- By Exit Type ---")
    for outcome in ['TARGET', 'STOP', 'EOD']:
        count = (trades_df['outcome'] == outcome).sum()
        pct = count / total * 100 if total > 0 else 0
        print(f"  {outcome}: {count} ({pct:.1f}%)")
    
    # By direction
    print("\n--- By Direction ---")
    for direction in ['LONG', 'SHORT']:
        mask = trades_df['direction'] == direction
        dir_trades = trades_df[mask]
        if len(dir_trades) > 0:
            dir_winners = dir_trades['is_winner'].sum()
            dir_wr = dir_winners / len(dir_trades) * 100
            avg_rr = dir_trades['rr_achieved'].mean()
            print(f"  {direction}: {len(dir_trades)} trades, {dir_wr:.1f}% win rate, avg R:R = {avg_rr:.2f}")
    
    # P&L analysis
    print("\n--- P&L Analysis ---")
    print(f"  Avg P&L per share: ${trades_df['pnl_per_share'].mean():.2f}")
    print(f"  Avg R:R achieved: {trades_df['rr_achieved'].mean():.2f}")
    print(f"  Avg bars held: {trades_df['bars_held'].mean():.1f}")
    
    # Winners vs Losers
    winners_df = trades_df[trades_df['is_winner']]
    losers_df = trades_df[~trades_df['is_winner']]
    
    if len(winners_df) > 0:
        print(f"\n  Winners avg P&L: ${winners_df['pnl_per_share'].mean():.2f}")
        print(f"  Winners avg R:R: {winners_df['rr_achieved'].mean():.2f}")
    
    if len(losers_df) > 0:
        print(f"  Losers avg P&L: ${losers_df['pnl_per_share'].mean():.2f}")
        print(f"  Losers avg R:R: {losers_df['rr_achieved'].mean():.2f}")
    
    # Expectancy
    avg_win = winners_df['pnl_per_share'].mean() if len(winners_df) > 0 else 0
    avg_loss = abs(losers_df['pnl_per_share'].mean()) if len(losers_df) > 0 else 0
    expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
    
    print(f"\n--- Expectancy ---")
    print(f"  Expected P&L per trade: ${expectancy:.2f}")
    
    # Profit factor
    total_wins = winners_df['pnl_per_share'].sum() if len(winners_df) > 0 else 0
    total_losses = abs(losers_df['pnl_per_share'].sum()) if len(losers_df) > 0 else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    print(f"  Profit Factor: {profit_factor:.2f}")


if __name__ == "__main__":
    import sys
    import os
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    from indicators import calc_all_indicators, identify_reversal_setups, calc_theo_targets
    
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    input_file = os.path.join(data_dir, 'tsla_5min_2025_01.csv')
    output_file = os.path.join(data_dir, 'tsla_trade_outcomes.csv')
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, parse_dates=['time'], index_col='time')
    print(f"Loaded {len(df)} bars")
    
    # Calculate indicators
    print("\nCalculating indicators...")
    df_ind = calc_all_indicators(df)
    
    # Identify setups
    df_setups = identify_reversal_setups(df_ind, max_rsi_long=35.0, min_rsi_short=65.0, min_rel_vol=1.0)
    
    # Calculate targets (VWAP mean-reversion)
    df_targets = calc_theo_targets(df_setups, stop_atr_mult=1.0, fallback_target_atr=1.5)
    
    # Simulate outcomes
    print("\nSimulating trade outcomes...")
    trades_df = simulate_all_setups(df_targets)
    
    # Print summary
    print_outcome_summary(trades_df)
    
    # Save results
    print(f"\nSaving trade outcomes to {output_file}...")
    trades_df.to_csv(output_file, index=False)
    print(f"Saved {len(trades_df)} trades")
    
    # Show sample trades
    print("\n--- Sample Trades ---")
    print(trades_df[['entry_time', 'direction', 'entry_price', 'target', 'stop', 
                     'outcome', 'pnl_per_share', 'rr_achieved', 'is_winner']].head(10).to_string())
