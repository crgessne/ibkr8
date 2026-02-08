"""
Test different stop and target parameters on 2-year dataset.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from indicators import calc_all_indicators, identify_reversal_setups, calc_theo_targets
from outcome_sim import simulate_all_setups

def main():
    # Load 2-year data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    input_file = os.path.join(data_dir, 'tsla_5min_2years.csv')
    
    print(f"Loading data from {input_file}...", flush=True)
    df = pd.read_csv(input_file, parse_dates=['time'], index_col='time')
    print(f"Loaded {len(df)} bars", flush=True)
    print(f"Date range: {df.index.min()} to {df.index.max()}", flush=True)
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}", flush=True)
    
    # Calculate indicators once
    print("\nCalculating indicators...", flush=True)
    df_ind = calc_all_indicators(df)
    df_setups = identify_reversal_setups(df_ind, max_rsi_long=35.0, min_rsi_short=65.0, min_rel_vol=1.0)
    
    long_setups = df_setups['long_setup'].sum()
    short_setups = df_setups['short_setup'].sum()
    print(f"Found {long_setups} LONG setups, {short_setups} SHORT setups", flush=True)
    
    # Test different stop widths
    stop_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    
    print("\n" + "=" * 100, flush=True)
    print("STOP WIDTH PARAMETER SWEEP (2-Year Dataset)", flush=True)
    print("=" * 100, flush=True)
    print(f"{'Stop ATR':<10} {'Trades':<8} {'Win%':<8} {'Long%':<8} {'Short%':<8} "
          f"{'Avg PnL':<10} {'PF':<8} {'Target%':<10} {'Stop%':<10} {'EOD%':<8}", flush=True)
    print("-" * 100, flush=True)
    
    results = []
    for stop_atr in stop_multipliers:
        df_targets = calc_theo_targets(df_setups, stop_atr_mult=stop_atr, fallback_target_atr=1.5)
        trades = simulate_all_setups(df_targets)
        
        if len(trades) == 0:
            continue
        
        total = len(trades)
        winners = trades['is_winner'].sum()
        wr = winners / total * 100
        
        long_t = trades[trades['direction'] == 'LONG']
        short_t = trades[trades['direction'] == 'SHORT']
        long_wr = long_t['is_winner'].mean() * 100 if len(long_t) > 0 else 0
        short_wr = short_t['is_winner'].mean() * 100 if len(short_t) > 0 else 0
        
        avg_pnl = trades['pnl_per_share'].mean()
        
        win_sum = trades[trades['is_winner']]['pnl_per_share'].sum()
        loss_sum = abs(trades[~trades['is_winner']]['pnl_per_share'].sum())
        pf = win_sum / loss_sum if loss_sum > 0 else 999
        
        target_pct = (trades['outcome'] == 'TARGET').mean() * 100
        stop_pct = (trades['outcome'] == 'STOP').mean() * 100
        eod_pct = (trades['outcome'] == 'EOD').mean() * 100
        
        results.append({
            'stop_atr': stop_atr,
            'trades': total,
            'win_rate': wr,
            'long_wr': long_wr,
            'short_wr': short_wr,
            'avg_pnl': avg_pnl,
            'profit_factor': pf,
            'target_pct': target_pct,
            'stop_pct': stop_pct,
            'eod_pct': eod_pct
        })
        
        print(f"{stop_atr:<10.2f} {total:<8} {wr:<8.1f} {long_wr:<8.1f} {short_wr:<8.1f} "
              f"${avg_pnl:<9.2f} {pf:<8.2f} {target_pct:<10.1f} {stop_pct:<10.1f} {eod_pct:<8.1f}", flush=True)
    
    # Find best parameters
    print("\n" + "=" * 100, flush=True)
    print("OPTIMAL PARAMETERS", flush=True)
    print("=" * 100, flush=True)
    
    if results:
        # Best by win rate
        best_wr = max(results, key=lambda x: x['win_rate'])
        print(f"\nBest Win Rate: {best_wr['win_rate']:.1f}% with stop={best_wr['stop_atr']:.2f} ATR", flush=True)
        
        # Best by profit factor
        best_pf = max(results, key=lambda x: x['profit_factor'] if x['profit_factor'] < 900 else 0)
        print(f"Best Profit Factor: {best_pf['profit_factor']:.2f} with stop={best_pf['stop_atr']:.2f} ATR", flush=True)
        
        # Best by avg P&L
        best_pnl = max(results, key=lambda x: x['avg_pnl'])
        print(f"Best Avg P&L: ${best_pnl['avg_pnl']:.2f} with stop={best_pnl['stop_atr']:.2f} ATR", flush=True)
        
        # Best balanced (win rate > 45% and PF > 1.2)
        balanced = [r for r in results if r['win_rate'] > 45 and r['profit_factor'] > 1.2]
        if balanced:
            best_balanced = max(balanced, key=lambda x: x['avg_pnl'])
            print(f"\nBest Balanced (WR>45%, PF>1.2): stop={best_balanced['stop_atr']:.2f} ATR", flush=True)
            print(f"  Win Rate: {best_balanced['win_rate']:.1f}%", flush=True)
            print(f"  Profit Factor: {best_balanced['profit_factor']:.2f}", flush=True)
            print(f"  Avg P&L: ${best_balanced['avg_pnl']:.2f}", flush=True)
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = os.path.join(data_dir, 'stop_param_sweep_2years.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}", flush=True)


if __name__ == "__main__":
    main()
