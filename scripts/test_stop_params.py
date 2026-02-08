"""
Test different stop and target parameters to find optimal settings.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from indicators import calc_all_indicators, identify_reversal_setups, calc_theo_targets
from outcome_sim import simulate_all_setups, print_outcome_summary


def run_backtest(df_ind: pd.DataFrame, stop_atr: float, fallback_atr: float, 
                 max_rsi_long: float = 35.0, min_rsi_short: float = 65.0) -> dict:
    """Run backtest with given parameters and return metrics."""
    
    # Identify setups
    df_setups = identify_reversal_setups(df_ind, max_rsi_long=max_rsi_long, 
                                          min_rsi_short=min_rsi_short, min_rel_vol=1.0)
    
    # Calculate targets with given params
    df_targets = calc_theo_targets(df_setups, stop_atr_mult=stop_atr, 
                                    fallback_target_atr=fallback_atr)
    
    # Simulate outcomes
    trades_df = simulate_all_setups(df_targets)
    
    if len(trades_df) == 0:
        return None
    
    # Calculate metrics
    total = len(trades_df)
    winners = trades_df['is_winner'].sum()
    win_rate = winners / total * 100
    
    # By direction
    long_trades = trades_df[trades_df['direction'] == 'LONG']
    short_trades = trades_df[trades_df['direction'] == 'SHORT']
    
    long_wr = long_trades['is_winner'].mean() * 100 if len(long_trades) > 0 else 0
    short_wr = short_trades['is_winner'].mean() * 100 if len(short_trades) > 0 else 0
    
    # P&L metrics
    avg_pnl = trades_df['pnl_per_share'].mean()
    avg_rr = trades_df['rr_achieved'].mean()
    
    winners_df = trades_df[trades_df['is_winner']]
    losers_df = trades_df[~trades_df['is_winner']]
    
    total_wins = winners_df['pnl_per_share'].sum() if len(winners_df) > 0 else 0
    total_losses = abs(losers_df['pnl_per_share'].sum()) if len(losers_df) > 0 else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    # Exit types
    target_pct = (trades_df['outcome'] == 'TARGET').mean() * 100
    stop_pct = (trades_df['outcome'] == 'STOP').mean() * 100
    eod_pct = (trades_df['outcome'] == 'EOD').mean() * 100
    
    return {
        'stop_atr': stop_atr,
        'fallback_atr': fallback_atr,
        'total_trades': total,
        'win_rate': win_rate,
        'long_wr': long_wr,
        'short_wr': short_wr,
        'avg_pnl': avg_pnl,
        'avg_rr': avg_rr,
        'profit_factor': profit_factor,
        'target_pct': target_pct,
        'stop_pct': stop_pct,
        'eod_pct': eod_pct
    }


def main():
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    input_file = os.path.join(data_dir, 'tsla_5min_2025_01.csv')
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, parse_dates=['time'], index_col='time')
    print(f"Loaded {len(df)} bars\n")
    
    # Calculate indicators once
    print("Calculating indicators...")
    df_ind = calc_all_indicators(df)
    
    # Test different stop widths
    stop_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
    fallback_atr = 1.5  # Keep fallback constant for now
    
    print("\n" + "=" * 90)
    print("STOP WIDTH PARAMETER SWEEP")
    print("=" * 90)
    print(f"{'Stop ATR':<10} {'Trades':<8} {'Win%':<8} {'Long%':<8} {'Short%':<8} "
          f"{'Avg PnL':<10} {'PF':<8} {'Target%':<10} {'Stop%':<10}")
    print("-" * 90)
    
    results = []
    for stop_atr in stop_multipliers:
        result = run_backtest(df_ind, stop_atr=stop_atr, fallback_atr=fallback_atr)
        if result:
            results.append(result)
            print(f"{result['stop_atr']:<10.2f} {result['total_trades']:<8} "
                  f"{result['win_rate']:<8.1f} {result['long_wr']:<8.1f} {result['short_wr']:<8.1f} "
                  f"${result['avg_pnl']:<9.2f} {result['profit_factor']:<8.2f} "
                  f"{result['target_pct']:<10.1f} {result['stop_pct']:<10.1f}")
    
    # Find best parameters
    print("\n" + "=" * 90)
    print("BEST PARAMETERS")
    print("=" * 90)
    
    if results:
        # Best by win rate
        best_wr = max(results, key=lambda x: x['win_rate'])
        print(f"\nBest Win Rate: {best_wr['win_rate']:.1f}% with stop={best_wr['stop_atr']:.2f} ATR")
        
        # Best by profit factor
        best_pf = max(results, key=lambda x: x['profit_factor'] if x['profit_factor'] < float('inf') else 0)
        print(f"Best Profit Factor: {best_pf['profit_factor']:.2f} with stop={best_pf['stop_atr']:.2f} ATR")
        
        # Best by avg P&L
        best_pnl = max(results, key=lambda x: x['avg_pnl'])
        print(f"Best Avg P&L: ${best_pnl['avg_pnl']:.2f} with stop={best_pnl['stop_atr']:.2f} ATR")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = os.path.join(data_dir, 'stop_param_sweep_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
