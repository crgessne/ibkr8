"""
Compare Concurrent Backtest vs Master Pipeline Results
Performs detailed trade-by-trade analysis to identify differences
"""

import pandas as pd
import numpy as np
from datetime import datetime

def analyze_concurrent_results():
    """Analyze the concurrent backtest results"""
    print("="*80)
    print("CONCURRENT BACKTEST ANALYSIS")
    print("="*80)
    
    # Load concurrent trades
    try:
        concurrent_trades = pd.read_csv('data/concurrent_backtest_trades_concurrent.csv')
        print(f"\nConcurrent Trades Loaded: {len(concurrent_trades)} trades")
        print(f"Columns: {list(concurrent_trades.columns)}")
        
        # Convert timestamps
        concurrent_trades['entry_time'] = pd.to_datetime(concurrent_trades['entry_time'])
        concurrent_trades['exit_time'] = pd.to_datetime(concurrent_trades['exit_time'])
        
        # Basic statistics
        print(f"\n--- Concurrent Trade Statistics ---")
        print(f"Total Trades: {len(concurrent_trades)}")
        print(f"Total P&L: ${concurrent_trades['pnl'].sum():,.2f}")
        print(f"Avg P&L per trade: ${concurrent_trades['pnl'].mean():,.2f}")
        
        # Win/Loss breakdown
        winners = concurrent_trades[concurrent_trades['pnl'] > 0]
        losers = concurrent_trades[concurrent_trades['pnl'] <= 0]
        print(f"\nWinners: {len(winners)} ({len(winners)/len(concurrent_trades)*100:.1f}%)")
        print(f"  Avg Win: ${winners['pnl'].mean():,.2f}")
        print(f"  Total Win P&L: ${winners['pnl'].sum():,.2f}")
        
        print(f"\nLosers: {len(losers)} ({len(losers)/len(concurrent_trades)*100:.1f}%)")
        print(f"  Avg Loss: ${losers['pnl'].mean():,.2f}")
        print(f"  Total Loss P&L: ${losers['pnl'].sum():,.2f}")
        
        # Exit reasons
        print(f"\n--- Exit Reasons ---")
        print(concurrent_trades['reason'].value_counts())
        
        # Temporal analysis
        concurrent_trades['entry_date'] = concurrent_trades['entry_time'].dt.date
        daily_trades = concurrent_trades.groupby('entry_date').size()
        print(f"\n--- Daily Trade Distribution ---")
        print(f"Avg trades per day: {daily_trades.mean():.1f}")
        print(f"Max trades in a day: {daily_trades.max()}")
        print(f"Min trades in a day: {daily_trades.min()}")
        print(f"Total trading days: {len(daily_trades)}")
        
        # Monthly breakdown
        concurrent_trades['month'] = concurrent_trades['entry_time'].dt.to_period('M')
        monthly_stats = concurrent_trades.groupby('month').agg({
            'pnl': ['sum', 'mean', 'count'],
        }).round(2)
        print(f"\n--- Monthly Breakdown ---")
        print(monthly_stats)
        
        return concurrent_trades
        
    except Exception as e:
        print(f"Error loading concurrent trades: {e}")
        return None


def analyze_master_pipeline():
    """Analyze master pipeline results"""
    print("\n" + "="*80)
    print("MASTER PIPELINE ANALYSIS")
    print("="*80)
    
    try:
        # Load master pipeline summary
        master = pd.read_csv('data/master_pipeline_results_20260209_155832.csv')
        
        # Filter for our specific configuration
        target = master[(master['stop_atr'] == 1.5) & (master['rf_threshold'] == 0.5)]
        
        if len(target) > 0:
            row = target.iloc[0]
            print(f"\n--- Master Pipeline (Stop ATR=1.5, RF Threshold=0.5) ---")
            print(f"Total Trades: {row['n_trades']:.0f}")
            print(f"Win Rate: {row['win_rate']*100:.2f}%")
            print(f"Expected Value: {row['ev']:.4f}")
            print(f"Total Net P&L: ${row['total_net_pnl']:,.2f}")
            print(f"Total Return %: {row['total_return_pct']:.2f}%")
            print(f"Max Positions Held: {row['max_positions_held']:.0f}")
            print(f"Signals Filtered: {row['pct_filtered']:.2f}%")
            
            return row
        else:
            print("No matching master pipeline configuration found!")
            return None
            
    except Exception as e:
        print(f"Error loading master pipeline: {e}")
        return None


def compare_results(concurrent_trades, master_row):
    """Compare concurrent vs master results"""
    print("\n" + "="*80)
    print("COMPARISON ANALYSIS")
    print("="*80)
    
    if concurrent_trades is None or master_row is None:
        print("Cannot compare - missing data")
        return
    
    # Calculate concurrent stats
    concurrent_total_pnl = concurrent_trades['pnl'].sum()
    concurrent_win_rate = (concurrent_trades['pnl'] > 0).mean()
    concurrent_n_trades = len(concurrent_trades)
    
    # Master stats
    master_total_pnl = master_row['total_net_pnl']
    master_win_rate = master_row['win_rate']
    master_n_trades = master_row['n_trades']
    
    print(f"\n{'Metric':<30} {'Concurrent':<20} {'Master':<20} {'Difference':<20}")
    print("-"*90)
    print(f"{'Total Trades':<30} {concurrent_n_trades:<20.0f} {master_n_trades:<20.0f} {concurrent_n_trades-master_n_trades:<20.0f}")
    print(f"{'Win Rate':<30} {concurrent_win_rate*100:<20.2f} {master_win_rate*100:<20.2f} {(concurrent_win_rate-master_win_rate)*100:<20.2f}")
    print(f"{'Total P&L ($)':<30} {concurrent_total_pnl:<20,.2f} {master_total_pnl:<20,.2f} {concurrent_total_pnl-master_total_pnl:<20,.2f}")
    
    # Calculate percentage differences
    trade_diff_pct = ((concurrent_n_trades - master_n_trades) / master_n_trades) * 100
    pnl_diff_pct = ((concurrent_total_pnl - master_total_pnl) / abs(master_total_pnl)) * 100
    
    print(f"\n{'% Difference':<30}")
    print(f"  {'Trades':<28} {trade_diff_pct:>20.2f}%")
    print(f"  {'P&L':<28} {pnl_diff_pct:>20.2f}%")
    
    print(f"\n🚨 KEY FINDINGS:")
    print(f"  • Concurrent has {abs(trade_diff_pct):.1f}% {'FEWER' if trade_diff_pct < 0 else 'MORE'} trades")
    print(f"  • Win rate is {abs(concurrent_win_rate-master_win_rate)*100:.1f} percentage points {'LOWER' if concurrent_win_rate < master_win_rate else 'HIGHER'}")
    print(f"  • P&L difference: ${abs(concurrent_total_pnl-master_total_pnl):,.2f}")
    
    if concurrent_total_pnl < 0 and master_total_pnl > 0:
        print(f"\n⚠️  CRITICAL: Concurrent LOSES money while Master MAKES money!")


def main():
    """Main analysis function"""
    print(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: Stop ATR = 1.5, RF Threshold = 0.5, Year = 2024\n")
    
    # Analyze both
    concurrent_trades = analyze_concurrent_results()
    master_row = analyze_master_pipeline()
    
    # Compare
    compare_results(concurrent_trades, master_row)
    
    print("\n" + "="*80)
    print("Analysis complete! See CONCURRENT_VS_MASTER_RECONCILIATION.md for details.")
    print("="*80)


if __name__ == "__main__":
    main()
