"""
Win Rate Gap Analysis - Concurrent vs Master Pipeline
Investigates why concurrent has 47.9% win rate vs master's 66.4%
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def analyze_concurrent_trades():
    """Deep dive into concurrent backtest trade execution"""
    
    print("="*80)
    print("CONCURRENT BACKTEST TRADE ANALYSIS")
    print("="*80)
    
    # Load concurrent trades
    trades = pd.read_csv('data/concurrent_backtest_trades_concurrent.csv')
    trades['entry_time'] = pd.to_datetime(trades['entry_time'])
    trades['exit_time'] = pd.to_datetime(trades['exit_time'])
    trades['duration'] = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 60  # minutes
    trades['pnl_pct'] = (trades['pnl'] / (trades['entry_price'] * trades['quantity'])) * 100
    
    print(f"\nTotal Trades: {len(trades):,}")
    print(f"Date Range: {trades['entry_time'].min()} to {trades['exit_time'].max()}")
    
    # Win/Loss breakdown
    winners = trades[trades['pnl'] > 0]
    losers = trades[trades['pnl'] <= 0]
    
    print(f"\n--- Win/Loss Statistics ---")
    print(f"Winners: {len(winners):,} ({len(winners)/len(trades)*100:.2f}%)")
    print(f"  Avg Win: ${winners['pnl'].mean():.2f}")
    print(f"  Avg Win %: {winners['pnl_pct'].mean():.2f}%")
    print(f"  Max Win: ${winners['pnl'].max():.2f}")
    print(f"  Median Win: ${winners['pnl'].median():.2f}")
    
    print(f"\nLosers: {len(losers):,} ({len(losers)/len(trades)*100:.2f}%)")
    print(f"  Avg Loss: ${losers['pnl'].mean():.2f}")
    print(f"  Avg Loss %: {losers['pnl_pct'].mean():.2f}%")
    print(f"  Max Loss: ${losers['pnl'].min():.2f}")
    print(f"  Median Loss: ${losers['pnl'].median():.2f}")
    
    # Exit reason analysis
    print(f"\n--- Exit Reason Breakdown ---")
    exit_reasons = trades.groupby('reason').agg({
        'pnl': ['count', 'sum', 'mean']
    }).round(2)
    print(exit_reasons)
    
    # More detailed by reason
    for reason in trades['reason'].unique():
        reason_trades = trades[trades['reason'] == reason]
        reason_winners = reason_trades[reason_trades['pnl'] > 0]
        win_rate = len(reason_winners) / len(reason_trades) * 100 if len(reason_trades) > 0 else 0
        
        print(f"\n{reason.upper()}:")
        print(f"  Count: {len(reason_trades):,}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Total P&L: ${reason_trades['pnl'].sum():,.2f}")
        print(f"  Avg P&L: ${reason_trades['pnl'].mean():.2f}")
    
    # Duration analysis
    print(f"\n--- Trade Duration Analysis ---")
    print(f"Avg Duration: {trades['duration'].mean():.1f} minutes ({trades['duration'].mean()/60:.2f} hours)")
    print(f"Median Duration: {trades['duration'].median():.1f} minutes")
    print(f"Min Duration: {trades['duration'].min():.1f} minutes")
    print(f"Max Duration: {trades['duration'].max():.1f} minutes")
    
    print(f"\nWinners Duration: {winners['duration'].mean():.1f} minutes")
    print(f"Losers Duration: {losers['duration'].mean():.1f} minutes")
    
    # Check if stops are being hit more than targets
    stops = trades[trades['reason'] == 'stop']
    targets = trades[trades['reason'] == 'target']
    
    print(f"\n--- Stop vs Target Analysis ---")
    print(f"Stops Hit: {len(stops):,} ({len(stops)/len(trades)*100:.1f}%)")
    print(f"  Stop Win Rate: {(stops['pnl'] > 0).sum() / len(stops) * 100:.1f}%")
    print(f"  Stop Avg P&L: ${stops['pnl'].mean():.2f}")
    
    print(f"\nTargets Hit: {len(targets):,} ({len(targets)/len(trades)*100:.1f}%)")
    print(f"  Target Win Rate: {(targets['pnl'] > 0).sum() / len(targets) * 100:.1f}%")
    print(f"  Target Avg P&L: ${targets['pnl'].mean():.2f}")
    
    # Risk/Reward analysis
    print(f"\n--- Risk/Reward Analysis ---")
    print(f"Win/Loss Ratio: {abs(winners['pnl'].mean() / losers['pnl'].mean()):.2f}")
    print(f"Expectancy: ${(len(winners)/len(trades) * winners['pnl'].mean() + len(losers)/len(trades) * losers['pnl'].mean()):.2f}")
    
    # Price movement analysis
    trades['price_change'] = trades['exit_price'] - trades['entry_price']
    trades['price_change_pct'] = (trades['price_change'] / trades['entry_price']) * 100
    
    print(f"\n--- Price Movement Analysis ---")
    print(f"Avg Price Change: ${trades['price_change'].mean():.2f} ({trades['price_change_pct'].mean():.3f}%)")
    print(f"Winners Avg Price Change: ${winners['price_change'].mean():.2f} ({winners['price_change_pct'].mean():.3f}%)")
    print(f"Losers Avg Price Change: ${losers['price_change'].mean():.2f} ({losers['price_change_pct'].mean():.3f}%)")
    
    return trades


def compare_stop_target_logic():
    """Analyze stop and target placement and hit logic"""
    
    print("\n" + "="*80)
    print("STOP/TARGET PLACEMENT ANALYSIS")
    print("="*80)
    
    trades = pd.read_csv('data/concurrent_backtest_trades_concurrent.csv')
    
    # Calculate implied stop and target distances
    # For losers that hit stop, calculate how far stop was
    # For winners that hit target, calculate how far target was
    
    stops = trades[trades['reason'] == 'stop'].copy()
    targets = trades[trades['reason'] == 'target'].copy()
    
    if len(stops) > 0:
        # For stops, exit_price should be at stop level
        # Stop distance = entry - exit (for longs)
        stops['stop_distance'] = stops['entry_price'] - stops['exit_price']
        stops['stop_distance_pct'] = (stops['stop_distance'] / stops['entry_price']) * 100
        
        print(f"\nStop Distance Analysis:")
        print(f"  Avg Stop Distance: ${stops['stop_distance'].mean():.2f} ({stops['stop_distance_pct'].mean():.3f}%)")
        print(f"  Median Stop Distance: ${stops['stop_distance'].median():.2f} ({stops['stop_distance_pct'].median():.3f}%)")
        print(f"  Min Stop Distance: ${stops['stop_distance'].min():.2f}")
        print(f"  Max Stop Distance: ${stops['stop_distance'].max():.2f}")
    
    if len(targets) > 0:
        # For targets, exit_price should be at target level
        # Target distance = exit - entry (for longs)
        targets['target_distance'] = targets['exit_price'] - targets['entry_price']
        targets['target_distance_pct'] = (targets['target_distance'] / targets['entry_price']) * 100
        
        print(f"\nTarget Distance Analysis:")
        print(f"  Avg Target Distance: ${targets['target_distance'].mean():.2f} ({targets['target_distance_pct'].mean():.3f}%)")
        print(f"  Median Target Distance: ${targets['target_distance'].median():.2f} ({targets['target_distance_pct'].median():.3f}%)")
        print(f"  Min Target Distance: ${targets['target_distance'].min():.2f}")
        print(f"  Max Target Distance: ${targets['target_distance'].max():.2f}")
    
    if len(stops) > 0 and len(targets) > 0:
        print(f"\nTarget/Stop Ratio: {targets['target_distance'].mean() / stops['stop_distance'].mean():.2f}")
        print("(This should match the configured R:R ratio)")


def identify_potential_issues():
    """Identify potential issues causing low win rate"""
    
    print("\n" + "="*80)
    print("POTENTIAL ISSUES ANALYSIS")
    print("="*80)
    
    trades = pd.read_csv('data/concurrent_backtest_trades_concurrent.csv')
    trades['entry_time'] = pd.to_datetime(trades['entry_time'])
    trades['exit_time'] = pd.to_datetime(trades['exit_time'])
    
    issues = []
    
    # Issue 1: Too many stops being hit
    stop_rate = (trades['reason'] == 'stop').sum() / len(trades) * 100
    if stop_rate > 60:
        issues.append(f"⚠️  HIGH STOP RATE: {stop_rate:.1f}% of trades hit stop (suggests stops too tight)")
    
    # Issue 2: Negative expectancy
    winners = trades[trades['pnl'] > 0]
    losers = trades[trades['pnl'] <= 0]
    expectancy = (len(winners)/len(trades) * winners['pnl'].mean() + 
                  len(losers)/len(trades) * losers['pnl'].mean())
    if expectancy < 0:
        issues.append(f"⚠️  NEGATIVE EXPECTANCY: ${expectancy:.2f} per trade")
    
    # Issue 3: Poor win/loss ratio
    if len(winners) > 0 and len(losers) > 0:
        wl_ratio = abs(winners['pnl'].mean() / losers['pnl'].mean())
        if wl_ratio < 1.0:
            issues.append(f"⚠️  POOR WIN/LOSS RATIO: {wl_ratio:.2f} (avg win smaller than avg loss)")
    
    # Issue 4: Very short duration (possible slippage/execution issues)
    avg_duration_minutes = (trades['exit_time'] - trades['entry_time']).dt.total_seconds().mean() / 60
    if avg_duration_minutes < 30:
        issues.append(f"⚠️  VERY SHORT TRADES: Avg {avg_duration_minutes:.1f} minutes (may indicate premature exits)")
    
    # Issue 5: Asymmetric stop/target hits
    stops = (trades['reason'] == 'stop').sum()
    targets = (trades['reason'] == 'target').sum()
    if stops > 0 and targets > 0:
        stop_target_ratio = stops / targets
        if stop_target_ratio > 2.5:
            issues.append(f"⚠️  ASYMMETRIC EXITS: {stop_target_ratio:.1f}x more stops than targets")
    
    if issues:
        print("\nIdentified Issues:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print("\n✓ No obvious execution issues detected")
    
    return issues


def compare_with_master_expectations():
    """Compare concurrent results with master pipeline expectations"""
    
    print("\n" + "="*80)
    print("COMPARISON WITH MASTER PIPELINE")
    print("="*80)
    
    # Load master results
    try:
        master = pd.read_csv('data/master_pipeline_results_20260209_155832.csv')
        master_row = master[(master['stop_atr'] == 1.5) & (master['rf_threshold'] == 0.5)].iloc[0]
        
        # Load concurrent results
        concurrent_trades = pd.read_csv('data/concurrent_backtest_trades_concurrent.csv')
        concurrent_wr = (concurrent_trades['pnl'] > 0).mean()
        concurrent_pnl = concurrent_trades['pnl'].sum()
        
        print(f"\nMaster Pipeline (Stop ATR=1.5, RF=0.5):")
        print(f"  Win Rate: {master_row['win_rate']*100:.1f}%")
        print(f"  Total Trades: {master_row['n_trades']:.0f}")
        print(f"  Total P&L: ${master_row['total_net_pnl']:,.2f}")
        print(f"  Expected Value: {master_row['ev']:.4f}")
        
        print(f"\nConcurrent Backtest (2024, $1M capital):")
        print(f"  Win Rate: {concurrent_wr*100:.1f}%")
        print(f"  Total Trades: {len(concurrent_trades):,}")
        print(f"  Total P&L: ${concurrent_pnl:,.2f}")
        
        print(f"\n📊 GAPS:")
        print(f"  Win Rate Gap: {(concurrent_wr - master_row['win_rate'])*100:.1f} percentage points")
        print(f"  Trade Count Gap: {len(concurrent_trades) - master_row['n_trades']:.0f} trades")
        print(f"  P&L Gap: ${concurrent_pnl - master_row['total_net_pnl']:,.2f}")
        
        # Hypothetical: What if concurrent had master's win rate?
        winners = concurrent_trades[concurrent_trades['pnl'] > 0]
        losers = concurrent_trades[concurrent_trades['pnl'] <= 0]
        
        if len(winners) > 0 and len(losers) > 0:
            hypothetical_wins = int(len(concurrent_trades) * master_row['win_rate'])
            hypothetical_losses = len(concurrent_trades) - hypothetical_wins
            hypothetical_pnl = (hypothetical_wins * winners['pnl'].mean() + 
                               hypothetical_losses * losers['pnl'].mean())
            
            print(f"\n💡 HYPOTHETICAL: If concurrent had master's {master_row['win_rate']*100:.1f}% win rate:")
            print(f"  Expected P&L: ${hypothetical_pnl:,.2f}")
            print(f"  Improvement: ${hypothetical_pnl - concurrent_pnl:,.2f}")
        
    except Exception as e:
        print(f"Could not load master results: {e}")


def main():
    print(f"\n{'='*80}")
    print("WIN RATE GAP INVESTIGATION")
    print(f"{'='*80}")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nGoal: Understand why concurrent backtest has 47.9% win rate")
    print(f"      vs master pipeline's 66.4% win rate (18.5 point gap)")
    
    # Run analyses
    trades = analyze_concurrent_trades()
    compare_stop_target_logic()
    issues = identify_potential_issues()
    compare_with_master_expectations()
    
    # Final summary
    print("\n" + "="*80)
    print("SUMMARY & NEXT STEPS")
    print("="*80)
    
    print("\nKey Findings:")
    print(f"1. Concurrent win rate: 47.9% (2,265 winners out of 4,728 trades)")
    print(f"2. Master win rate: 66.4% (expected for same configuration)")
    print(f"3. Gap: 18.5 percentage points")
    
    print("\nPossible Root Causes:")
    print("□ Stop/target placement logic differs from master")
    print("□ Entry price execution differs (concurrent uses close)")
    print("□ Stop hit detection on same bar as entry")
    print("□ Different bar-level data (high/low used for exits)")
    print("□ Forward-looking bias in master's labels")
    print("□ Position sizing differences")
    
    print("\nRecommended Next Steps:")
    print("1. Export master pipeline's actual trades for 2024")
    print("2. Compare entry/exit prices bar-by-bar")
    print("3. Check if stops are being hit on entry bar")
    print("4. Verify stop/target calculation matches master exactly")
    print("5. Test if master uses different entry logic (not just close)")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
