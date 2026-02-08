"""
P&L Projections - Monthly and Annual Estimates

Takes the calculated P&L results and projects them forward to estimate
monthly and annual returns based on observed trade frequency.
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np


def load_pnl_summary(filepath='data/pnl_summary_comparison.csv'):
    """Load the P&L summary comparison."""
    return pd.read_csv(filepath)


def load_trade_log(filepath='data/pnl_trade_log_rf_filtered.csv'):
    """Load detailed trade log."""
    df = pd.read_csv(filepath, parse_dates=['entry_time', 'exit_time'])
    return df


def analyze_trade_frequency(trades_df):
    """Analyze how often trades occur."""
    if len(trades_df) == 0:
        return {}
    
    # Calculate date range
    start_date = trades_df['entry_time'].min()
    end_date = trades_df['entry_time'].max()
    days_total = (end_date - start_date).days
    
    # Trading days (assuming ~252 per year, ~21 per month)
    trading_days = days_total * (252 / 365)
    
    total_trades = len(trades_df)
    trades_per_day = total_trades / trading_days if trading_days > 0 else 0
    trades_per_week = trades_per_day * 5
    trades_per_month = trades_per_day * 21
    trades_per_year = trades_per_day * 252
    
    return {
        'start_date': start_date,
        'end_date': end_date,
        'calendar_days': days_total,
        'trading_days': trading_days,
        'total_trades': total_trades,
        'trades_per_day': trades_per_day,
        'trades_per_week': trades_per_week,
        'trades_per_month': trades_per_month,
        'trades_per_year': trades_per_year,
    }


def project_monthly_pnl(summary_df, trades_df):
    """Project monthly P&L based on trade frequency."""
    
    print("="*70)
    print("MONTHLY & ANNUAL P&L PROJECTIONS")
    print("="*70)
    
    for idx, row in summary_df.iterrows():
        scenario = row['Scenario']
        
        # Load the appropriate trade log
        if 'Unfiltered' in scenario:
            trades_df = pd.read_csv('data/pnl_trade_log_unfiltered.csv', parse_dates=['entry_time', 'exit_time'])
        else:
            trades_df = pd.read_csv('data/pnl_trade_log_rf_filtered.csv', parse_dates=['entry_time', 'exit_time'])
        
        freq = analyze_trade_frequency(trades_df)
        
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario}")
        print(f"{'='*70}")
        
        print(f"\nBacktest Period:")
        print(f"  Start:           {freq['start_date'].strftime('%Y-%m-%d')}")
        print(f"  End:             {freq['end_date'].strftime('%Y-%m-%d')}")
        print(f"  Calendar Days:   {freq['calendar_days']:.0f}")
        print(f"  Trading Days:    {freq['trading_days']:.0f}")
        
        print(f"\nTrade Frequency:")
        print(f"  Total Trades:    {freq['total_trades']:,.0f}")
        print(f"  Per Day:         {freq['trades_per_day']:.2f}")
        print(f"  Per Week:        {freq['trades_per_week']:.1f}")
        print(f"  Per Month:       {freq['trades_per_month']:.0f}")
        print(f"  Per Year:        {freq['trades_per_year']:.0f}")
        
        # Projections
        total_pnl = row['Total_PnL']
        total_ev = row['Total_EV']
        ev_per_trade = row['EV_per_Trade']
        win_rate = row['Win_Rate']
        
        # Monthly projections
        monthly_pnl = (total_pnl / freq['trading_days']) * 21 if freq['trading_days'] > 0 else 0
        monthly_ev = ev_per_trade * freq['trades_per_month']
        
        # Annual projections
        annual_pnl = (total_pnl / freq['trading_days']) * 252 if freq['trading_days'] > 0 else 0
        annual_ev = ev_per_trade * freq['trades_per_year']
        
        print(f"\nP&L Performance:")
        print(f"  Backtest Total:  ${total_pnl:,.2f}")
        print(f"  Total EV:        ${total_ev:,.2f}")
        print(f"  EV per Trade:    ${ev_per_trade:.2f}")
        print(f"  Win Rate:        {win_rate*100:.1f}%")
        
        print(f"\nMonthly Projections (21 trading days):")
        print(f"  Expected Trades: {freq['trades_per_month']:.0f}")
        print(f"  Expected P&L:    ${monthly_pnl:,.2f}")
        print(f"  Expected EV:     ${monthly_ev:,.2f}")
        
        print(f"\nAnnual Projections (252 trading days):")
        print(f"  Expected Trades: {freq['trades_per_year']:.0f}")
        print(f"  Expected P&L:    ${annual_pnl:,.2f}")
        print(f"  Expected EV:     ${annual_ev:,.2f}")
        
        # Capital efficiency
        avg_capital = row.get('Avg_Capital_Used', 0)
        if avg_capital > 0:
            annual_return_pct = (annual_pnl / avg_capital) * 100
            print(f"\nCapital Efficiency:")
            print(f"  Avg Capital:     ${avg_capital:,.2f}")
            print(f"  Annual Return:   {annual_return_pct:.1f}%")


def compare_scenarios(summary_df):
    """Compare projections between scenarios."""
    
    print("\n" + "="*70)
    print("SCENARIO COMPARISON")
    print("="*70)
    
    if len(summary_df) < 2:
        print("Need at least 2 scenarios to compare")
        return
    
    # Load trade logs
    unfiltered_trades = pd.read_csv('data/pnl_trade_log_unfiltered.csv', parse_dates=['entry_time', 'exit_time'])
    filtered_trades = pd.read_csv('data/pnl_trade_log_rf_filtered.csv', parse_dates=['entry_time', 'exit_time'])
    
    unfiltered_freq = analyze_trade_frequency(unfiltered_trades)
    filtered_freq = analyze_trade_frequency(filtered_trades)
    
    unfiltered = summary_df.iloc[0]
    filtered = summary_df.iloc[1]
    
    # Monthly projections
    unfiltered_monthly = (unfiltered['Total_PnL'] / unfiltered_freq['trading_days']) * 21
    filtered_monthly = (filtered['Total_PnL'] / filtered_freq['trading_days']) * 21
    
    # Annual projections
    unfiltered_annual = (unfiltered['Total_PnL'] / unfiltered_freq['trading_days']) * 252
    filtered_annual = (filtered['Total_PnL'] / filtered_freq['trading_days']) * 252
    
    print(f"\n{'Metric':<30} {'Unfiltered':>18} {'RF Filtered':>18} {'Improvement':>15}")
    print("-"*85)
    
    print(f"{'Trades per Month':<30} {unfiltered_freq['trades_per_month']:>18.0f} {filtered_freq['trades_per_month']:>18.0f}")
    print(f"{'Monthly P&L':<30} ${unfiltered_monthly:>17,.0f} ${filtered_monthly:>17,.0f}")
    print(f"{'Annual P&L':<30} ${unfiltered_annual:>17,.0f} ${filtered_annual:>17,.0f}")
    print(f"{'Win Rate':<30} {unfiltered['Win_Rate']*100:>17.1f}% {filtered['Win_Rate']*100:>17.1f}%")
    print(f"{'EV per Trade':<30} ${unfiltered['EV_per_Trade']:>17.2f} ${filtered['EV_per_Trade']:>17.2f}")
    
    # Calculate improvements
    if unfiltered_monthly != 0:
        monthly_improvement = ((filtered_monthly - unfiltered_monthly) / abs(unfiltered_monthly)) * 100
        print(f"\nMonthly P&L Improvement: {monthly_improvement:+.1f}%")
    
    if unfiltered_annual != 0:
        annual_improvement = ((filtered_annual - unfiltered_annual) / abs(unfiltered_annual)) * 100
        print(f"Annual P&L Improvement:  {annual_improvement:+.1f}%")


if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("P&L PROJECTIONS REPORT")
    print("="*70)
    
    # Load summary
    try:
        summary_df = load_pnl_summary()
        print(f"\nLoaded summary with {len(summary_df)} scenarios")
    except FileNotFoundError:
        print("\nError: pnl_summary_comparison.csv not found!")
        print("Please run calculate_pnl_with_rf.py first")
        sys.exit(1)
    
    # Project monthly/annual P&L
    try:
        filtered_trades = load_trade_log('data/pnl_trade_log_rf_filtered.csv')
        project_monthly_pnl(summary_df, filtered_trades)
    except FileNotFoundError:
        print("\nError: Trade log files not found!")
        print("Please run calculate_pnl_with_rf.py first")
        sys.exit(1)
    
    # Compare scenarios
    compare_scenarios(summary_df)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
