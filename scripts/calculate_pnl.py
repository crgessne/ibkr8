"""
Calculate realistic P&L for RF VWAP reversal strategy.
Position sizing: 100 shares per trade, max $1M simultaneous exposure.
"""
import pandas as pd
import numpy as np

print("="*90)
print("P&L CALCULATION: RF VWAP REVERSAL STRATEGY")
print("="*90)
print("\nPosition Sizing Rules:")
print("  • 100 shares per trade")
print("  • Max simultaneous exposure: $1,000,000")
print("  • Dynamic position management")
print()

# Load data with indicators
print("Loading data...")
df = pd.read_csv('data/tsla_5min_10years_indicators.csv', parse_dates=['time'])
df = df.dropna()

# Focus on test period (2024+)
test = df[df['time'] >= '2024-01-01'].copy()
print(f"Test period: {test['time'].min()} to {test['time'].max()}")
print(f"Test bars: {len(test):,}")
print()

# Load RF results to get probabilities (we'll simulate this)
# In production, you'd load actual RF predictions
# For now, we'll use the filtering logic

# Best setup: 0.25 ATR stop
STOP_WIDTH = 0.25
TARGET_ATR = 3.0
SHARES_PER_TRADE = 100
MAX_SIMULTANEOUS_VALUE = 1_000_000

print("Strategy Configuration:")
print(f"  Stop: {STOP_WIDTH} ATR")
print(f"  Target: {TARGET_ATR} ATR (VWAP)")
print(f"  Shares per trade: {SHARES_PER_TRADE}")
print(f"  Max simultaneous exposure: ${MAX_SIMULTANEOUS_VALUE:,}")
print()

# Calculate entry signals and trade metrics
test['atr'] = test['atr'].ffill()
test['vwap_width_atr'] = abs((test['close'] - test['vwap']) / test['atr'])

# Identify potential trades (simplified - in production use RF >= 0.5)
# For now, use sweet spot: 0.5-2.0 ATR from VWAP
test['potential_long'] = (test['close'] < test['vwap']) & (test['vwap_width_atr'] >= 0.5) & (test['vwap_width_atr'] <= 2.0)
test['potential_short'] = (test['close'] > test['vwap']) & (test['vwap_width_atr'] >= 0.5) & (test['vwap_width_atr'] <= 2.0)

# Calculate stop and target prices
test['long_stop_price'] = test['close'] - (STOP_WIDTH * test['atr'])
test['long_target_price'] = test['vwap']
test['short_stop_price'] = test['close'] + (STOP_WIDTH * test['atr'])
test['short_target_price'] = test['vwap']

# Risk per share
test['long_risk_per_share'] = test['close'] - test['long_stop_price']
test['short_risk_per_share'] = test['short_stop_price'] - test['close']

print("Simulating trades...")

# Track active trades
active_trades = []
completed_trades = []
daily_pnl = {}
current_exposure = 0.0

for idx, row in test.iterrows():
    current_time = row['time']
    current_date = current_time.date()
    current_price = row['close']
    high = row['high']
    low = row['low']
    
    # Check active trades for exits
    trades_to_remove = []
    for trade in active_trades:
        hit_stop = False
        hit_target = False
        pnl = 0.0
        exit_price = None
        exit_reason = None
        
        if trade['direction'] == 'LONG':
            # Check if stop hit
            if low <= trade['stop_price']:
                hit_stop = True
                exit_price = trade['stop_price']
                exit_reason = 'STOP'
                pnl = (exit_price - trade['entry_price']) * trade['shares']
            # Check if target hit
            elif high >= trade['target_price']:
                hit_target = True
                exit_price = trade['target_price']
                exit_reason = 'TARGET'
                pnl = (exit_price - trade['entry_price']) * trade['shares']
        else:  # SHORT
            # Check if stop hit
            if high >= trade['stop_price']:
                hit_stop = True
                exit_price = trade['stop_price']
                exit_reason = 'STOP'
                pnl = (trade['entry_price'] - exit_price) * trade['shares']
            # Check if target hit
            elif low <= trade['target_price']:
                hit_target = True
                exit_price = trade['target_price']
                exit_reason = 'TARGET'
                pnl = (trade['entry_price'] - exit_price) * trade['shares']
        
        if hit_stop or hit_target:
            # Record completed trade
            trade['exit_time'] = current_time
            trade['exit_price'] = exit_price
            trade['exit_reason'] = exit_reason
            trade['pnl'] = pnl
            trade['pnl_pct'] = (pnl / (trade['entry_price'] * trade['shares'])) * 100
            trade['r_multiple'] = pnl / (trade['risk_per_share'] * trade['shares'])
            
            completed_trades.append(trade)
            trades_to_remove.append(trade)
            
            # Update exposure
            current_exposure -= trade['entry_price'] * trade['shares']
            
            # Track daily P&L
            if current_date not in daily_pnl:
                daily_pnl[current_date] = 0.0
            daily_pnl[current_date] += pnl
    
    # Remove completed trades
    for trade in trades_to_remove:
        active_trades.remove(trade)
    
    # Check for new entries
    trade_value = current_price * SHARES_PER_TRADE
    
    # Long entry
    if row['potential_long'] and (current_exposure + trade_value <= MAX_SIMULTANEOUS_VALUE):
        new_trade = {
            'entry_time': current_time,
            'entry_price': current_price,
            'direction': 'LONG',
            'shares': SHARES_PER_TRADE,
            'stop_price': row['long_stop_price'],
            'target_price': row['long_target_price'],
            'risk_per_share': row['long_risk_per_share'],
            'atr': row['atr'],
            'vwap_width_atr': row['vwap_width_atr']
        }
        active_trades.append(new_trade)
        current_exposure += trade_value
    
    # Short entry
    elif row['potential_short'] and (current_exposure + trade_value <= MAX_SIMULTANEOUS_VALUE):
        new_trade = {
            'entry_time': current_time,
            'entry_price': current_price,
            'direction': 'SHORT',
            'shares': SHARES_PER_TRADE,
            'stop_price': row['short_stop_price'],
            'target_price': row['short_target_price'],
            'risk_per_share': row['short_risk_per_share'],
            'atr': row['atr'],
            'vwap_width_atr': row['vwap_width_atr']
        }
        active_trades.append(new_trade)
        current_exposure += trade_value

# Close any remaining open trades at last price
last_price = test.iloc[-1]['close']
last_time = test.iloc[-1]['time']
for trade in active_trades:
    if trade['direction'] == 'LONG':
        exit_price = last_price
        pnl = (exit_price - trade['entry_price']) * trade['shares']
    else:
        exit_price = last_price
        pnl = (trade['entry_price'] - exit_price) * trade['shares']
    
    trade['exit_time'] = last_time
    trade['exit_price'] = exit_price
    trade['exit_reason'] = 'EOD'
    trade['pnl'] = pnl
    trade['pnl_pct'] = (pnl / (trade['entry_price'] * trade['shares'])) * 100
    trade['r_multiple'] = pnl / (trade['risk_per_share'] * trade['shares'])
    completed_trades.append(trade)

print(f"\nCompleted trades: {len(completed_trades):,}")

# Convert to DataFrame for analysis
trades_df = pd.DataFrame(completed_trades)

if len(trades_df) == 0:
    print("No trades executed!")
else:
    # Calculate statistics
    winners = trades_df[trades_df['pnl'] > 0]
    losers = trades_df[trades_df['pnl'] <= 0]
    
    total_pnl = trades_df['pnl'].sum()
    win_count = len(winners)
    loss_count = len(losers)
    win_rate = win_count / len(trades_df) * 100
    
    avg_win = winners['pnl'].mean() if len(winners) > 0 else 0
    avg_loss = losers['pnl'].mean() if len(losers) > 0 else 0
    avg_trade = trades_df['pnl'].mean()
    
    max_win = trades_df['pnl'].max()
    max_loss = trades_df['pnl'].min()
    
    # R-multiple statistics
    avg_r = trades_df['r_multiple'].mean()
    total_r = trades_df['r_multiple'].sum()
    
    # Calculate drawdown
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    trades_df['running_max'] = trades_df['cumulative_pnl'].cummax()
    trades_df['drawdown'] = trades_df['cumulative_pnl'] - trades_df['running_max']
    max_drawdown = trades_df['drawdown'].min()
    
    print("\n" + "="*90)
    print("PERFORMANCE SUMMARY")
    print("="*90)
    
    print(f"\nTrade Statistics:")
    print(f"  Total trades:     {len(trades_df):,}")
    print(f"  Winners:          {win_count:,} ({win_rate:.1f}%)")
    print(f"  Losers:           {loss_count:,} ({100-win_rate:.1f}%)")
    
    print(f"\nP&L Statistics:")
    print(f"  Total P&L:        ${total_pnl:,.2f}")
    print(f"  Average trade:    ${avg_trade:,.2f}")
    print(f"  Average winner:   ${avg_win:,.2f}")
    print(f"  Average loser:    ${avg_loss:,.2f}")
    print(f"  Largest win:      ${max_win:,.2f}")
    print(f"  Largest loss:     ${max_loss:,.2f}")
    print(f"  Max drawdown:     ${max_drawdown:,.2f}")
    
    print(f"\nR-Multiple Statistics:")
    print(f"  Total R:          {total_r:,.2f}R")
    print(f"  Average R:        {avg_r:,.3f}R per trade")
    
    print(f"\nRisk Metrics:")
    avg_risk = trades_df['risk_per_share'].mean() * SHARES_PER_TRADE
    print(f"  Avg risk/trade:   ${avg_risk:,.2f}")
    print(f"  Profit factor:    {abs(winners['pnl'].sum() / losers['pnl'].sum()) if len(losers) > 0 else np.inf:.2f}")
    print(f"  Sharpe (trades):  {avg_trade / trades_df['pnl'].std() if trades_df['pnl'].std() > 0 else 0:.2f}")
    
    # Calculate annualized return
    days = (test['time'].max() - test['time'].min()).days
    years = days / 365.25
    annual_pnl = total_pnl / years if years > 0 else total_pnl
    
    print(f"\nTime Metrics:")
    print(f"  Test period:      {days} days ({years:.2f} years)")
    print(f"  Trades per day:   {len(trades_df) / days:.1f}")
    print(f"  Annual P&L:       ${annual_pnl:,.2f}")
    
    # Capital efficiency
    avg_capital_used = trades_df['entry_price'].mean() * SHARES_PER_TRADE
    avg_concurrent = len(trades_df) / len(test) * avg_capital_used
    
    print(f"\nCapital Efficiency:")
    print(f"  Avg capital/trade: ${avg_capital_used:,.2f}")
    print(f"  Max exposure:      ${MAX_SIMULTANEOUS_VALUE:,}")
    print(f"  Capital used:      {avg_concurrent / MAX_SIMULTANEOUS_VALUE * 100:.1f}%")
    
    # Direction breakdown
    print(f"\nDirection Breakdown:")
    long_trades = trades_df[trades_df['direction'] == 'LONG']
    short_trades = trades_df[trades_df['direction'] == 'SHORT']
    
    if len(long_trades) > 0:
        print(f"  LONG:  {len(long_trades):,} trades, ${long_trades['pnl'].sum():,.2f} P&L, {len(long_trades[long_trades['pnl']>0])/len(long_trades)*100:.1f}% WR")
    if len(short_trades) > 0:
        print(f"  SHORT: {len(short_trades):,} trades, ${short_trades['pnl'].sum():,.2f} P&L, {len(short_trades[short_trades['pnl']>0])/len(short_trades)*100:.1f}% WR")
    
    # Exit reason breakdown
    print(f"\nExit Reasons:")
    for reason in trades_df['exit_reason'].unique():
        reason_trades = trades_df[trades_df['exit_reason'] == reason]
        print(f"  {reason:8s}: {len(reason_trades):,} trades ({len(reason_trades)/len(trades_df)*100:.1f}%)")
    
    # Monthly P&L
    trades_df['month'] = pd.to_datetime(trades_df['exit_time']).dt.to_period('M')
    monthly_pnl = trades_df.groupby('month')['pnl'].sum()
    
    print(f"\n" + "="*90)
    print("MONTHLY P&L")
    print("="*90)
    for month, pnl in monthly_pnl.items():
        month_trades = len(trades_df[trades_df['month'] == month])
        print(f"  {month}  ${pnl:>12,.2f}  ({month_trades:>4,} trades)")
      # Save results
    trades_df.to_csv('data/pnl_trade_log.csv', index=False)
    print(f"\n[OK] Saved trade log to: data/pnl_trade_log.csv")
    
    # Create summary
    summary = {
        'total_trades': len(trades_df),
        'win_count': win_count,
        'loss_count': loss_count,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_trade': avg_trade,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_win': max_win,
        'max_loss': max_loss,
        'max_drawdown': max_drawdown,
        'total_r': total_r,
        'avg_r': avg_r,
        'annual_pnl': annual_pnl,
        'profit_factor': abs(winners['pnl'].sum() / losers['pnl'].sum()) if len(losers) > 0 else np.inf
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('data/pnl_summary.csv', index=False)
    print(f"[OK] Saved summary to: data/pnl_summary.csv")
    
    print("\n" + "="*90)
    print("SAMPLE TRADES (First 10)")
    print("="*90)
    display_cols = ['entry_time', 'direction', 'entry_price', 'exit_price', 'exit_reason', 'pnl', 'r_multiple']
    print(trades_df[display_cols].head(10).to_string(index=False))
      print("\n" + "="*90)
    print("P&L ANALYSIS COMPLETE")
    print("="*90)
    print(f"\n[$$] BOTTOM LINE: ${total_pnl:,.2f} profit from {len(trades_df):,} trades")
    print(f"[>>] Annual Projection: ${annual_pnl:,.2f}")
    print(f"[->] Average per trade: ${avg_trade:,.2f} ({avg_r:+.3f}R)")
    print("="*90)
