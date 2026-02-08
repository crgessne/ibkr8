"""
Calculate P&L with Random Forest filtering.
Uses trained RF model to filter trades (RF >= 0.5 threshold).
Position sizing: 100 shares per trade, max $1M simultaneous exposure.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

print("="*90)
print("P&L CALCULATION: RF FILTERED VWAP REVERSAL STRATEGY")
print("="*90)
print("\nPosition Sizing Rules:")
print("  • 100 shares per trade")
print("  • Max simultaneous exposure: $1,000,000")
print("  • RF probability >= 0.5 filter")
print()

# Configuration
STOP_WIDTH = 0.25
TARGET_ATR = 3.0
SHARES_PER_TRADE = 100
MAX_SIMULTANEOUS_VALUE = 1_000_000
RF_THRESHOLD = 0.5

# Load data
print("Loading data...")
df = pd.read_csv('data/tsla_5min_10years_indicators.csv', parse_dates=['time'])
df = df.dropna()

# Split train/test
train = df[df['time'] < '2024-01-01'].copy()
test = df[df['time'] >= '2024-01-01'].copy()

print(f"Train period: {train['time'].min()} to {train['time'].max()} ({len(train):,} bars)")
print(f"Test period:  {test['time'].min()} to {test['time'].max()} ({len(test):,} bars)")
print()

# Define cleaned feature set (35 features)
cleaned_features = [
    'vwap_width_atr', 'vwap_helping', 'bars_from_vwap', 'vwap_slope_5',
    'vol_at_extension', 'vwap_slope', 'rel_vol', 'bar_range_atr',
    'bb_pct', 'vol_trend_3', 'dist_bb_lower_atr', 'reversal_close_position',
    'vwap_dist_delta_5', 'vwap_dist_delta_3', 'dist_bb_upper_atr',
    'rsi_extremity', 'extension_velocity_3', 'extension_velocity_5',
    'rsi_slope_3', 'rsi_slope_5', 'bb_extension', 'bb_extension_upper',
    'bb_extension_lower', 'vol_declining', 'price_slope_3', 'price_slope_5',
    'momentum_divergence_3', 'momentum_divergence_5',
    'upper_wick_pct', 'lower_wick_pct', 'body_pct', 'reversal_wick',
    'range_vs_prev', 'range_shrinking', 'consecutive_shrinking', 'close_position'
]

# Filter to available features
available_features = [f for f in cleaned_features if f in df.columns]
print(f"Using {len(available_features)} features (from {len(cleaned_features)} requested)")

# Train RF model if not already trained
model_path = 'data/rf_model_025atr.pkl'

if os.path.exists(model_path):
    print(f"\nLoading existing model from {model_path}...")
    with open(model_path, 'rb') as f:
        rf_model = pickle.load(f)
else:
    print(f"\nTraining RF model...")
    label_col = 'win_025'  # 0.25 ATR stop width
    
    if label_col not in train.columns:
        print(f"ERROR: Label column '{label_col}' not found!")
        print("Available columns:", [c for c in train.columns if 'win' in c])
        exit(1)
    
    X_train = train[available_features]
    y_train = train[label_col]
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=100,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1
    )
    
    print("  Training...")
    rf_model.fit(X_train, y_train)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"  Model saved to {model_path}")

# Get RF predictions for test set
print("\nGenerating RF predictions...")
X_test = test[available_features]
test['rf_prob'] = rf_model.predict_proba(X_test)[:, 1]  # Probability of winning

print(f"RF probabilities generated for {len(test):,} bars")
print(f"  Mean probability: {test['rf_prob'].mean():.3f}")
print(f"  Bars >= {RF_THRESHOLD}: {(test['rf_prob'] >= RF_THRESHOLD).sum():,} ({(test['rf_prob'] >= RF_THRESHOLD).sum()/len(test)*100:.1f}%)")
print()

print("Strategy Configuration:")
print(f"  Stop: {STOP_WIDTH} ATR")
print(f"  Target: {TARGET_ATR} ATR (VWAP)")
print(f"  RF Filter: >= {RF_THRESHOLD}")
print(f"  Shares per trade: {SHARES_PER_TRADE}")
print(f"  Max simultaneous exposure: ${MAX_SIMULTANEOUS_VALUE:,}")
print()

# Calculate entry signals
test['atr'] = test['atr'].ffill()
test['vwap_width_atr'] = abs((test['close'] - test['vwap']) / test['atr'])

# Entry criteria: RF >= threshold AND reasonable distance from VWAP
test['potential_long'] = (
    (test['close'] < test['vwap']) & 
    (test['vwap_width_atr'] >= 0.5) & 
    (test['vwap_width_atr'] <= 2.0) &
    (test['rf_prob'] >= RF_THRESHOLD)
)

test['potential_short'] = (
    (test['close'] > test['vwap']) & 
    (test['vwap_width_atr'] >= 0.5) & 
    (test['vwap_width_atr'] <= 2.0) &
    (test['rf_prob'] >= RF_THRESHOLD)
)

print(f"Potential trades after RF filter:")
print(f"  LONG:  {test['potential_long'].sum():,} bars")
print(f"  SHORT: {test['potential_short'].sum():,} bars")
print()

# Calculate stop and target prices
test['long_stop_price'] = test['close'] - (STOP_WIDTH * test['atr'])
test['long_target_price'] = test['vwap']
test['short_stop_price'] = test['close'] + (STOP_WIDTH * test['atr'])
test['short_target_price'] = test['vwap']
test['long_risk_per_share'] = test['close'] - test['long_stop_price']
test['short_risk_per_share'] = test['short_stop_price'] - test['close']

print("Simulating trades...")

# Track active trades
active_trades = []
completed_trades = []
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
            if low <= trade['stop_price']:
                hit_stop = True
                exit_price = trade['stop_price']
                exit_reason = 'STOP'
                pnl = (exit_price - trade['entry_price']) * trade['shares']
            elif high >= trade['target_price']:
                hit_target = True
                exit_price = trade['target_price']
                exit_reason = 'TARGET'
                pnl = (exit_price - trade['entry_price']) * trade['shares']
        else:  # SHORT
            if high >= trade['stop_price']:
                hit_stop = True
                exit_price = trade['stop_price']
                exit_reason = 'STOP'
                pnl = (trade['entry_price'] - exit_price) * trade['shares']
            elif low <= trade['target_price']:
                hit_target = True
                exit_price = trade['target_price']
                exit_reason = 'TARGET'
                pnl = (trade['entry_price'] - exit_price) * trade['shares']
        
        if hit_stop or hit_target:
            trade['exit_time'] = current_time
            trade['exit_price'] = exit_price
            trade['exit_reason'] = exit_reason
            trade['pnl'] = pnl
            trade['pnl_pct'] = (pnl / (trade['entry_price'] * trade['shares'])) * 100
            trade['r_multiple'] = pnl / (trade['risk_per_share'] * trade['shares'])
            completed_trades.append(trade)
            trades_to_remove.append(trade)
            current_exposure -= trade['entry_price'] * trade['shares']
    
    for trade in trades_to_remove:
        active_trades.remove(trade)
    
    # Check for new entries
    trade_value = current_price * SHARES_PER_TRADE
    
    if row['potential_long'] and (current_exposure + trade_value <= MAX_SIMULTANEOUS_VALUE):
        new_trade = {
            'entry_time': current_time,
            'entry_price': current_price,
            'direction': 'LONG',
            'shares': SHARES_PER_TRADE,
            'stop_price': row['long_stop_price'],
            'target_price': row['long_target_price'],
            'risk_per_share': row['long_risk_per_share'],
            'rf_prob': row['rf_prob'],
            'atr': row['atr'],
            'vwap_width_atr': row['vwap_width_atr']
        }
        active_trades.append(new_trade)
        current_exposure += trade_value
    elif row['potential_short'] and (current_exposure + trade_value <= MAX_SIMULTANEOUS_VALUE):
        new_trade = {
            'entry_time': current_time,
            'entry_price': current_price,
            'direction': 'SHORT',
            'shares': SHARES_PER_TRADE,
            'stop_price': row['short_stop_price'],
            'target_price': row['short_target_price'],
            'risk_per_share': row['short_risk_per_share'],
            'rf_prob': row['rf_prob'],
            'atr': row['atr'],
            'vwap_width_atr': row['vwap_width_atr']
        }
        active_trades.append(new_trade)
        current_exposure += trade_value

# Close remaining trades
last_price = test.iloc[-1]['close']
last_time = test.iloc[-1]['time']
for trade in active_trades:
    exit_price = last_price
    if trade['direction'] == 'LONG':
        pnl = (exit_price - trade['entry_price']) * trade['shares']
    else:
        pnl = (trade['entry_price'] - exit_price) * trade['shares']
    
    trade['exit_time'] = last_time
    trade['exit_price'] = exit_price
    trade['exit_reason'] = 'EOD'
    trade['pnl'] = pnl
    trade['pnl_pct'] = (pnl / (trade['entry_price'] * trade['shares'])) * 100
    trade['r_multiple'] = pnl / (trade['risk_per_share'] * trade['shares'])
    completed_trades.append(trade)

print(f"\nCompleted trades: {len(completed_trades):,}")

if len(completed_trades) == 0:
    print("No trades executed!")
else:
    trades_df = pd.DataFrame(completed_trades)
    
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
    
    # Drawdown
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    trades_df['running_max'] = trades_df['cumulative_pnl'].cummax()
    trades_df['drawdown'] = trades_df['cumulative_pnl'] - trades_df['running_max']
    max_drawdown = trades_df['drawdown'].min()
    
    print("\n" + "="*90)
    print("PERFORMANCE SUMMARY (RF FILTERED)")
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
    profit_factor = abs(winners['pnl'].sum() / losers['pnl'].sum()) if len(losers) > 0 and losers['pnl'].sum() != 0 else np.inf
    print(f"  Avg risk/trade:   ${avg_risk:,.2f}")
    print(f"  Profit factor:    {profit_factor:.2f}")
    print(f"  Sharpe (trades):  {avg_trade / trades_df['pnl'].std() if trades_df['pnl'].std() > 0 else 0:.2f}")
    
    # Time metrics
    days = (test['time'].max() - test['time'].min()).days
    years = days / 365.25
    annual_pnl = total_pnl / years if years > 0 else total_pnl
    
    print(f"\nTime Metrics:")
    print(f"  Test period:      {days} days ({years:.2f} years)")
    print(f"  Trades per day:   {len(trades_df) / days:.1f}")
    print(f"  Annual P&L:       ${annual_pnl:,.2f}")
    
    # Capital efficiency
    avg_capital_used = trades_df['entry_price'].mean() * SHARES_PER_TRADE
    
    print(f"\nCapital Efficiency:")
    print(f"  Avg capital/trade: ${avg_capital_used:,.2f}")
    print(f"  Max exposure:      ${MAX_SIMULTANEOUS_VALUE:,}")
    
    # Direction breakdown
    print(f"\nDirection Breakdown:")
    long_trades = trades_df[trades_df['direction'] == 'LONG']
    short_trades = trades_df[trades_df['direction'] == 'SHORT']
    
    if len(long_trades) > 0:
        long_wr = len(long_trades[long_trades['pnl']>0])/len(long_trades)*100
        print(f"  LONG:  {len(long_trades):,} trades, ${long_trades['pnl'].sum():,.2f} P&L, {long_wr:.1f}% WR")
    if len(short_trades) > 0:
        short_wr = len(short_trades[short_trades['pnl']>0])/len(short_trades)*100
        print(f"  SHORT: {len(short_trades):,} trades, ${short_trades['pnl'].sum():,.2f} P&L, {short_wr:.1f}% WR")
    
    # Exit reasons
    print(f"\nExit Reasons:")
    for reason in trades_df['exit_reason'].unique():
        reason_trades = trades_df[trades_df['exit_reason'] == reason]
        print(f"  {reason:8s}: {len(reason_trades):,} trades ({len(reason_trades)/len(trades_df)*100:.1f}%)")
    
    # RF probability analysis
    print(f"\nRF Probability Analysis:")
    print(f"  Avg RF prob:      {trades_df['rf_prob'].mean():.3f}")
    print(f"  Min RF prob:      {trades_df['rf_prob'].min():.3f}")
    print(f"  Max RF prob:      {trades_df['rf_prob'].max():.3f}")
    
    # Save results
    trades_df.to_csv('data/pnl_trade_log_rf_filtered.csv', index=False)
    print(f"\n[OK] Saved trade log to: data/pnl_trade_log_rf_filtered.csv")
    
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
        'profit_factor': profit_factor
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('data/pnl_summary_rf_filtered.csv', index=False)
    print(f"[OK] Saved summary to: data/pnl_summary_rf_filtered.csv")
    
    print("\n" + "="*90)
    print("P&L ANALYSIS COMPLETE")
    print("="*90)
    print(f"\n[$$] BOTTOM LINE: ${total_pnl:,.2f} profit from {len(trades_df):,} trades")
    print(f"[>>] Annual Projection: ${annual_pnl:,.2f}")
    print(f"[->] Average per trade: ${avg_trade:,.2f} ({avg_r:+.3f}R)")
    print(f"[%%] Win Rate: {win_rate:.1f}% (backtest: 25.3%)")
    print("="*90)
