"""
Calculate P&L with RF Model Filtering

Trains RF model on historical data, generates probabilities for test set,
and calculates dollar-based P&L with 100 shares per trade.

This integrates:
1. Indicator calculation (indicators.py)
2. RF model training and prediction
3. P&L calculation with position sizing (pnl_calculator.py)
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from indicators import calc_all_indicators
from pnl_calculator import calculate_pnl, print_pnl_summary, save_trade_log, PnLConfig
from label_generator import generate_outcome_labels


# ============================================================================
# CONFIGURATION
# ============================================================================

# Data file
DATA_FILE = 'data/tsla_5min_10years_indicators.csv'

# P&L Configuration
SHARES_PER_TRADE = 100
MAX_SIMULTANEOUS_EXPOSURE = 1_000_000
STOP_ATR_WIDTH = 0.25
TARGET_RR = 6.0

# RF Configuration
RF_THRESHOLD = 0.5  # Minimum probability to take trade
TRAIN_TEST_SPLIT_DATE = '2024-01-01'  # Train before, test after

# Feature set (cleaned features from your analysis)
FEATURE_COLS = [
    'vwap_width_atr', 'vwap_helping', 'bars_from_vwap',
    'vwap_slope_5', 'vol_at_extension',
    'rsi_extremity', 'momentum_divergence_5',
    'reversal_wick', 'reversal_close_position',
    'bar_range_atr', 'range_vs_prev', 'consecutive_shrinking',
    'extension_velocity_5', 'extension_accel',
    'vol_declining', 'vol_trend_3',
    'bb_extension_abs', 'dist_bb_lower_atr',
    'close_position', 'body_pct', 'upper_wick_pct', 'lower_wick_pct',
    'rsi_slope_5', 'momentum_divergence_3',
    'dist_bb_upper_atr', 'extension_velocity_3',
    'price_to_vwap_atr', 'vwap_dist_delta_5', 'vwap_dist_delta_3',
    'rsi', 'bb_pct', 'rel_vol', 'price_slope_5', 'price_slope_3', 'rsi_slope_3',
]


# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("="*70)
print("RF + P&L CALCULATOR")
print("="*70)

# Load data
print(f"\nLoading data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

if 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

print(f"Loaded {len(df):,} bars")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Check if indicators exist, if not calculate them
required_cols = ['close', 'vwap', 'atr', 'vwap_width_atr']
missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    print(f"\nMissing indicators: {missing_cols}")
    print("Calculating indicators...")
    df = calc_all_indicators(df)
    print("Indicators calculated")

# Generate outcome labels if they don't exist
if 'win_025' not in df.columns:
    print("\nGenerating outcome labels...")
    df = generate_outcome_labels(
        df,
        stop_atr_width=STOP_ATR_WIDTH,
        target_atr_multiple=TARGET_RR * STOP_ATR_WIDTH,
        max_bars_held=100
    )
    print("Outcome labels generated")

# Filter to VWAP reversal zone (0.5 - 2.0 ATR)
df_zone = df[
    (df['vwap_width_atr'] >= 0.5) & 
    (df['vwap_width_atr'] <= 2.0)
].copy()

print(f"\nFiltered to reversal zone: {len(df_zone):,} bars ({len(df_zone)/len(df)*100:.1f}%)")

# ============================================================================
# SPLIT DATA: TRAIN vs TEST
# ============================================================================

train = df_zone[df_zone.index < TRAIN_TEST_SPLIT_DATE].copy()
test = df_zone[df_zone.index >= TRAIN_TEST_SPLIT_DATE].copy()

print(f"\nTrain set: {len(train):,} bars ({train.index.min()} to {train.index.max()})")
print(f"Test set:  {len(test):,} bars ({test.index.min()} to {test.index.max()})")

# ============================================================================
# TRAIN RF MODEL
# ============================================================================

print("\n" + "="*70)
print("TRAINING RF MODEL")
print("="*70)

# Prepare features and labels
available_features = [f for f in FEATURE_COLS if f in train.columns]
print(f"\nUsing {len(available_features)} features")

X_train = train[available_features].fillna(0)
y_train = train['win_025'].fillna(0).astype(int)

X_test = test[available_features].fillna(0)
y_test = test['win_025'].fillna(0).astype(int)

print(f"Train samples: {len(X_train):,} (Win rate: {y_train.mean()*100:.1f}%)")
print(f"Test samples:  {len(X_test):,} (Win rate: {y_test.mean()*100:.1f}%)")

# Train RF model
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=100,
    min_samples_leaf=50,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
print("Model trained successfully")

# Generate predictions
print("\nGenerating predictions...")
test['rf_prob'] = rf.predict_proba(X_test)[:, 1]
print(f"Average RF probability: {test['rf_prob'].mean():.3f}")

# ============================================================================
# CALCULATE P&L: UNFILTERED (ALL TRADES IN ZONE)
# ============================================================================

print("\n" + "="*70)
print("SCENARIO 1: UNFILTERED (ALL TRADES IN ZONE)")
print("="*70)

# Entry signal: any bar in the reversal zone
unfiltered_signals = pd.Series(True, index=test.index)

config = PnLConfig(
    shares_per_trade=SHARES_PER_TRADE,
    max_simultaneous_exposure=MAX_SIMULTANEOUS_EXPOSURE,
    stop_atr_width=STOP_ATR_WIDTH,
    target_rr=TARGET_RR
)

trades_unfiltered, summary_unfiltered = calculate_pnl(
    df=test,
    entry_signals=unfiltered_signals,
    config=config
)

print_pnl_summary(summary_unfiltered)

# ============================================================================
# CALCULATE P&L: RF FILTERED (RF PROBABILITY >= THRESHOLD)
# ============================================================================

print("\n" + "="*70)
print(f"SCENARIO 2: RF FILTERED (Probability >= {RF_THRESHOLD})")
print("="*70)

# Entry signal: RF probability >= threshold
filtered_signals = test['rf_prob'] >= RF_THRESHOLD

print(f"\nSignals passed filter: {filtered_signals.sum():,} / {len(filtered_signals):,} ({filtered_signals.mean()*100:.1f}%)")

trades_filtered, summary_filtered = calculate_pnl(
    df=test,
    entry_signals=filtered_signals,
    config=config
)

print_pnl_summary(summary_filtered)

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "="*70)
print("COMPARISON: UNFILTERED vs RF FILTERED")
print("="*70)

print(f"\n{'Metric':<25} {'Unfiltered':>15} {'RF Filtered':>15} {'Improvement':>15}")
print("-"*70)

metrics = [
    ('Total Trades', summary_unfiltered.total_trades, summary_filtered.total_trades),
    ('Win Rate', f"{summary_unfiltered.win_rate*100:.1f}%", f"{summary_filtered.win_rate*100:.1f}%"),
    ('Total P&L', f"${summary_unfiltered.total_pnl:,.0f}", f"${summary_filtered.total_pnl:,.0f}"),
    ('Total EV', f"${summary_unfiltered.total_ev:,.0f}", f"${summary_filtered.total_ev:,.0f}"),
    ('EV per Trade', f"${summary_unfiltered.ev_per_trade:.2f}", f"${summary_filtered.ev_per_trade:.2f}"),
    ('Avg P&L per Trade', f"${summary_unfiltered.avg_pnl_per_trade:.2f}", f"${summary_filtered.avg_pnl_per_trade:.2f}"),
    ('Avg Win', f"${summary_unfiltered.avg_win:.2f}", f"${summary_filtered.avg_win:.2f}"),
    ('Avg Loss', f"${summary_unfiltered.avg_loss:.2f}", f"${summary_filtered.avg_loss:.2f}"),
    ('Profit Factor', f"{summary_unfiltered.profit_factor:.2f}", f"{summary_filtered.profit_factor:.2f}"),
    ('Sharpe Ratio', f"{summary_unfiltered.sharpe_ratio:.2f}", f"{summary_filtered.sharpe_ratio:.2f}"),
    ('Max Drawdown', f"${summary_unfiltered.max_drawdown:,.0f}", f"${summary_filtered.max_drawdown:,.0f}"),
]

for metric, unfilt, filt in metrics:
    print(f"{metric:<25} {unfilt:>15} {filt:>15}")

# Calculate improvement
if summary_unfiltered.total_pnl != 0:
    pnl_improvement = (summary_filtered.total_pnl - summary_unfiltered.total_pnl) / abs(summary_unfiltered.total_pnl) * 100
    print(f"\nP&L Improvement: {pnl_improvement:+.1f}%")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save trade logs
save_trade_log(trades_unfiltered, 'data/pnl_trade_log_unfiltered.csv')
save_trade_log(trades_filtered, 'data/pnl_trade_log_rf_filtered.csv')

# Save summary
summary_df = pd.DataFrame({
    'Scenario': ['Unfiltered', 'RF Filtered'],
    'Total_Trades': [summary_unfiltered.total_trades, summary_filtered.total_trades],
    'Win_Rate': [summary_unfiltered.win_rate, summary_filtered.win_rate],
    'Total_PnL': [summary_unfiltered.total_pnl, summary_filtered.total_pnl],
    'Total_EV': [summary_unfiltered.total_ev, summary_filtered.total_ev],
    'EV_per_Trade': [summary_unfiltered.ev_per_trade, summary_filtered.ev_per_trade],
    'Avg_PnL_per_Trade': [summary_unfiltered.avg_pnl_per_trade, summary_filtered.avg_pnl_per_trade],
    'Avg_Win': [summary_unfiltered.avg_win, summary_filtered.avg_win],
    'Avg_Loss': [summary_unfiltered.avg_loss, summary_filtered.avg_loss],
    'Profit_Factor': [summary_unfiltered.profit_factor, summary_filtered.profit_factor],
    'Sharpe_Ratio': [summary_unfiltered.sharpe_ratio, summary_filtered.sharpe_ratio],
    'Max_Drawdown': [summary_unfiltered.max_drawdown, summary_filtered.max_drawdown],
    'Total_Commission': [summary_unfiltered.total_commission, summary_filtered.total_commission],
    'Total_Slippage': [summary_unfiltered.total_slippage, summary_filtered.total_slippage],
})

summary_df.to_csv('data/pnl_summary_comparison.csv', index=False)
print("Summary saved to: data/pnl_summary_comparison.csv")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
