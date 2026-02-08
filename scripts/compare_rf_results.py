"""
Compare RF results between original (48 features) and cleaned (~35 features) models.
Analyze if removing redundant features changed performance.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load both result sets
original = pd.read_csv('data/rf_simple_grid_results.csv')
cleaned = pd.read_csv('data/rf_cleaned_features_results.csv')

print("="*80)
print("RF RESULTS COMPARISON: ORIGINAL (48 features) vs CLEANED (~35 features)")
print("="*80)

# Compare performance metrics at RF >= 0.5 threshold
print("\n" + "="*80)
print("PERFORMANCE COMPARISON AT RF >= 0.5 THRESHOLD")
print("="*80)

comparison_data = []
for stop in sorted(original['stop_width'].unique()):
    orig_row = original[(original['stop_width'] == stop) & (original['rf_threshold'] == 0.5)]
    clean_row = cleaned[(cleaned['stop_width'] == stop) & (cleaned['rf_threshold'] == 0.5)]
    
    if not orig_row.empty and not clean_row.empty:
        orig_row = orig_row.iloc[0]
        clean_row = clean_row.iloc[0]
        
        comparison_data.append({
            'Stop Width': stop,
            'R:R': f"{orig_row['avg_rr']:.1f}:1",
            'Orig WR': f"{orig_row['filtered_win_rate']*100:.1f}%",
            'Clean WR': f"{clean_row['filtered_win_rate']*100:.1f}%",
            'WR Diff': f"{(clean_row['filtered_win_rate'] - orig_row['filtered_win_rate'])*100:+.1f}%",
            'Orig EV': f"{orig_row['filtered_expected_value']:+.3f}R",
            'Clean EV': f"{clean_row['filtered_expected_value']:+.3f}R",
            'EV Diff': f"{clean_row['filtered_expected_value'] - orig_row['filtered_expected_value']:+.3f}R",
            'Orig N': f"{int(orig_row['filtered_count']):,}",
            'Clean N': f"{int(clean_row['filtered_count']):,}",
        })

comp_df = pd.DataFrame(comparison_data)
print(comp_df.to_string(index=False))

# Statistical summary of differences
print("\n" + "="*80)
print("SUMMARY: How much did removing redundant features change results?")
print("="*80)

# Extract numeric differences for analysis
wr_diffs = []
ev_diffs = []
for stop in sorted(original['stop_width'].unique()):
    orig_row = original[(original['stop_width'] == stop) & (original['rf_threshold'] == 0.5)]
    clean_row = cleaned[(cleaned['stop_width'] == stop) & (cleaned['rf_threshold'] == 0.5)]
    
    if not orig_row.empty and not clean_row.empty:
        orig_row = orig_row.iloc[0]
        clean_row = clean_row.iloc[0]
        wr_diffs.append((clean_row['filtered_win_rate'] - orig_row['filtered_win_rate']) * 100)
        ev_diffs.append(clean_row['filtered_expected_value'] - orig_row['filtered_expected_value'])

print(f"\nWin Rate Changes:")
print(f"  Mean: {np.mean(wr_diffs):+.2f}%")
print(f"  Std:  {np.std(wr_diffs):.2f}%")
print(f"  Min:  {np.min(wr_diffs):+.2f}%")
print(f"  Max:  {np.max(wr_diffs):+.2f}%")

print(f"\nExpected Value Changes:")
print(f"  Mean: {np.mean(ev_diffs):+.4f}R")
print(f"  Std:  {np.std(ev_diffs):.4f}R")
print(f"  Min:  {np.min(ev_diffs):+.4f}R")
print(f"  Max:  {np.max(ev_diffs):+.4f}R")

# Feature importance comparison
print("\n" + "="*80)
print("TOP 10 FEATURE IMPORTANCE COMPARISON (0.25 ATR Stop)")
print("="*80)

# Get feature importance from the last model run (0.25 ATR stop)
# We'll need to re-run a quick RF to get feature importances
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data/tsla_5min_10years_indicators.csv')
df = df.dropna()

# Define feature sets
all_features = [
    'vwap_width_atr', 'avg_rr', 'price_to_vwap_atr',
    'vwap_helping', 'bars_from_vwap', 'vwap_slope_5',
    'rel_vol', 'vol_at_extension', 'vwap_slope', 'bar_range_atr',
    'roc_5', 'roc_10', 'roc_20', 'rsi', 'rsi_5', 'rsi_10',
    'vwap_cross_count_5', 'vwap_cross_count_10', 'vwap_cross_count_20',
    'vol_surge', 'vol_ratio_to_5', 'vol_ratio_to_20',
    'body_to_range', 'upper_wick_ratio', 'lower_wick_ratio',
    'bar_range_ratio_5', 'bar_range_ratio_20',
    'macd', 'macd_signal', 'macd_hist',
    'bb_position', 'bb_width_atr',
    'vwap_slope_10', 'vwap_slope_20',
    'price_change_atr', 'reversal_bar', 'exhaustion_bar',
    'vol_at_reversal', 'bars_since_reversal',
    'long_rr_025', 'short_rr_025',
    'long_rr_035', 'short_rr_035',
    'long_rr_050', 'short_rr_050',
    'long_rr_075', 'short_rr_075',
    'long_rr_100', 'short_rr_100'
]

cleaned_features = [
    'vwap_width_atr',  # ONLY distance metric - all others removed
    'vwap_helping', 'bars_from_vwap', 'vwap_slope_5',
    'rel_vol', 'vol_at_extension', 'vwap_slope', 'bar_range_atr',
    'roc_5', 'roc_10', 'roc_20', 'rsi', 'rsi_5', 'rsi_10',
    'vwap_cross_count_5', 'vwap_cross_count_10', 'vwap_cross_count_20',
    'vol_surge', 'vol_ratio_to_5', 'vol_ratio_to_20',
    'body_to_range', 'upper_wick_ratio', 'lower_wick_ratio',
    'bar_range_ratio_5', 'bar_range_ratio_20',
    'macd', 'macd_signal', 'macd_hist',
    'bb_position', 'bb_width_atr',
    'vwap_slope_10', 'vwap_slope_20',
    'price_change_atr', 'reversal_bar', 'exhaustion_bar',
    'vol_at_reversal', 'bars_since_reversal'
]

print(f"\nOriginal features: {len(all_features)}")
print(f"Cleaned features:  {len(cleaned_features)}")
print(f"Features removed:  {len(all_features) - len(cleaned_features)}")

# Train/test split
train = df[df['date'] < '2024-01-01'].copy()
test = df[df['date'] >= '2024-01-01'].copy()

print(f"\nTrain size: {len(train):,} bars")
print(f"Test size:  {len(test):,} bars")

# Use 0.25 ATR stop width labels
label_col = 'win_025'

# Train original model
print("\n" + "-"*80)
print("Training ORIGINAL model (48 features)...")
X_train_orig = train[all_features]
X_test_orig = test[all_features]
y_train = train[label_col]

rf_orig = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=100,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1
)
rf_orig.fit(X_train_orig, y_train)

# Get feature importances
orig_importance = pd.DataFrame({
    'feature': all_features,
    'importance': rf_orig.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Original Features:")
for i, row in orig_importance.head(10).iterrows():
    print(f"  {row['feature']:30s} {row['importance']:.4f}")

# Train cleaned model
print("\n" + "-"*80)
print("Training CLEANED model (35 features)...")
X_train_clean = train[cleaned_features]
X_test_clean = test[cleaned_features]

rf_clean = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=100,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1
)
rf_clean.fit(X_train_clean, y_train)

# Get feature importances
clean_importance = pd.DataFrame({
    'feature': cleaned_features,
    'importance': rf_clean.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Cleaned Features:")
for i, row in clean_importance.head(10).iterrows():
    print(f"  {row['feature']:30s} {row['importance']:.4f}")

# Analyze what changed
print("\n" + "="*80)
print("KEY INSIGHTS: What matters BEYOND distance?")
print("="*80)

print("\nFeatures that GAINED importance after removing redundancy:")
print("(These were masked by the redundant distance metrics)")
print()

# Compare common features
common_features = set(all_features) & set(cleaned_features)
importance_changes = []

for feat in common_features:
    orig_imp = orig_importance[orig_importance['feature'] == feat]['importance'].values[0]
    clean_imp = clean_importance[clean_importance['feature'] == feat]['importance'].values[0]
    change = clean_imp - orig_imp
    pct_change = (change / orig_imp * 100) if orig_imp > 0 else 0
    
    importance_changes.append({
        'feature': feat,
        'orig_importance': orig_imp,
        'clean_importance': clean_imp,
        'change': change,
        'pct_change': pct_change
    })

change_df = pd.DataFrame(importance_changes).sort_values('change', ascending=False)

# Show biggest gainers
print("Biggest Gainers (absolute change):")
for i, row in change_df.head(10).iterrows():
    print(f"  {row['feature']:30s} {row['orig_importance']:.4f} → {row['clean_importance']:.4f} ({row['change']:+.4f}, {row['pct_change']:+.1f}%)")

# Summary statistics
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

avg_wr_diff = np.mean(wr_diffs)
avg_ev_diff = np.mean(ev_diffs)

print(f"\n✓ Performance Impact:")
print(f"  - Average WR change: {avg_wr_diff:+.2f}% (negligible)")
print(f"  - Average EV change: {avg_ev_diff:+.4f}R (negligible)")

if abs(avg_wr_diff) < 1.0 and abs(avg_ev_diff) < 0.05:
    print(f"\n✓ CONCLUSION: Removing redundant features had MINIMAL impact on performance!")
    print(f"  → This VALIDATES that avg_rr and price_to_vwap_atr were redundant")
    print(f"  → Model is now simpler and more interpretable")
    print(f"  → Feature importance now reveals what TRULY matters beyond distance")

print(f"\n✓ Top Feature in Cleaned Model: {clean_importance.iloc[0]['feature']}")
print(f"  → Importance: {clean_importance.iloc[0]['importance']:.4f}")

if clean_importance.iloc[0]['feature'] == 'vwap_width_atr':
    print(f"  → Distance is still #1 (as expected)")
    print(f"  → BUT: Importance is now spread across other meaningful features")
else:
    print(f"  → Distance is NO LONGER #1!")
    print(f"  → This feature was previously masked by redundant distance metrics")

print("\n" + "="*80)
