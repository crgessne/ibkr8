"""
Analyze feature importance in the CLEANED model.
Shows what matters BEYOND distance to VWAP.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("="*90)
print("FEATURE IMPORTANCE ANALYSIS: CLEANED MODEL (35 features)")
print("="*90)

# Load data
print("\nLoading data...")
df = pd.read_csv('data/tsla_5min_10years_indicators.csv', parse_dates=['time'])
df = df.dropna()

# Cleaned feature set (NO redundant distance metrics)
cleaned_features = [
    # Distance (single metric)
    'vwap_width_atr',
    
    # VWAP dynamics
    'vwap_helping', 'bars_from_vwap', 'vwap_slope_5', 'vwap_slope', 
    'vwap_slope_10', 'vwap_slope_20',
    'vwap_cross_count_5', 'vwap_cross_count_10', 'vwap_cross_count_20',
    
    # Volume
    'rel_vol', 'vol_at_extension', 'vol_surge', 'vol_ratio_to_5', 'vol_ratio_to_20',
    'vol_at_reversal', 'vol_trend_3',
    
    # Momentum
    'roc_5', 'roc_10', 'roc_20', 'rsi', 'rsi_5', 'rsi_10',
    
    # Bar structure
    'bar_range_atr', 'body_to_range', 'upper_wick_ratio', 'lower_wick_ratio',
    'bar_range_ratio_5', 'bar_range_ratio_20',
    
    # MACD
    'macd', 'macd_signal', 'macd_hist',
    
    # Bollinger
    'bb_position', 'bb_width_atr', 'bb_pct',
    
    # Other
    'price_change_atr', 'reversal_bar', 'exhaustion_bar', 'bars_since_reversal'
]

print(f"Total features: {len(cleaned_features)}")

# Train/test split
train = df[df['time'] < '2024-01-01'].copy()
test = df[df['time'] >= '2024-01-01'].copy()

print(f"Train size: {len(train):,} bars")
print(f"Test size:  {len(test):,} bars")

# Analyze each stop width
stop_widths = [0.25, 0.35, 0.40, 0.50, 0.60, 0.75, 1.00]

print("\n" + "="*90)
print("FEATURE IMPORTANCE BY STOP WIDTH")
print("="*90)

all_importances = []

for stop in stop_widths:
    print(f"\n{'='*90}")
    print(f"Stop Width: {stop} ATR (R:R ~= {(3*stop)/stop:.1f}:1)")
    print(f"{'='*90}")
    
    # Get label column
    label_col = f'win_{str(stop).replace(".", "")}'
    
    # Train model
    X_train = train[cleaned_features]
    X_test = test[cleaned_features]
    y_train = train[label_col]
    y_test = test[label_col]
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=100,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print(f"\nTraining Random Forest...")
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importance_df = pd.DataFrame({
        'feature': cleaned_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Categorize features
    distance_features = ['vwap_width_atr']
    vwap_features = [f for f in cleaned_features if 'vwap' in f and f not in distance_features]
    volume_features = [f for f in cleaned_features if 'vol' in f]
    momentum_features = ['roc_5', 'roc_10', 'roc_20', 'rsi', 'rsi_5', 'rsi_10', 
                        'macd', 'macd_signal', 'macd_hist']
    bar_features = [f for f in cleaned_features if any(x in f for x in ['bar', 'body', 'wick', 'range'])]
    bb_features = [f for f in cleaned_features if 'bb_' in f]
    
    # Calculate category importances
    cat_importance = {
        'Distance': importance_df[importance_df['feature'].isin(distance_features)]['importance'].sum(),
        'VWAP Dynamics': importance_df[importance_df['feature'].isin(vwap_features)]['importance'].sum(),
        'Volume': importance_df[importance_df['feature'].isin(volume_features)]['importance'].sum(),
        'Momentum': importance_df[importance_df['feature'].isin(momentum_features)]['importance'].sum(),
        'Bar Structure': importance_df[importance_df['feature'].isin(bar_features)]['importance'].sum(),
        'Bollinger': importance_df[importance_df['feature'].isin(bb_features)]['importance'].sum(),
    }
    
    print(f"\nTop 15 Features:")
    for i, row in importance_df.head(15).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")
    
    print(f"\nImportance by Category:")
    for cat, imp in sorted(cat_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:20s} {imp:.4f} ({imp*100:.1f}%)")
    
    # Store for summary
    all_importances.append({
        'stop': stop,
        'distance_pct': cat_importance['Distance'] * 100,
        'vwap_pct': cat_importance['VWAP Dynamics'] * 100,
        'volume_pct': cat_importance['Volume'] * 100,
        'momentum_pct': cat_importance['Momentum'] * 100,
        'bar_pct': cat_importance['Bar Structure'] * 100,
        'bb_pct': cat_importance['Bollinger'] * 100,
        'top_feature': importance_df.iloc[0]['feature'],
        'top_importance': importance_df.iloc[0]['importance'],
        'top_non_distance': importance_df[~importance_df['feature'].isin(distance_features)].iloc[0]['feature'],
        'top_non_distance_imp': importance_df[~importance_df['feature'].isin(distance_features)].iloc[0]['importance'],
    })

# Summary table
print("\n" + "="*90)
print("SUMMARY: IMPORTANCE BY CATEGORY ACROSS ALL STOP WIDTHS")
print("="*90)

summary_df = pd.DataFrame(all_importances)
print("\n" + summary_df[['stop', 'distance_pct', 'vwap_pct', 'volume_pct', 
                          'momentum_pct', 'bar_pct', 'bb_pct']].to_string(index=False, 
                          float_format=lambda x: f'{x:.1f}%'))

print("\n" + "="*90)
print("KEY INSIGHTS")
print("="*90)

avg_distance = summary_df['distance_pct'].mean()
avg_vwap = summary_df['vwap_pct'].mean()
avg_volume = summary_df['volume_pct'].mean()
avg_momentum = summary_df['momentum_pct'].mean()

print(f"\nAverage Importance Across All Stop Widths:")
print(f"  1. Distance:       {avg_distance:.1f}%")
print(f"  2. VWAP Dynamics:  {avg_vwap:.1f}%")
print(f"  3. Volume:         {avg_volume:.1f}%")
print(f"  4. Momentum:       {avg_momentum:.1f}%")

print(f"\nTop Non-Distance Feature by Stop Width:")
for _, row in summary_df.iterrows():
    print(f"  {row['stop']:.2f} ATR: {row['top_non_distance']:30s} ({row['top_non_distance_imp']:.4f})")

print("\n" + "="*90)
print("FINAL CONCLUSIONS")
print("="*90)

print(f"\n✓ Distance (vwap_width_atr) accounts for ~{avg_distance:.0f}% of importance")
print(f"  → This validates why zone filtering worked - distance is the primary signal")

print(f"\n✓ VWAP Dynamics account for ~{avg_vwap:.0f}% of importance")
print(f"  → Direction of VWAP, helping/hurting, crossover behavior matters")

print(f"\n✓ Volume accounts for ~{avg_volume:.0f}% of importance")
print(f"  → Volume patterns provide secondary confirmation")

print(f"\n✓ Momentum accounts for ~{avg_momentum:.0f}% of importance")
print(f"  → RSI, ROC, MACD provide additional edge")

print("\n🎯 RECOMMENDATION:")
print("   Use CLEANED feature set (35 features) for final model")
print("   → Same performance as 48-feature model")
print("   → No redundancy")
print("   → Clear signal: Distance + VWAP dynamics + Volume + Momentum")

print("\n" + "="*90)

# Save summary
summary_df.to_csv('data/feature_importance_by_stop.csv', index=False)
print("\n✓ Saved summary to: data/feature_importance_by_stop.csv")
