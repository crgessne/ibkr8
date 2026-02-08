"""
Compare RF results: Original (48 features) vs Cleaned (~35 features)
Shows if removing redundant features changed performance.
"""
import pandas as pd
import numpy as np

# Load both result sets
orig = pd.read_csv('data/rf_simple_grid_results.csv')
clean = pd.read_csv('data/rf_cleaned_features_results.csv')

print("="*90)
print("RF COMPARISON: ORIGINAL (48 features) vs CLEANED (35 features)")
print("="*90)
print(f"\nOriginal features: vwap_width_atr, avg_rr, price_to_vwap_atr + 45 others")
print(f"Cleaned features:  vwap_width_atr (only distance) + 34 others")
print(f"Removed: avg_rr, price_to_vwap_atr, all rr_* columns (redundant with vwap_width_atr)")

# Compare at RF >= 0.5 threshold
print("\n" + "="*90)
print("PERFORMANCE AT RF >= 0.5 THRESHOLD")
print("="*90)

comparison_data = []
for idx in range(len(orig)):
    o = orig.iloc[idx]
    c = clean.iloc[idx]
    
    comparison_data.append({
        'Stop': f"{o['stop_atr']:.2f}",
        'R:R': f"{o['rr']:.1f}:1",
        'Orig_WR': f"{o['rf0.5_wr']*100:.1f}%",
        'Clean_WR': f"{c['rf0.5_wr']*100:.1f}%",
        'Δ_WR': f"{(c['rf0.5_wr'] - o['rf0.5_wr'])*100:+.1f}%",
        'Orig_EV': f"{o['rf0.5_ev']:+.3f}R",
        'Clean_EV': f"{c['rf0.5_ev']:+.3f}R",
        'Δ_EV': f"{c['rf0.5_ev'] - o['rf0.5_ev']:+.3f}R",
        'Orig_N': f"{int(o['rf0.5_n']):,}",
        'Clean_N': f"{int(c['rf0.5_n']):,}",
    })

comp_df = pd.DataFrame(comparison_data)
print("\n" + comp_df.to_string(index=False))

# Statistical summary
print("\n" + "="*90)
print("STATISTICAL SUMMARY OF CHANGES")
print("="*90)

wr_diffs = [(clean.iloc[i]['rf0.5_wr'] - orig.iloc[i]['rf0.5_wr']) * 100 
            for i in range(len(orig))]
ev_diffs = [clean.iloc[i]['rf0.5_ev'] - orig.iloc[i]['rf0.5_ev'] 
            for i in range(len(orig))]
n_diffs = [clean.iloc[i]['rf0.5_n'] - orig.iloc[i]['rf0.5_n'] 
           for i in range(len(orig))]

print(f"\nWin Rate Changes (percentage points):")
print(f"  Mean:   {np.mean(wr_diffs):+.2f}%")
print(f"  Std:    {np.std(wr_diffs):.2f}%")
print(f"  Range:  {np.min(wr_diffs):+.2f}% to {np.max(wr_diffs):+.2f}%")

print(f"\nExpected Value Changes (R):")
print(f"  Mean:   {np.mean(ev_diffs):+.4f}R")
print(f"  Std:    {np.std(ev_diffs):.4f}R")
print(f"  Range:  {np.min(ev_diffs):+.4f}R to {np.max(ev_diffs):+.4f}R")

print(f"\nTrade Count Changes:")
print(f"  Mean:   {np.mean(n_diffs):+.0f} trades")
print(f"  Range:  {np.min(n_diffs):+.0f} to {np.max(n_diffs):+.0f} trades")

# Feature importance comparison
print("\n" + "="*90)
print("TOP 10 FEATURES COMPARISON (0.25 ATR Stop)")
print("="*90)

orig_features_str = orig.iloc[0]['top_features']
clean_features_str = clean.iloc[0]['top_features']

orig_features = [f.strip() for f in orig_features_str.split(',')[:10]]
clean_features = [f.strip() for f in clean_features_str.split(',')[:10]]

print("\nORIGINAL (48 features):")
for i, feat in enumerate(orig_features, 1):
    print(f"  {i:2d}. {feat}")

print("\nCLEANED (35 features):")
for i, feat in enumerate(clean_features, 1):
    print(f"  {i:2d}. {feat}")

# Analyze what changed
print("\n" + "="*90)
print("FEATURE RANKING CHANGES")
print("="*90)

# Features that moved up in ranking
new_top_10 = set(clean_features) - set(orig_features)
if new_top_10:
    print(f"\n✓ NEW features in top 10 (were previously masked by redundancy):")
    for feat in new_top_10:
        print(f"  → {feat}")
else:
    print(f"\n→ No new features entered top 10")

# Features that dropped out
dropped = set(orig_features) - set(clean_features)
if dropped:
    print(f"\n✗ Features that dropped out of top 10:")
    for feat in dropped:
        print(f"  → {feat}")

# Final verdict
print("\n" + "="*90)
print("FINAL VERDICT")
print("="*90)

avg_wr_diff = np.mean(wr_diffs)
avg_ev_diff = np.mean(ev_diffs)
max_abs_wr_diff = max(abs(d) for d in wr_diffs)
max_abs_ev_diff = max(abs(d) for d in ev_diffs)

print(f"\n📊 Performance Impact:")
print(f"   Average WR change: {avg_wr_diff:+.2f}%  (max: {max_abs_wr_diff:.2f}%)")
print(f"   Average EV change: {avg_ev_diff:+.4f}R  (max: {max_abs_ev_diff:.4f}R)")

if max_abs_wr_diff < 1.0 and max_abs_ev_diff < 0.05:
    print(f"\n✅ CONCLUSION: Removing redundant features had MINIMAL impact!")
    print(f"\n   This VALIDATES that the following were redundant:")
    print(f"   • avg_rr (derived from vwap_width_atr / stop)")
    print(f"   • price_to_vwap_atr (signed version of vwap_width_atr)")
    print(f"   • All long_rr_*/short_rr_* columns (redundant distance metrics)")
    
    print(f"\n   ✓ Model performance is virtually IDENTICAL")
    print(f"   ✓ The cleaned model is SIMPLER (35 vs 48 features)")
    print(f"   ✓ Feature importance is now more INTERPRETABLE")
    
    print(f"\n🎯 RECOMMENDATION: Use CLEANED feature set going forward")
    print(f"   → Same predictive power")
    print(f"   → No redundant features")
    print(f"   → Clearer signal of what matters beyond distance")
    
elif max_abs_wr_diff < 2.0 and max_abs_ev_diff < 0.10:
    print(f"\n⚠️  CONCLUSION: Removing features had SMALL impact")
    print(f"   → Changes are minor but noticeable")
    print(f"   → May indicate some information loss")
    print(f"   → Recommend using cleaned set but monitor performance")
    
else:
    print(f"\n❌ CONCLUSION: Removing features had SIGNIFICANT impact")
    print(f"   → Changes are large enough to matter")
    print(f"   → Some features may not have been as redundant as expected")
    print(f"   → Recommend further investigation")

print("\n" + "="*90)
print("NEXT STEPS")
print("="*90)
print("\n1. Analyze feature importance in cleaned model")
print("2. Identify which features matter most BEYOND distance")
print("3. Investigate if momentum/volume indicators now rank higher")
print("4. Create final feature set recommendation")
print("="*90)
