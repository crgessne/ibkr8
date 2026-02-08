"""
Quick comparison of RF results between original and cleaned feature sets.
"""
import pandas as pd
import numpy as np

# Load both result sets
print("Loading results...")
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
            'Stop': stop,
            'R:R': f"{orig_row['avg_rr']:.1f}:1",
            'Orig_WR': f"{orig_row['filtered_win_rate']*100:.1f}%",
            'Clean_WR': f"{clean_row['filtered_win_rate']*100:.1f}%",
            'WR_Diff': f"{(clean_row['filtered_win_rate'] - orig_row['filtered_win_rate'])*100:+.1f}%",
            'Orig_EV': f"{orig_row['filtered_expected_value']:+.3f}R",
            'Clean_EV': f"{clean_row['filtered_expected_value']:+.3f}R",
            'EV_Diff': f"{clean_row['filtered_expected_value'] - orig_row['filtered_expected_value']:+.3f}R",
            'Orig_N': int(orig_row['filtered_count']),
            'Clean_N': int(clean_row['filtered_count']),
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

# Summary
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

avg_wr_diff = np.mean(wr_diffs)
avg_ev_diff = np.mean(ev_diffs)
max_abs_wr_diff = max(abs(d) for d in wr_diffs)
max_abs_ev_diff = max(abs(d) for d in ev_diffs)

print(f"\n✓ Performance Impact:")
print(f"  - Average WR change: {avg_wr_diff:+.2f}% (max: {max_abs_wr_diff:.2f}%)")
print(f"  - Average EV change: {avg_ev_diff:+.4f}R (max: {max_abs_ev_diff:.4f}R)")

if abs(avg_wr_diff) < 1.0 and abs(avg_ev_diff) < 0.05:
    print(f"\n✓ CONCLUSION: Removing redundant features had MINIMAL impact!")
    print(f"  → This VALIDATES that avg_rr and price_to_vwap_atr were redundant")
    print(f"  → Model performance is virtually identical")
    print(f"  → The cleaned model is simpler and more interpretable")
    print(f"\n✓ RECOMMENDATION: Use the CLEANED feature set going forward")
    print(f"  → Same predictive power with fewer redundant features")
    print(f"  → Feature importance will now reveal what matters beyond distance")
else:
    print(f"\n⚠ CONCLUSION: Removing features had SIGNIFICANT impact")
    print(f"  → May need to investigate further")

print("\n" + "="*80)
print("Next step: Analyze feature importance from cleaned model")
print("="*80)
