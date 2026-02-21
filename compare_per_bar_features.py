"""
Compare Per-Bar Features: Master Pipeline vs Concurrent Backtest
Identifies exact differences in signal generation
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from datetime import datetime

def compare_per_bar_features():
    print("="*80)
    print("PER-BAR FEATURE COMPARISON")
    print("="*80)
    
    # Load both datasets
    print("\nLoading data...")
    try:
        master = pd.read_csv("data/master_per_bar_features_2024.csv")
        master['datetime'] = pd.to_datetime(master['datetime'])
        print(f"Master: {len(master):,} bars")
    except Exception as e:
        print(f"Error loading master data: {e}")
        return
    
    try:
        concurrent = pd.read_csv("data/concurrent_per_bar_features_concurrent.csv")
        concurrent['datetime'] = pd.to_datetime(concurrent['datetime'])
        print(f"Concurrent: {len(concurrent):,} bars")
    except Exception as e:
        print(f"Error loading concurrent data: {e}")
        return
    
    # Merge on datetime
    print("\nMerging datasets...")
    merged = pd.merge(
        master, 
        concurrent, 
        on='datetime', 
        how='outer',
        suffixes=('_master', '_concurrent'),
        indicator=True
    )
    
    print(f"Merged: {len(merged):,} bars")
    print(f"  Both: {(merged['_merge'] == 'both').sum():,}")
    print(f"  Master only: {(merged['_merge'] == 'left_only').sum():,}")
    print(f"  Concurrent only: {(merged['_merge'] == 'right_only').sum():,}")
    
    # Compare setups
    print("\n" + "="*80)
    print("SETUP COMPARISON")
    print("="*80)
    
    both_data = merged[merged['_merge'] == 'both'].copy()
    
    if len(both_data) > 0:
        # Compare is_setup
        setup_match = both_data['is_setup_master'] == both_data['is_setup_concurrent']
        print(f"\nSetup agreement: {setup_match.sum():,}/{len(both_data):,} ({setup_match.mean()*100:.2f}%)")
        
        if not setup_match.all():
            print(f"\nSetup mismatches: {(~setup_match).sum():,}")
            mismatches = both_data[~setup_match][['datetime', 'is_setup_master', 'is_setup_concurrent']].head(10)
            print(mismatches.to_string(index=False))
        
        # Compare probabilities
        print("\n" + "="*80)
        print("PROBABILITY COMPARISON")
        print("="*80)
        
        # Only compare where both have valid probabilities
        both_prob = both_data[
            both_data['prob_master'].notna() & 
            both_data['prob_concurrent'].notna()
        ].copy()
        
        print(f"\nBars with both probabilities: {len(both_prob):,}")
        
        if len(both_prob) > 0:
            # Calculate differences
            both_prob['prob_diff'] = both_prob['prob_master'] - both_prob['prob_concurrent']
            both_prob['prob_diff_abs'] = both_prob['prob_diff'].abs()
            
            print(f"\nProbability Statistics:")
            print(f"  Mean difference: {both_prob['prob_diff'].mean():.6f}")
            print(f"  Median difference: {both_prob['prob_diff'].median():.6f}")
            print(f"  Max difference: {both_prob['prob_diff_abs'].max():.6f}")
            print(f"  Std dev: {both_prob['prob_diff'].std():.6f}")
            
            # Significant differences (>0.01)
            sig_diff = both_prob['prob_diff_abs'] > 0.01
            if sig_diff.any():
                print(f"\nSignificant differences (>0.01): {sig_diff.sum():,}")
                print("\nTop 10 largest differences:")
                top_diffs = both_prob.nlargest(10, 'prob_diff_abs')[
                    ['datetime', 'prob_master', 'prob_concurrent', 'prob_diff']
                ]
                print(top_diffs.to_string(index=False))
            else:
                print("\n✓ All probabilities match within 0.01!")
            
            # Compare signal generation (prob >= 0.5)
            both_prob['signal_master'] = both_prob['prob_master'] >= 0.5
            both_prob['signal_concurrent'] = both_prob['prob_concurrent'] >= 0.5
            
            signal_match = both_prob['signal_master'] == both_prob['signal_concurrent']
            print(f"\nSignal agreement: {signal_match.sum():,}/{len(both_prob):,} ({signal_match.mean()*100:.2f}%)")
            
            if not signal_match.all():
                print(f"\nSignal mismatches: {(~signal_match).sum():,}")
                signal_mismatches = both_prob[~signal_match][
                    ['datetime', 'prob_master', 'prob_concurrent', 'signal_master', 'signal_concurrent']
                ].head(20)
                print(signal_mismatches.to_string(index=False))
        
        # Compare feature values
        print("\n" + "="*80)
        print("FEATURE VALUE COMPARISON")
        print("="*80)
          # Get common feature columns
        master_features = [c for c in master.columns if c not in ['datetime', 'is_setup', 'has_all_features', 'prob']]
        concurrent_features = [c for c in concurrent.columns if c not in ['datetime', 'is_setup', 'prob']]
        common_features = set(master_features) & set(concurrent_features)
        
        print(f"\nCommon features: {len(common_features)}")
        
        if len(common_features) > 0:
            feature_diffs = []
            
            for feature in sorted(common_features):
                master_col = f"{feature}_master" if f"{feature}_master" in both_data.columns else feature
                concurrent_col = f"{feature}_concurrent" if f"{feature}_concurrent" in both_data.columns else feature
                
                if master_col in both_data.columns and concurrent_col in both_data.columns:
                    # Compare where both are not NaN
                    valid_mask = both_data[master_col].notna() & both_data[concurrent_col].notna()
                    valid_data = both_data[valid_mask]
                    
                    if len(valid_data) > 0:
                        # Skip boolean columns (can't subtract)
                        if valid_data[master_col].dtype == 'bool' or valid_data[concurrent_col].dtype == 'bool':
                            # For booleans, check equality
                            matches = (valid_data[master_col] == valid_data[concurrent_col]).sum()
                            feature_diffs.append({
                                'feature': feature,
                                'max_diff': 0.0 if matches == len(valid_data) else 1.0,
                                'mean_diff': 0.0 if matches == len(valid_data) else 1.0,
                                'valid_bars': len(valid_data)
                            })
                        else:
                            # For numeric columns, calculate difference
                            diff = (valid_data[master_col] - valid_data[concurrent_col]).abs()
                            max_diff = diff.max()
                            mean_diff = diff.mean()
                            
                            feature_diffs.append({
                                'feature': feature,
                                'max_diff': max_diff,
                                'mean_diff': mean_diff,
                                'valid_bars': len(valid_data)
                            })
            
            if feature_diffs:
                diff_df = pd.DataFrame(feature_diffs).sort_values('max_diff', ascending=False)
                print("\nTop 10 features with largest differences:")
                print(diff_df.head(10).to_string(index=False))
                
                # Features with significant differences
                sig_features = diff_df[diff_df['max_diff'] > 0.001]
                if len(sig_features) > 0:
                    print(f"\n⚠️  Features with max diff > 0.001: {len(sig_features)}")
                    print(sig_features.to_string(index=False))
                else:
                    print("\n✓ All features match within 0.001!")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nMaster pipeline signals (prob >= 0.5): {(master['prob'] >= 0.5).sum():,}")
    print(f"Concurrent signals (prob >= 0.5): {(concurrent['prob'] >= 0.5).sum():,}")
    print(f"Difference: {(master['prob'] >= 0.5).sum() - (concurrent['prob'] >= 0.5).sum():,}")
    
    # Export divergence details if any found
    if len(both_data) > 0:
        both_prob = both_data[
            both_data['prob_master'].notna() & 
            both_data['prob_concurrent'].notna()
        ].copy()
        
        if len(both_prob) > 0:
            both_prob['prob_diff'] = (both_prob['prob_master'] - both_prob['prob_concurrent']).abs()
            both_prob['signal_master'] = both_prob['prob_master'] >= 0.5
            both_prob['signal_concurrent'] = both_prob['prob_concurrent'] >= 0.5
            both_prob['signal_mismatch'] = both_prob['signal_master'] != both_prob['signal_concurrent']
            
            divergent = both_prob[
                (both_prob['prob_diff'] > 0.01) | 
                (both_prob['signal_mismatch'])
            ].copy()
            
            if len(divergent) > 0:
                output_file = "data/per_bar_divergences.csv"
                divergent.to_csv(output_file, index=False)
                print(f"\n⚠️  Exported {len(divergent):,} divergent bars to {output_file}")
            else:
                print("\n✓ No significant divergences found!")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    compare_per_bar_features()
