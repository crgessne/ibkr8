"""
Compare per-bar features between concurrent backtest and master pipeline
Find exact divergences in feature values, probabilities, and signal timing
"""

import pandas as pd
import numpy as np
from datetime import datetime

def compare_per_bar_features(concurrent_file, master_file, tolerance=1e-6):
    """
    Compare per-bar features from concurrent vs master
    
    Args:
        concurrent_file: Path to concurrent per-bar CSV
        master_file: Path to master per-bar CSV
        tolerance: Numerical tolerance for float comparisons
    """
    
    print(f"\n{'='*80}")
    print("PER-BAR FEATURE COMPARISON")
    print(f"{'='*80}\n")
    
    # Load data
    print(f"Loading concurrent data from {concurrent_file}...")
    df_concurrent = pd.read_csv(concurrent_file)
    df_concurrent['datetime'] = pd.to_datetime(df_concurrent['datetime'])
    
    print(f"Loading master data from {master_file}...")
    df_master = pd.read_csv(master_file)
    df_master['datetime'] = pd.to_datetime(df_master['datetime'])
    
    print(f"\nConcurrent bars: {len(df_concurrent):,}")
    print(f"Master bars:     {len(df_master):,}")
    print(f"Difference:      {abs(len(df_concurrent) - len(df_master)):,}")
    
    # Merge on datetime
    print(f"\nMerging on datetime...")
    merged = pd.merge(
        df_concurrent,
        df_master,
        on='datetime',
        how='outer',
        suffixes=('_concurrent', '_master'),
        indicator=True
    )
    
    # Analyze merge results
    only_concurrent = merged[merged['_merge'] == 'left_only']
    only_master = merged[merged['_merge'] == 'right_only']
    both = merged[merged['_merge'] == 'both']
    
    print(f"\n{'='*80}")
    print("SIGNAL TIMING COMPARISON")
    print(f"{'='*80}")
    print(f"Signals in BOTH:           {len(both):,}")
    print(f"Signals ONLY in concurrent: {len(only_concurrent):,}")
    print(f"Signals ONLY in master:     {len(only_master):,}")
    
    if len(only_concurrent) > 0:
        print(f"\n⚠️  First 10 signals ONLY in concurrent:")
        print(only_concurrent[['datetime', 'prob_concurrent']].head(10))
    
    if len(only_master) > 0:
        print(f"\n⚠️  First 10 signals ONLY in master:")
        print(only_master[['datetime', 'prob_master']].head(10))
    
    # Compare feature values for overlapping bars
    if len(both) > 0:
        print(f"\n{'='*80}")
        print("FEATURE VALUE COMPARISON (overlapping signals)")
        print(f"{'='*80}")
        
        # Get feature columns (exclude datetime, is_setup, prob, _merge)
        feature_cols = [c for c in df_concurrent.columns 
                       if c not in ['datetime', 'is_setup', 'prob']]
        
        divergences = []
        
        for col in feature_cols:
            col_concurrent = f"{col}_concurrent"
            col_master = f"{col}_master"
            
            if col_concurrent in both.columns and col_master in both.columns:
                # Compare values
                diff = np.abs(both[col_concurrent] - both[col_master])
                max_diff = diff.max()
                mean_diff = diff.mean()
                num_divergent = (diff > tolerance).sum()
                
                if num_divergent > 0:
                    divergences.append({
                        'feature': col,
                        'max_diff': max_diff,
                        'mean_diff': mean_diff,
                        'num_divergent': num_divergent,
                        'pct_divergent': (num_divergent / len(both)) * 100
                    })
        
        if len(divergences) > 0:
            print(f"\n⚠️  DIVERGENT FEATURES FOUND: {len(divergences)}")
            div_df = pd.DataFrame(divergences).sort_values('pct_divergent', ascending=False)
            print(div_df.to_string(index=False))
            
            # Find first bar with divergence
            print(f"\n{'='*80}")
            print("FIRST DIVERGENT BAR")
            print(f"{'='*80}")
            
            for _, row in div_df.head(5).iterrows():
                col = row['feature']
                col_concurrent = f"{col}_concurrent"
                col_master = f"{col}_master"
                
                diff = np.abs(both[col_concurrent] - both[col_master])
                first_divergent_idx = (diff > tolerance).idxmax() if (diff > tolerance).any() else None
                
                if first_divergent_idx is not None:
                    bar = both.iloc[first_divergent_idx]
                    print(f"\nFeature: {col}")
                    print(f"  Datetime: {bar['datetime']}")
                    print(f"  Concurrent value: {bar[col_concurrent]}")
                    print(f"  Master value:     {bar[col_master]}")
                    print(f"  Difference:       {bar[col_concurrent] - bar[col_master]}")
        else:
            print(f"\n✓ All feature values match within tolerance ({tolerance})")
        
        # Compare probabilities
        print(f"\n{'='*80}")
        print("RF PROBABILITY COMPARISON")
        print(f"{'='*80}")
        
        prob_diff = np.abs(both['prob_concurrent'] - both['prob_master'])
        prob_max_diff = prob_diff.max()
        prob_mean_diff = prob_diff.mean()
        prob_divergent = (prob_diff > tolerance).sum()
        
        print(f"Max probability difference:  {prob_max_diff:.6f}")
        print(f"Mean probability difference: {prob_mean_diff:.6f}")
        print(f"Bars with prob diff > {tolerance}: {prob_divergent:,} ({prob_divergent/len(both)*100:.2f}%)")
        
        if prob_divergent > 0:
            print(f"\n⚠️  First 10 bars with divergent probabilities:")
            divergent_probs = both[prob_diff > tolerance].head(10)
            print(divergent_probs[['datetime', 'prob_concurrent', 'prob_master']].to_string(index=False))
        
        # Signal threshold comparison
        threshold = 0.5
        concurrent_signals = (both['prob_concurrent'] >= threshold).sum()
        master_signals = (both['prob_master'] >= threshold).sum()
        
        print(f"\n{'='*80}")
        print(f"SIGNAL GENERATION (prob >= {threshold})")
        print(f"{'='*80}")
        print(f"Concurrent signals: {concurrent_signals:,}")
        print(f"Master signals:     {master_signals:,}")
        print(f"Difference:         {abs(concurrent_signals - master_signals):,}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    if len(only_concurrent) == 0 and len(only_master) == 0:
        print("✓ Signal timing matches perfectly (same bars)")
    else:
        print(f"⚠️  Signal timing differs:")
        print(f"   {len(only_concurrent):,} bars only in concurrent")
        print(f"   {len(only_master):,} bars only in master")
    
    if len(both) > 0 and len(divergences) == 0:
        print("✓ All feature values match (within tolerance)")
    elif len(both) > 0:
        print(f"⚠️  {len(divergences)} features have divergent values")
    
    # Save divergence report
    output_file = "data/per_bar_divergence_report.csv"
    
    if len(both) > 0:
        # Export all overlapping bars with differences
        export_cols = ['datetime', 'prob_concurrent', 'prob_master']
        for col in feature_cols[:10]:  # First 10 features
            col_c = f"{col}_concurrent"
            col_m = f"{col}_master"
            if col_c in both.columns and col_m in both.columns:
                export_cols.extend([col_c, col_m])
        
        both[export_cols].to_csv(output_file, index=False)
        print(f"\n✓ Detailed comparison saved to {output_file}")
    
    return merged, divergences


def main():
    """Run the comparison"""
    concurrent_file = "data/concurrent_per_bar_features_concurrent.csv"
    master_file = "data/master_per_bar_features_2024.csv"
    
    try:
        merged, divergences = compare_per_bar_features(concurrent_file, master_file)
        
        print(f"\n{'='*80}")
        print("Analysis complete!")
        print(f"{'='*80}\n")
        
    except FileNotFoundError as e:
        print(f"\n⚠️  ERROR: {e}")
        print(f"\nPlease run export_master_per_bar.py first to generate master per-bar data.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
