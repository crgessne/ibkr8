"""
Calculate indicators for TSLA bar data and save results.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from indicators import (
    calc_all_indicators, 
    identify_reversal_setups, 
    calc_theo_targets,
    IndicatorConfig
)


def main():
    # Load the bar data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    input_file = os.path.join(data_dir, 'tsla_5min_2025_01.csv')
    output_file = os.path.join(data_dir, 'tsla_5min_2025_01_indicators.csv')
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, parse_dates=['time'], index_col='time')
    
    print(f"Loaded {len(df)} bars")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Configure indicators
    config = IndicatorConfig(
        atr_period=14,
        rsi_period=14,
        bb_period=20,
        bb_std=2.0,
        vwap_slope_period=5,
        rsi_slope_period=3,
        rel_vol_period=20
    )
    
    print("\n" + "="*60)
    print("CALCULATING INDICATORS")
    print("="*60)
    
    # Calculate all indicators
    df_ind = calc_all_indicators(df, config)
    
    # Identify reversal setups
    df_setups = identify_reversal_setups(
        df_ind,
        max_rsi_long=35.0,
        min_rsi_short=65.0,
        min_rel_vol=1.0
    )
    
    # Calculate theoretical targets (VWAP-based mean reversion)
    df_final = calc_theo_targets(df_setups, stop_atr_mult=1.0, fallback_target_atr=1.5)
    
    # Summary statistics
    print("\n" + "="*60)
    print("INDICATOR SUMMARY")
    print("="*60)
    
    # Drop NaN rows for stats
    df_valid = df_final.dropna()
    
    print(f"\nValid bars (after warmup): {len(df_valid)}")
    
    print("\n--- Price & ATR ---")
    print(f"Avg Close:     ${df_valid['close'].mean():.2f}")
    print(f"Avg ATR:       ${df_valid['atr'].mean():.2f} ({df_valid['atr'].mean()/df_valid['close'].mean()*100:.2f}%)")
    
    print("\n--- RSI Distribution ---")
    print(f"Mean RSI:      {df_valid['rsi'].mean():.1f}")
    print(f"RSI < 30:      {(df_valid['rsi'] < 30).sum()} bars ({(df_valid['rsi'] < 30).mean()*100:.1f}%)")
    print(f"RSI > 70:      {(df_valid['rsi'] > 70).sum()} bars ({(df_valid['rsi'] > 70).mean()*100:.1f}%)")
    
    print("\n--- VWAP Distance ---")
    print(f"Avg VWAP Dist: {df_valid['vwap_dist_pct'].mean():.3f}%")
    print(f"Below VWAP:    {df_valid['price_below_vwap'].sum()} bars ({df_valid['price_below_vwap'].mean()*100:.1f}%)")
    print(f"Avg VWAP dist (ATR): {df_valid['vwap_dist_atr'].mean():.2f} ATR")
    
    print("\n--- Bollinger Band Position ---")
    print(f"Avg BB %:      {df_valid['bb_pct'].mean():.2f}")
    print(f"Below lower BB:{(df_valid['bb_pct'] < 0).sum()} bars")
    print(f"Above upper BB:{(df_valid['bb_pct'] > 1).sum()} bars")
    
    print("\n--- Relative Volume ---")
    print(f"Avg Rel Vol:   {df_valid['rel_vol'].mean():.2f}x")
    print(f"High Vol (>2x):{(df_valid['rel_vol'] > 2).sum()} bars")
    
    print("\n" + "="*60)
    print("REVERSAL SETUPS")
    print("="*60)
    
    long_setups = df_final['long_setup'].sum()
    short_setups = df_final['short_setup'].sum()
    print(f"\nLong setups:   {long_setups} ({long_setups/len(df_final)*100:.2f}%)")
    print(f"Short setups:  {short_setups} ({short_setups/len(df_final)*100:.2f}%)")
    
    # Show some example setups
    if long_setups > 0:
        print("\n--- Sample Long Setups ---")
        long_examples = df_final[df_final['long_setup']].head(5)
        for idx, row in long_examples.iterrows():
            print(f"  {idx}: Close=${row['close']:.2f}, RSI={row['rsi']:.1f}, "
                  f"VWAP=${row['vwap']:.2f}, Target=${row['long_target']:.2f}, "
                  f"Stop=${row['long_stop']:.2f}, RR={row['long_rr']:.2f}")
    
    if short_setups > 0:
        print("\n--- Sample Short Setups ---")
        short_examples = df_final[df_final['short_setup']].head(5)
        for idx, row in short_examples.iterrows():
            print(f"  {idx}: Close=${row['close']:.2f}, RSI={row['rsi']:.1f}, "
                  f"VWAP=${row['vwap']:.2f}, Target=${row['short_target']:.2f}, "
                  f"Stop=${row['short_stop']:.2f}, RR={row['short_rr']:.2f}")
    
    print("\n" + "="*60)
    print("THEORETICAL R:R ANALYSIS")
    print("="*60)
    
    # Analyze R:R for setups only
    if long_setups > 0:
        long_rrs = df_final.loc[df_final['long_setup'], 'long_rr']
        print(f"\nLong R:R Stats:")
        print(f"  Mean:   {long_rrs.mean():.2f}")
        print(f"  Median: {long_rrs.median():.2f}")
        print(f"  Min:    {long_rrs.min():.2f}")
        print(f"  Max:    {long_rrs.max():.2f}")
        print(f"  R:R > 1: {(long_rrs > 1).sum()} ({(long_rrs > 1).mean()*100:.1f}%)")
    
    if short_setups > 0:
        short_rrs = df_final.loc[df_final['short_setup'], 'short_rr']
        print(f"\nShort R:R Stats:")
        print(f"  Mean:   {short_rrs.mean():.2f}")
        print(f"  Median: {short_rrs.median():.2f}")
        print(f"  Min:    {short_rrs.min():.2f}")
        print(f"  Max:    {short_rrs.max():.2f}")
        print(f"  R:R > 1: {(short_rrs > 1).sum()} ({(short_rrs > 1).mean()*100:.1f}%)")
    
    # Save results
    print("\n" + "="*60)
    print(f"Saving results to {output_file}...")
    df_final.to_csv(output_file)
    print(f"Saved {len(df_final)} rows with {len(df_final.columns)} columns")
    
    print("\nColumns in output:")
    print(df_final.columns.tolist())


if __name__ == "__main__":
    main()
