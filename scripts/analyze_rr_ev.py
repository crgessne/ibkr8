"""
Analyze R:R and Expected Value for VWAP reversals.

Key insight: At 2:1 R:R, only need 33.3% WR to break even.
Let's calculate actual R:R for each zone/stop combination.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load data
data_path = Path(__file__).parent.parent / "data" / "tsla_5min_10years_indicators.csv"
print(f"Loading {data_path}...")
df = pd.read_csv(data_path, parse_dates=['time'], index_col='time')
print(f"Loaded {len(df):,} bars")

# Get dates for each bar
df['date'] = df.index.date

# Calculate end-of-day VWAP for each day
eod_vwap = df.groupby('date')['vwap'].last().to_dict()
df['eod_vwap'] = df['date'].map(eod_vwap)

# For each bar, calculate forward prices until EOD
print("\nCalculating forward price paths...")

# We need to track:
# 1. Did price touch VWAP before EOD? (win condition)
# 2. What was the MAE before touching VWAP? (to see if stopped out)
# 3. What's the actual R:R based on zone?

results = []

zones = [
    (0.5, 1.0),
    (1.0, 1.5),
    (1.5, 2.0),
    (2.0, 2.5),
    (2.5, 3.0),
]

stop_widths = [0.5, 1.0, 1.5, 2.0, 2.5]

for zone_low, zone_high in zones:
    zone_name = f"{zone_low}-{zone_high} ATR"
    
    # Filter to bars in this zone
    mask = (df['vwap_width_atr'] >= zone_low) & (df['vwap_width_atr'] < zone_high)
    zone_df = df[mask].copy()
    
    if len(zone_df) == 0:
        continue
    
    # For each bar, we need forward-looking info
    # Direction: long if below VWAP, short if above
    zone_df['is_long'] = zone_df['close'] < zone_df['vwap']
    
    # Profit potential = distance to VWAP in ATR
    zone_df['profit_atr'] = zone_df['vwap_width_atr']
    
    # Midpoint of zone for R:R calculation
    mid_zone = (zone_low + zone_high) / 2
    
    print(f"\n{'='*60}")
    print(f"Zone: {zone_name} (n={len(zone_df):,})")
    print(f"Average profit potential: {zone_df['profit_atr'].mean():.2f} ATR")
    print(f"{'='*60}")
    
    for stop_atr in stop_widths:
        # R:R = profit / risk = zone_mid / stop
        rr = mid_zone / stop_atr
        
        # Breakeven win rate at this R:R
        breakeven_wr = 1 / (1 + rr)
        
        print(f"\n  Stop: {stop_atr} ATR | R:R = {rr:.2f}:1 | Breakeven WR = {breakeven_wr:.1%}")
        
        results.append({
            'zone': zone_name,
            'zone_low': zone_low,
            'zone_high': zone_high,
            'stop_atr': stop_atr,
            'rr': rr,
            'breakeven_wr': breakeven_wr,
            'n_samples': len(zone_df),
        })

# Now let's compute actual win rates accounting for stops
print("\n" + "="*80)
print("COMPUTING ACTUAL WIN RATES WITH STOPS")
print("="*80)

# Pre-compute forward price data for each date
dates = df['date'].unique()
date_to_idx = {date: df[df['date'] == date].index for date in dates}

def compute_trade_outcome(row_idx, df, stop_atr_mult):
    """
    For a given entry bar, compute:
    1. Did price hit VWAP before stop?
    2. What was the MAE?
    """
    row = df.loc[row_idx]
    entry_price = row['close']
    vwap = row['vwap']
    atr = row['atr']
    date = row['date']
    
    is_long = entry_price < vwap
    stop_dist = stop_atr_mult * atr
    
    if is_long:
        stop_price = entry_price - stop_dist
        target_price = vwap
    else:
        stop_price = entry_price + stop_dist
        target_price = vwap
    
    # Get remaining bars for the day
    day_bars = df[df['date'] == date]
    remaining = day_bars.loc[row_idx:]
    
    if len(remaining) <= 1:
        return None, None, None
    
    remaining = remaining.iloc[1:]  # Skip entry bar
    
    mae = 0
    hit_target = False
    hit_stop = False
    
    for idx, bar in remaining.iterrows():
        if is_long:
            # Check stop first (more conservative)
            if bar['low'] <= stop_price:
                hit_stop = True
                break
            # Check target
            if bar['high'] >= target_price:
                hit_target = True
                break
            # Track MAE
            drawdown = entry_price - bar['low']
            mae = max(mae, drawdown)
        else:
            # Short
            if bar['high'] >= stop_price:
                hit_stop = True
                break
            if bar['low'] <= target_price:
                hit_target = True
                break
            drawdown = bar['high'] - entry_price
            mae = max(mae, drawdown)
    
    mae_atr = mae / atr if atr > 0 else 0
    
    return hit_target, hit_stop, mae_atr


# Sample analysis for 1.5-2.0 ATR zone with various stops
print("\nDetailed analysis for 1.5-2.0 ATR zone:")

zone_mask = (df['vwap_width_atr'] >= 1.5) & (df['vwap_width_atr'] < 2.0)
zone_df = df[zone_mask].copy()

# Sample for speed (full analysis takes too long)
np.random.seed(42)
sample_size = min(5000, len(zone_df))
sample_idx = np.random.choice(zone_df.index, size=sample_size, replace=False)

print(f"Sampling {sample_size} trades from {len(zone_df):,} total...")

for stop_atr in [1.0, 1.5, 2.0, 2.5, 3.0]:
    wins = 0
    losses = 0
    stopped_out = 0
    
    for idx in sample_idx:
        hit_target, hit_stop, mae_atr = compute_trade_outcome(idx, df, stop_atr)
        
        if hit_target is None:
            continue
        
        if hit_target:
            wins += 1
        elif hit_stop:
            stopped_out += 1
            losses += 1
        else:
            # EOD exit - neither target nor stop hit
            losses += 1
    
    total = wins + losses
    if total == 0:
        continue
    
    actual_wr = wins / total
    
    # R:R for this zone (using 1.75 ATR as midpoint)
    mid_zone = 1.75
    rr = mid_zone / stop_atr
    breakeven = 1 / (1 + rr)
    
    # Expected value per trade (in R)
    # EV = WR * R - (1-WR) * 1 = WR * R - 1 + WR = WR * (R + 1) - 1
    ev_r = actual_wr * rr - (1 - actual_wr)
    
    print(f"\n  Stop {stop_atr} ATR:")
    print(f"    Wins: {wins}, Stopped: {stopped_out}, EOD exits: {losses - stopped_out}")
    print(f"    Actual WR: {actual_wr:.1%}")
    print(f"    R:R: {rr:.2f}:1")
    print(f"    Breakeven WR: {breakeven:.1%}")
    print(f"    EV per trade: {ev_r:.3f}R  ({'POSITIVE' if ev_r > 0 else 'NEGATIVE'})")

# Repeat for other zones
print("\n" + "="*80)
print("FULL ZONE ANALYSIS")
print("="*80)

summary_results = []

for zone_low, zone_high in zones:
    zone_name = f"{zone_low}-{zone_high} ATR"
    mid_zone = (zone_low + zone_high) / 2
    
    zone_mask = (df['vwap_width_atr'] >= zone_low) & (df['vwap_width_atr'] < zone_high)
    zone_df = df[zone_mask]
    
    if len(zone_df) < 100:
        continue
    
    # Sample
    sample_size = min(3000, len(zone_df))
    sample_idx = np.random.choice(zone_df.index, size=sample_size, replace=False)
    
    print(f"\n{zone_name} (n={len(zone_df):,}, sampled={sample_size}):")
    
    for stop_atr in [1.0, 1.5, 2.0]:
        wins = 0
        losses = 0
        
        for idx in sample_idx:
            hit_target, hit_stop, _ = compute_trade_outcome(idx, df, stop_atr)
            if hit_target is None:
                continue
            if hit_target:
                wins += 1
            else:
                losses += 1
        
        total = wins + losses
        if total == 0:
            continue
        
        actual_wr = wins / total
        rr = mid_zone / stop_atr
        breakeven = 1 / (1 + rr)
        ev_r = actual_wr * rr - (1 - actual_wr)
        
        status = "✓ POSITIVE" if ev_r > 0 else "✗ negative"
        
        print(f"    Stop {stop_atr}: WR={actual_wr:.1%}, R:R={rr:.2f}:1, BE={breakeven:.1%}, EV={ev_r:+.3f}R {status}")
        
        summary_results.append({
            'zone': zone_name,
            'stop_atr': stop_atr,
            'win_rate': actual_wr,
            'rr': rr,
            'breakeven_wr': breakeven,
            'ev_r': ev_r,
            'positive_ev': ev_r > 0,
            'n_trades': total,
        })

# Summary table
print("\n" + "="*80)
print("SUMMARY: Expected Value by Zone and Stop")
print("="*80)

summary_df = pd.DataFrame(summary_results)
print("\nPositive EV combinations:")
positive = summary_df[summary_df['positive_ev'] == True]
if len(positive) > 0:
    print(positive.to_string(index=False))
else:
    print("No positive EV combinations found in raw analysis")

print("\nAll combinations sorted by EV:")
print(summary_df.sort_values('ev_r', ascending=False).head(15).to_string(index=False))
