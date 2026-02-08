# filepath: c:\Users\Administrator\ibkr8\src\label_generator.py
"""
Label generator for VWAP reversion trades.

Generates forward-looking labels for each stop width.
Labels are computed for ALL bars - the RF/grid search filters by ATR distance.

Label definition:
- WIN (1): Price touches VWAP before hitting stop, within same trading day
- LOSS (0): Price hits stop OR EOD without touching VWAP
- NaN: Can't evaluate (last bar of day, missing data, etc.)
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class LabelConfig:
    """Configuration for label generation."""
    # Stop widths in ATR to generate labels for
    stop_atrs: List[float] = None
    
    def __post_init__(self):
        if self.stop_atrs is None:
            self.stop_atrs = [0.25, 0.35, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0]


def compute_trade_outcome_vectorized(df: pd.DataFrame, stop_atr: float) -> np.ndarray:
    """
    Compute trade outcomes for all bars with a given stop width.
    
    Args:
        df: DataFrame with 'close', 'high', 'low', 'vwap', 'atr', 'date' columns
        stop_atr: Stop distance in ATR units
        
    Returns:
        Array with 1 (win), 0 (loss), or NaN for each bar
    """
    # Pre-compute arrays for speed
    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    vwap_arr = df['vwap'].values
    atr_arr = df['atr'].values
    date_arr = df['date'].values
    n = len(df)
    
    result_arr = np.full(n, np.nan)
    
    # Process by day
    i = 0
    while i < n:
        current_date = date_arr[i]
        
        # Find end of this day
        day_end = i
        while day_end < n and date_arr[day_end] == current_date:
            day_end += 1
        
        # Process all bars for this day
        for j in range(i, day_end):
            entry_price = close_arr[j]
            vwap = vwap_arr[j]
            atr = atr_arr[j]
            
            if np.isnan(atr) or atr <= 0 or np.isnan(vwap):
                continue
            
            is_long = entry_price < vwap
            stop_dist = stop_atr * atr
            
            if is_long:
                stop_price = entry_price - stop_dist
                target_price = vwap
            else:
                stop_price = entry_price + stop_dist
                target_price = vwap
            
            # Look at remaining bars in the day
            if j + 1 >= day_end:
                continue  # Last bar, can't evaluate
            
            hit_target = False
            hit_stop = False
            
            for k in range(j + 1, day_end):
                if is_long:
                    if low_arr[k] <= stop_price:
                        hit_stop = True
                        break
                    if high_arr[k] >= target_price:
                        hit_target = True
                        break
                else:
                    if high_arr[k] >= stop_price:
                        hit_stop = True
                        break
                    if low_arr[k] <= target_price:
                        hit_target = True
                        break
            
            result_arr[j] = 1 if hit_target else 0
        
        i = day_end
    
    return result_arr


def generate_labels(df: pd.DataFrame, config: Optional[LabelConfig] = None) -> pd.DataFrame:
    """
    Generate labels for all stop widths.
    
    Creates columns: label_s{stop} for each stop width.
    
    Args:
        df: DataFrame with indicators (must have close, high, low, vwap, atr)
        config: Label configuration
        
    Returns:
        DataFrame with label columns added
    """
    if config is None:
        config = LabelConfig()
    
    result = df.copy()
    
    # Ensure we have date column
    if 'date' not in result.columns:
        if hasattr(result.index, 'tz') and result.index.tz is not None:
            result['date'] = result.index.tz_localize(None).date
        elif hasattr(result.index, 'date'):
            result['date'] = result.index.date
        else:
            result['date'] = pd.to_datetime(result.index).date
    
    # Clean inf values
    result = result.replace([np.inf, -np.inf], np.nan)
    
    # Generate labels for each stop width
    print(f"Generating labels for {len(config.stop_atrs)} stop widths...")
    for stop_atr in config.stop_atrs:
        stop_col = f"label_s{stop_atr}".replace(".", "_")
        print(f"  Computing labels for stop={stop_atr} ATR...")
        result[stop_col] = compute_trade_outcome_vectorized(result, stop_atr)
    
    return result


def get_stats_for_range(df: pd.DataFrame, stop_atr: float, 
                        atr_min: float, atr_max: float) -> dict:
    """
    Get win rate and EV stats for a specific ATR range and stop width.
    
    Args:
        df: DataFrame with labels and vwap_width_atr
        stop_atr: Stop width used
        atr_min: Minimum vwap_width_atr (inclusive)
        atr_max: Maximum vwap_width_atr (exclusive)
        
    Returns:
        Dict with n, win_rate, rr, breakeven, ev
    """
    stop_col = f"label_s{stop_atr}".replace(".", "_")
    
    # Filter to ATR range
    mask = (df['vwap_width_atr'] >= atr_min) & (df['vwap_width_atr'] < atr_max)
    subset = df.loc[mask, stop_col].dropna()
    
    n = len(subset)
    if n == 0:
        return None
    
    win_rate = subset.mean()
    
    # R:R calculation: reward = mid of ATR range, risk = stop width
    mid_atr = (atr_min + atr_max) / 2
    rr = mid_atr / stop_atr
    breakeven = 1 / (1 + rr)
    ev = win_rate * rr - (1 - win_rate)
    
    return {
        'atr_min': atr_min,
        'atr_max': atr_max,
        'stop_atr': stop_atr,
        'n': n,
        'win_rate': win_rate,
        'rr': rr,
        'breakeven': breakeven,
        'ev': ev,
    }


if __name__ == "__main__":
    # Test with sample data
    from pathlib import Path
    
    data_path = Path(__file__).parent.parent / "data" / "tsla_5min_10years_indicators.csv"
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        exit(1)
    
    print(f"Loading {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['time'], index_col='time')
    print(f"Loaded {len(df):,} bars")
    
    # Generate labels
    print("\nGenerating labels...")
    config = LabelConfig()
    df = generate_labels(df, config)
    
    # Show label columns
    label_cols = [c for c in df.columns if c.startswith('label_')]
    print(f"\nLabel columns: {label_cols}")
    
    # Quick stats for a few ranges
    print("\n" + "="*70)
    print("SAMPLE STATS: Raw win rates by ATR range and stop width")
    print("="*70)
    
    ranges = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]
    stops = [0.35, 0.5, 0.75, 1.0]
    
    print(f"{'Range':<12} {'Stop':<8} {'N':<8} {'WR':<8} {'R:R':<8} {'BE':<8} {'EV':<10}")
    print("-"*70)
    
    for atr_min, atr_max in ranges:
        for stop in stops:
            stats = get_stats_for_range(df, stop, atr_min, atr_max)
            if stats:
                print(f"{atr_min}-{atr_max:<7} {stop:<8.2f} {stats['n']:<8} "
                      f"{stats['win_rate']:<8.1%} {stats['rr']:<8.2f} "
                      f"{stats['breakeven']:<8.1%} {stats['ev']:+.3f}")
