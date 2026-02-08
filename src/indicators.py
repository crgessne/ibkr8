"""
Indicator calculations for reversal trading strategy.
All indicators needed to evaluate mean-reversion setups.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class IndicatorConfig:
    """Configuration for indicator calculations."""
    atr_period: int = 14
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    vwap_slope_period: int = 5
    rsi_slope_period: int = 3
    rel_vol_period: int = 20


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR lookback period
        
    Returns:
        Series with ATR values
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        df: DataFrame with 'close' column
        period: RSI lookback period
        
    Returns:
        Series with RSI values (0-100)
    """
    close = df['close']
    delta = close.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calc_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    Args:
        df: DataFrame with 'close' column
        period: Moving average period
        std_dev: Number of standard deviations
        
    Returns:
        DataFrame with 'bb_upper', 'bb_middle', 'bb_lower', 'bb_pct' columns
    """
    close = df['close']
    
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    
    # Percent B: where is price relative to bands (0 = lower, 1 = upper)
    pct_b = (close - lower) / (upper - lower)
    
    return pd.DataFrame({
        'bb_upper': upper,
        'bb_middle': middle,
        'bb_lower': lower,
        'bb_pct': pct_b
    })


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate intraday VWAP (resets each day).
    
    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns and datetime index
        
    Returns:
        Series with VWAP values
    """
    # Typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Get the date for grouping - handle timezone-aware indices
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            # For timezone-aware index, extract dates directly
            if df.index.tz is not None:
                dates = df.index.tz_localize(None).date
            else:
                dates = df.index.date
        elif 'time' in df.columns:
            dates = pd.to_datetime(df['time']).dt.date.values
        else:
            dates = pd.to_datetime(df.index, utc=True).date
    except Exception:
        # Fallback: convert index to string and parse date
        dates = np.array([str(idx)[:10] for idx in df.index])
    
    # Cumulative values per day
    pv = typical_price * df['volume']
    
    result = pd.Series(index=df.index, dtype=float)
    
    for date in np.unique(dates):
        mask = dates == date
        cum_pv = pv[mask].cumsum()
        cum_vol = df.loc[mask, 'volume'].cumsum()
        result[mask] = cum_pv / cum_vol
    
    return result


def calc_vwap_distance(df: pd.DataFrame, vwap: pd.Series) -> pd.Series:
    """
    Calculate price distance from VWAP as a percentage.
    
    Positive = price above VWAP
    Negative = price below VWAP
    """
    return (df['close'] - vwap) / vwap * 100


def calc_slope(series: pd.Series, period: int = 5) -> pd.Series:
    """
    Calculate the slope of a series using linear regression.
    
    Args:
        series: Input series
        period: Number of bars for slope calculation
        
    Returns:
        Series with slope values
    """
    def linreg_slope(x):
        if len(x) < period or x.isna().any():
            return np.nan
        y = np.arange(len(x))
        slope, _ = np.polyfit(y, x, 1)
        return slope
    
    return series.rolling(window=period).apply(linreg_slope, raw=False)


def calc_relative_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate relative volume (current volume / average volume).
    
    Args:
        df: DataFrame with 'volume' column
        period: Lookback period for average
        
    Returns:
        Series with relative volume (1.0 = average, 2.0 = 2x average)
    """
    avg_vol = df['volume'].rolling(window=period).mean()
    return df['volume'] / avg_vol


def calc_price_vs_vwap(df: pd.DataFrame, vwap: pd.Series) -> pd.Series:
    """
    Boolean: is price below VWAP? (good for long reversals)
    """
    return df['close'] < vwap


def calc_atr_normalized_move(df: pd.DataFrame, atr: pd.Series) -> pd.Series:
    """
    Calculate how many ATRs price moved from previous close.
    """
    move = df['close'] - df['close'].shift(1)
    return move / atr


def calc_bar_range_atr(df: pd.DataFrame, atr: pd.Series) -> pd.Series:
    """
    Calculate current bar range as multiple of ATR.
    """
    bar_range = df['high'] - df['low']
    return bar_range / atr


def calc_reversal_context(df: pd.DataFrame, 
                           stop_atr_options: list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]) -> pd.DataFrame:
    """
    Calculate reversal context features for ML.
    
    Focus on features that indicate WHETHER a reversal will succeed,
    not just how far price is from VWAP.
    
    Key question: Given significant extension, what predicts successful mean-reversion?
    
    Args:
        df: DataFrame with indicators (must have 'close', 'vwap', 'atr', 'bb_*', 'rsi')
        stop_atr_options: List of stop widths (in ATR) to calculate R:R for
        
    Returns:
        DataFrame with reversal context columns added
    """
    result = df.copy()
    
    close = df['close']
    high = df['high']
    low = df['low']
    vwap = df['vwap']
    atr = df['atr']
    rsi = df['rsi']
    
    # === Core Distance Metrics (signed) ===
    # Positive = price above VWAP (short opportunity)
    # Negative = price below VWAP (long opportunity)
    result['price_to_vwap'] = close - vwap  # Raw distance
    result['price_to_vwap_atr'] = (close - vwap) / atr  # ATR-normalized
    result['price_to_vwap_pct'] = 100 * (close - vwap) / vwap  # Percentage
    
    # Absolute distance (magnitude of opportunity)
    result['vwap_width_atr'] = abs(result['price_to_vwap_atr'])
    result['vwap_width_pct'] = abs(result['price_to_vwap_pct'])
    
    # === Extension from Bollinger Bands ===
    bb_mid = df['bb_middle'] if 'bb_middle' in df.columns else (df['bb_upper'] + df['bb_lower']) / 2
    bb_width = df['bb_upper'] - df['bb_lower']
    
    # How many "BB widths" price is outside the band
    result['bb_extension_upper'] = np.maximum(0, close - df['bb_upper']) / (bb_width / 2)
    result['bb_extension_lower'] = np.maximum(0, df['bb_lower'] - close) / (bb_width / 2)
    result['bb_extension'] = result['bb_extension_upper'] - result['bb_extension_lower']  # Signed
    result['bb_extension_abs'] = np.maximum(result['bb_extension_upper'], result['bb_extension_lower'])
    
    # Distance to BB in ATR
    result['dist_bb_upper_atr'] = (df['bb_upper'] - close) / atr
    result['dist_bb_lower_atr'] = (close - df['bb_lower']) / atr
    
    # === REVERSAL QUALITY INDICATORS ===
    # These capture whether momentum is exhausting / reversing
    
    # 1. RSI extremity - how oversold/overbought (further from 50 = more extreme)
    result['rsi_extremity'] = abs(rsi - 50)
    
    # 2. RSI divergence from extension - is RSI starting to turn while price extended?
    #    For longs: price making new lows but RSI not (bullish divergence signal)
    #    For shorts: price making new highs but RSI not (bearish divergence signal)
    result['rsi_slope_3'] = rsi - rsi.shift(3)
    result['rsi_slope_5'] = rsi - rsi.shift(5)
    result['price_slope_3'] = (close - close.shift(3)) / atr
    result['price_slope_5'] = (close - close.shift(5)) / atr
    
    # Divergence: price going one way, RSI going the other (or not as much)
    # For longs (price falling): positive divergence if RSI rising or falling less
    # For shorts (price rising): positive divergence if RSI falling or rising less  
    result['momentum_divergence_3'] = -result['price_slope_3'] * np.sign(result['price_to_vwap_atr']) - result['rsi_slope_3'] / 10
    result['momentum_divergence_5'] = -result['price_slope_5'] * np.sign(result['price_to_vwap_atr']) - result['rsi_slope_5'] / 10
    
    # 3. Bar rejection signals - wicks showing buying/selling pressure
    bar_range = high - low
    upper_wick = high - np.maximum(close, df['open'])
    lower_wick = np.minimum(close, df['open']) - low
    body = abs(close - df['open'])
    
    result['upper_wick_pct'] = upper_wick / bar_range.replace(0, np.nan)
    result['lower_wick_pct'] = lower_wick / bar_range.replace(0, np.nan)
    result['body_pct'] = body / bar_range.replace(0, np.nan)
    
    # Rejection wick in direction of reversal
    # For longs (price below VWAP): lower wick = rejection of lower prices = bullish
    # For shorts (price above VWAP): upper wick = rejection of higher prices = bearish
    result['reversal_wick'] = np.where(
        close < vwap,
        result['lower_wick_pct'],  # Long setup - want lower wicks
        result['upper_wick_pct']   # Short setup - want upper wicks
    )
    
    # 4. Momentum exhaustion - is the move slowing down?
    result['bar_range_atr'] = bar_range / atr
    result['range_vs_prev'] = bar_range / bar_range.shift(1).replace(0, np.nan)
    result['range_shrinking'] = (bar_range < bar_range.shift(1)).astype(float)
    
    # Rolling count of shrinking bars (momentum fading)
    result['consecutive_shrinking'] = result['range_shrinking'].rolling(5).sum()
    
    # 5. VWAP slope - is VWAP moving toward or away from price?
    result['vwap_slope_5'] = (vwap - vwap.shift(5)) / atr
    
    # For longs: VWAP rising (positive slope) means target getting closer
    # For shorts: VWAP falling (negative slope) means target getting closer
    result['vwap_helping'] = np.where(
        close < vwap,
        result['vwap_slope_5'],   # Long - positive VWAP slope helps
        -result['vwap_slope_5']   # Short - negative VWAP slope helps
    )
    
    # 6. Extension velocity - how fast did we get here?
    result['extension_velocity_3'] = result['price_to_vwap_atr'] - result['price_to_vwap_atr'].shift(3)
    result['extension_velocity_5'] = result['price_to_vwap_atr'] - result['price_to_vwap_atr'].shift(5)
    
    # Is extension accelerating or decelerating?
    result['extension_accel'] = result['extension_velocity_3'] - result['extension_velocity_3'].shift(3)
    
    # 7. Volume profile at extension
    rel_vol = df['rel_vol'] if 'rel_vol' in df.columns else df['volume'] / df['volume'].rolling(20).mean()
    result['vol_at_extension'] = rel_vol
    
    # Volume declining while extended = momentum exhausting
    result['vol_declining'] = (rel_vol < rel_vol.shift(1)).astype(float)
    result['vol_trend_3'] = rel_vol - rel_vol.shift(3)
    
    # 8. Price action context - where is close relative to bar?
    result['close_position'] = (close - low) / bar_range.replace(0, np.nan)  # 0=low, 1=high
    
    # For longs: want close near high of bar (buyers stepping in)
    # For shorts: want close near low of bar (sellers stepping in)
    result['reversal_close_position'] = np.where(
        close < vwap,
        result['close_position'],      # Long - want close near high
        1 - result['close_position']   # Short - want close near low
    )
    
    # === R:R at Various Stop Widths ===
    for stop_atr in stop_atr_options:
        col_suffix = f"_{stop_atr:.1f}atr".replace(".", "_")
        
        # For LONG trades (price below VWAP, target = VWAP)
        long_profit = np.maximum(0, vwap - close)
        long_stop_dist = stop_atr * atr
        result[f'long_rr{col_suffix}'] = long_profit / long_stop_dist
        
        # For SHORT trades (price above VWAP, target = VWAP)
        short_profit = np.maximum(0, close - vwap)
        short_stop_dist = stop_atr * atr
        result[f'short_rr{col_suffix}'] = short_profit / short_stop_dist
        
        # Combined R:R (for the appropriate direction)
        result[f'rr{col_suffix}'] = np.where(
            close < vwap,
            result[f'long_rr{col_suffix}'],
            result[f'short_rr{col_suffix}']
        )
      # === Summary Metrics ===
    result['reversal_direction'] = np.sign(result['price_to_vwap_atr'])
    result['profit_potential_atr'] = result['vwap_width_atr']
    
    rr_cols = [f'rr_{s:.1f}atr'.replace(".", "_") for s in stop_atr_options]
    result['avg_rr'] = result[rr_cols].mean(axis=1)
    
    # === VWAP Zone Classification ===
    # Based on empirical analysis: win rates drop dramatically beyond 3 ATR
    # Sweet spot is 1.0-1.5 ATR (53% WR), avoid >3 ATR (17% WR) and >4 ATR (6% WR)
    result['in_sweet_spot'] = (result['vwap_width_atr'] >= 1.0) & (result['vwap_width_atr'] < 1.5)
    result['in_tradeable_zone'] = (result['vwap_width_atr'] >= 1.0) & (result['vwap_width_atr'] < 3.0)
    result['over_extended'] = result['vwap_width_atr'] >= 3.0
    result['extreme_extension'] = result['vwap_width_atr'] >= 4.0
    
    result['bars_from_vwap'] = calc_bars_since_condition(df['close'], vwap, crosses_from_above=True)
    result['vwap_dist_delta_3'] = result['price_to_vwap_atr'] - result['price_to_vwap_atr'].shift(3)
    result['vwap_dist_delta_5'] = result['price_to_vwap_atr'] - result['price_to_vwap_atr'].shift(5)
    
    return result


def calc_bars_since_condition(series: pd.Series, threshold: pd.Series, 
                                crosses_from_above: bool = True) -> pd.Series:
    """
    Count bars since price crossed a threshold.
    
    Args:
        series: Price series
        threshold: Threshold series (e.g., VWAP)
        crosses_from_above: If True, count since price crossed from above threshold
        
    Returns:
        Series with bar counts
    """
    # Simplified: just count bars since price and threshold were on same side
    if crosses_from_above:
        crossed = series > threshold
    else:
        crossed = series < threshold
    
    # Group consecutive True/False and count within each group
    groups = (~crossed).cumsum()
    counts = crossed.groupby(groups).cumcount() + 1
    counts = counts.where(crossed, 0)
    
    return counts


def calc_all_indicators(df: pd.DataFrame, config: Optional[IndicatorConfig] = None) -> pd.DataFrame:
    """
    Calculate all indicators needed for reversal trade analysis.
    
    Args:
        df: DataFrame with OHLCV data
        config: Indicator configuration (uses defaults if None)
        
    Returns:
        DataFrame with all original columns plus indicator columns
    """
    if config is None:
        config = IndicatorConfig()
    
    result = df.copy()
    
    # Core indicators
    result['atr'] = calc_atr(df, config.atr_period)
    result['rsi'] = calc_rsi(df, config.rsi_period)
    
    # Bollinger Bands
    bb = calc_bollinger_bands(df, config.bb_period, config.bb_std)
    result['bb_upper'] = bb['bb_upper']
    result['bb_middle'] = bb['bb_middle']
    result['bb_lower'] = bb['bb_lower']
    result['bb_pct'] = bb['bb_pct']
    
    # VWAP-related
    result['vwap'] = calc_vwap(df)
    result['vwap_dist_pct'] = calc_vwap_distance(df, result['vwap'])
    result['price_below_vwap'] = calc_price_vs_vwap(df, result['vwap'])
    
    # Slopes
    result['vwap_slope'] = calc_slope(result['vwap'], config.vwap_slope_period)
    result['rsi_slope'] = calc_slope(result['rsi'], config.rsi_slope_period)
    
    # Volume
    result['rel_vol'] = calc_relative_volume(df, config.rel_vol_period)
    
    # ATR-normalized metrics
    result['atr_move'] = calc_atr_normalized_move(df, result['atr'])
    result['bar_range_atr'] = calc_bar_range_atr(df, result['atr'])
      # Distance to Bollinger Bands in ATR units
    result['dist_to_bb_lower'] = (df['close'] - result['bb_lower']) / result['atr']
    result['dist_to_bb_upper'] = (result['bb_upper'] - df['close']) / result['atr']
    
    # Distance to VWAP in ATR units  
    result['vwap_dist_atr'] = (df['close'] - result['vwap']) / result['atr']
    
    # Add reversal context features for ML
    result = calc_reversal_context(result)
      return result


def identify_reversal_setups(df: pd.DataFrame,
                              max_rsi_long: float = 35.0,
                              min_rsi_short: float = 65.0,
                              min_rel_vol: float = 1.0) -> pd.DataFrame:
    """
    Identify potential reversal trade setups.
    
    Args:
        df: DataFrame with indicators already calculated
        max_rsi_long: RSI must be below this for long setups
        min_rsi_short: RSI must be above this for short setups
        min_rel_vol: Minimum relative volume
        
    Returns:
        DataFrame with 'long_setup' and 'short_setup' boolean columns
    """
    result = df.copy()
    
    # Long setup: oversold conditions
    result['long_setup'] = (
        (df['rsi'] < max_rsi_long) &
        (df['rel_vol'] >= min_rel_vol) &
        (df['close'] < df['bb_lower'])  # Price below lower BB
    )
    
    # Short setup: overbought conditions
    result['short_setup'] = (
        (df['rsi'] > min_rsi_short) &
        (df['rel_vol'] >= min_rel_vol) &
        (df['close'] > df['bb_upper'])  # Price above upper BB
    )
    
    return result


def calc_theo_targets(df: pd.DataFrame, 
                       stop_atr_mult: float = 1.5,
                       fallback_target_atr: float = 1.5) -> pd.DataFrame:
    """
    Calculate theoretical stop and target prices for VWAP mean-reversion.
    
    For LONG: target = VWAP (if above current price), stop = price - 1.5*ATR
    For SHORT: target = VWAP (if below current price), stop = price + 1.5*ATR
    
    Args:
        df: DataFrame with indicators (must have 'close', 'vwap', 'atr')
        stop_atr_mult: ATR multiplier for stop distance (default 1.5 for ~48% win rate)
        fallback_target_atr: ATR multiplier if already past VWAP
        
    Returns:
        DataFrame with target/stop columns added
    """
    result = df.copy()
    
    close = df['close']
    vwap = df['vwap']
    atr = df['atr']
    
    # LONG trades - target VWAP from below
    result['long_stop'] = close - (stop_atr_mult * atr)
    result['long_target'] = np.where(
        vwap > close,
        vwap,  # Mean reversion to VWAP
        close + (fallback_target_atr * atr)  # Fallback if already above VWAP
    )
    result['long_rr'] = (result['long_target'] - close) / (close - result['long_stop'])
    
    # SHORT trades - target VWAP from above
    result['short_stop'] = close + (stop_atr_mult * atr)
    result['short_target'] = np.where(
        vwap < close,
        vwap,  # Mean reversion to VWAP
        close - (fallback_target_atr * atr)  # Fallback if already below VWAP
    )
    result['short_rr'] = (close - result['short_target']) / (result['short_stop'] - close)
    
    return result


if __name__ == "__main__":
    # Test with sample data
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "../data/tsla_5min_2025_01.csv"
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, parse_dates=['time'], index_col='time')
    
    print(f"Loaded {len(df)} bars")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    print("\nCalculating indicators...")
    df_with_ind = calc_all_indicators(df)
    
    print("\nIndicator columns added:")
    new_cols = [c for c in df_with_ind.columns if c not in df.columns]
    print(f"Total: {len(new_cols)} columns")
    for col in new_cols:
        print(f"  - {col}")
    
    print("\nSample output (last 5 bars):")
    print(df_with_ind[['close', 'atr', 'rsi', 'vwap', 'vwap_dist_pct', 'bb_pct', 'rel_vol']].tail())
    
    # Show reversal context features
    print("\n=== REVERSAL CONTEXT FEATURES ===")
    reversal_cols = ['price_to_vwap_atr', 'vwap_width_atr', 'bb_extension', 
                     'rr_1_0atr', 'rr_1_5atr', 'rr_2_0atr', 'avg_rr', 'profit_potential_atr']
    available_cols = [c for c in reversal_cols if c in df_with_ind.columns]
    print(df_with_ind[available_cols].tail(10))
    
    # Show statistics for high-extension bars
    print("\n=== HIGH EXTENSION BARS (>1.5 ATR from VWAP) ===")
    high_ext = df_with_ind[df_with_ind['vwap_width_atr'] > 1.5]
    print(f"Found {len(high_ext)} bars with extension > 1.5 ATR")
    if len(high_ext) > 0:
        print(f"  Avg VWAP width: {high_ext['vwap_width_atr'].mean():.2f} ATR")
        print(f"  Avg R:R @1.5ATR stop: {high_ext['rr_1_5atr'].mean():.2f}")
        print(f"  Avg R:R @2.0ATR stop: {high_ext['rr_2_0atr'].mean():.2f}")
    
    # Identify setups
    df_setups = identify_reversal_setups(df_with_ind)
    long_setups = df_setups['long_setup'].sum()
    short_setups = df_setups['short_setup'].sum()
    print(f"\nReversal setups found: {long_setups} LONG, {short_setups} SHORT")
    
    # Calculate targets
    df_targets = calc_theo_targets(df_setups)
    print("\nSample theoretical targets:")
    print(df_targets[['close', 'vwap', 'long_target', 'long_stop', 'long_rr', 
                       'short_target', 'short_stop', 'short_rr']].tail())
