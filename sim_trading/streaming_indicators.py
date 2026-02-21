"""
Streaming Indicator Calculator

Calculates indicators on rolling windows of bars,
simulating how indicators would be calculated in real-time paper trading.
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys


class StreamingIndicators:
    """
    Calculate indicators on a rolling window of bars.
    
    This simulates real-time indicator calculation where you only have
    access to historical bars up to the current moment.
    """
    
    def __init__(
        self,
        atr_period: int = 14,
        rsi_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        vwap_slope_period: int = 5,
        rsi_slope_period: int = 3,
        rel_vol_period: int = 20,
        verbose: bool = False,
    ):
        """
        Initialize streaming indicators
        
        Args:
            atr_period: ATR period
            rsi_period: RSI period
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviations
            vwap_slope_period: VWAP slope lookback
            rsi_slope_period: RSI slope lookback
            rel_vol_period: Relative volume period
            verbose: Print debug info
        """
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.vwap_slope_period = vwap_slope_period
        self.rsi_slope_period = rsi_slope_period
        self.rel_vol_period = rel_vol_period
        self.verbose = verbose
        
        # Import indicator functions
        from src.indicators import (
            calc_atr, calc_rsi, calc_bollinger_bands,
            calc_vwap, calc_vwap_distance, calc_price_vs_vwap,
            calc_slope, calc_relative_volume,
            calc_atr_normalized_move, calc_bar_range_atr
        )
        
        self.calc_atr = calc_atr
        self.calc_rsi = calc_rsi
        self.calc_bollinger_bands = calc_bollinger_bands
        self.calc_vwap = calc_vwap
        self.calc_vwap_distance = calc_vwap_distance
        self.calc_price_vs_vwap = calc_price_vs_vwap
        self.calc_slope = calc_slope
        self.calc_relative_volume = calc_relative_volume
        self.calc_atr_normalized_move = calc_atr_normalized_move
        self.calc_bar_range_atr = calc_bar_range_atr
    
    def calculate(self, bars_df: pd.DataFrame) -> Dict:
        """
        Calculate all indicators for the current (last) bar
        
        Args:
            bars_df: DataFrame with historical bars (datetime, open, high, low, close, volume)
            
        Returns:
            Dictionary with indicator values for the current bar
        """
        if len(bars_df) == 0:
            return {}
        
        try:
            # Calculate all indicators on the full window
            result = bars_df.copy()
            
            # Core indicators
            result['atr'] = self.calc_atr(result, self.atr_period)
            result['rsi'] = self.calc_rsi(result, self.rsi_period)
            
            # Bollinger Bands
            bb = self.calc_bollinger_bands(result, self.bb_period, self.bb_std)
            result['bb_upper'] = bb['bb_upper']
            result['bb_middle'] = bb['bb_middle']
            result['bb_lower'] = bb['bb_lower']
            result['bb_pct'] = bb['bb_pct']
            
            # VWAP-related
            result['vwap'] = self.calc_vwap(result)
            result['vwap_dist_pct'] = self.calc_vwap_distance(result, result['vwap'])
            result['price_below_vwap'] = self.calc_price_vs_vwap(result, result['vwap'])
            
            # Slopes
            result['vwap_slope'] = self.calc_slope(result['vwap'], self.vwap_slope_period)
            result['rsi_slope'] = self.calc_slope(result['rsi'], self.rsi_slope_period)
            
            # Volume
            result['rel_vol'] = self.calc_relative_volume(result, self.rel_vol_period)
            
            # ATR-normalized metrics
            result['atr_move'] = self.calc_atr_normalized_move(result, result['atr'])
            result['bar_range_atr'] = self.calc_bar_range_atr(result, result['atr'])
            
            # Distance to Bollinger Bands in ATR units
            result['dist_to_bb_lower'] = (result['close'] - result['bb_lower']) / result['atr']
            result['dist_to_bb_upper'] = (result['bb_upper'] - result['close']) / result['atr']
            
            # Distance to VWAP in ATR units  
            result['vwap_dist_atr'] = (result['close'] - result['vwap']) / result['atr']
            
            # VWAP width (distance from VWAP in ATR units - absolute value)
            result['vwap_width_atr'] = abs(result['vwap_dist_atr'])
            
            # Long setup flag (price below VWAP)
            result['is_long_setup'] = result['price_below_vwap'] == 1.0
            
            # Return indicators for the LAST (current) bar only
            current_indicators = result.iloc[-1].to_dict()
            
            return current_indicators
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Error calculating indicators: {e}", file=sys.stderr, flush=True)
            return {}
    
    def __call__(self, bars_df: pd.DataFrame) -> Dict:
        """Allow using instance as a function"""
        return self.calculate(bars_df)
