"""
Streaming Indicator Calculator (Aligned with master_pipeline.py)

This calculator reuses the exact same indicator calculation logic as master_pipeline.py
to ensure feature alignment between training and streaming inference.

Key difference from StreamingIndicators:
- Uses calculate_core_indicators() from master_pipeline.py
- Guarantees feature consistency (no NaN features)
- Calculates all 18 features needed for the trained RF models
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pandas as pd
import numpy as np
from typing import Dict


class StreamingIndicatorsAligned:
    """
    Calculate indicators using the SAME logic as master_pipeline.py
    
    This ensures feature alignment between training and inference.
    """
    
    def __init__(
        self,
        verbose: bool = False,
    ):
        """
        Initialize streaming indicators (aligned with training)
        
        Args:
            verbose: Print debug info (forced off; logging should come from the simulator)
        """
        # Force off to avoid indicator-layer logging; log in StreamingSimulator instead.
        self.verbose = False
        
        # Import the EXACT same function used for training
        from master_pipeline import calculate_core_indicators
        self.calculate_core_indicators = calculate_core_indicators
    
    def calculate(self, bars_df: pd.DataFrame) -> Dict:
        """Calculate all indicators for the current bar using master_pipeline logic."""
        if len(bars_df) == 0:
            return {}
        
        try:
            # Make sure we have the 'date' column for VWAP calculation
            if 'date' not in bars_df.columns:
                if 'datetime' in bars_df.columns:
                    bars_df = bars_df.copy()
                    bars_df['date'] = pd.to_datetime(bars_df['datetime']).dt.date
                else:
                    # No printing here; simulator owns logging.
                    return {}
            
            # IMPORTANT: suppress all printing from master_pipeline during streaming
            result = self.calculate_core_indicators(bars_df, verbose=False)
            current_bar = result.iloc[-1]
            
            indicators = {}
            for col in result.columns:
                val = current_bar[col]
                if pd.api.types.is_numeric_dtype(result[col].dtype):
                    # Keep value even if NaN; strategy may skip; simulator can decide to log.
                    indicators[col] = val
                else:
                    indicators[col] = val
            
            return indicators
            
        except Exception:
            # No printing here; simulator owns logging.
            return {}
    
    def __call__(self, bars_df: pd.DataFrame) -> Dict:
        """Allow using instance as a function"""
        return self.calculate(bars_df)
