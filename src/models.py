"""Data models for bar data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import pandas as pd


@dataclass
class Bar:
    """Single OHLCV bar."""
    
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    wap: float = 0.0
    bar_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "wap": self.wap,
            "bar_count": self.bar_count,
        }


@dataclass
class BarDataResult:
    """Result container for fetched bar data."""
    
    symbol: str
    bar_size: str
    start_date: str
    end_date: str
    what_to_show: str
    bars: List[Bar] = field(default_factory=list)
    fetch_time: Optional[datetime] = None
    error: Optional[str] = None
    
    def __len__(self) -> int:
        return len(self.bars)
    
    def __bool__(self) -> bool:
        return len(self.bars) > 0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert bars to pandas DataFrame."""
        if not self.bars:
            return pd.DataFrame()
        
        data = [bar.to_dict() for bar in self.bars]
        df = pd.DataFrame(data)
        df["symbol"] = self.symbol
        df.set_index("time", inplace=True)
        return df
    
    def to_csv(self, path: str) -> None:
        """Save bars to CSV file."""
        df = self.to_dataframe()
        df.to_csv(path)
    
    def to_parquet(self, path: str) -> None:
        """Save bars to Parquet file."""
        df = self.to_dataframe()
        df.to_parquet(path)


@dataclass
class FetchRequest:
    """Request parameters for fetching bars."""
    
    symbol: str
    start_date: str
    end_date: str
    bar_size: str = "1 day"
    what_to_show: str = "TRADES"
    use_rth: bool = True
    exchange: str = "SMART"
    currency: str = "USD"
    sec_type: str = "STK"
    
    def validate(self) -> None:
        """Validate request parameters."""
        valid_bar_sizes = [
            "1 secs", "5 secs", "10 secs", "15 secs", "30 secs",
            "1 min", "2 mins", "3 mins", "5 mins", "10 mins", 
            "15 mins", "20 mins", "30 mins",
            "1 hour", "2 hours", "3 hours", "4 hours", "8 hours",
            "1 day", "1 week", "1 month"
        ]
        
        if self.bar_size not in valid_bar_sizes:
            raise ValueError(f"Invalid bar_size: {self.bar_size}. Valid: {valid_bar_sizes}")
        
        valid_what_to_show = [
            "TRADES", "MIDPOINT", "BID", "ASK", 
            "BID_ASK", "ADJUSTED_LAST", "HISTORICAL_VOLATILITY",
            "OPTION_IMPLIED_VOLATILITY"
        ]
        
        if self.what_to_show not in valid_what_to_show:
            raise ValueError(f"Invalid what_to_show: {self.what_to_show}")
