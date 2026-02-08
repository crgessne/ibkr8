"""Utility functions."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import pytz


def load_config(config_path: str = "config/settings.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    path = Path(config_path)
    if not path.exists():
        return {}
    
    with open(path, "r") as f:
        return json.load(f)


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime."""
    for fmt in ["%Y-%m-%d", "%Y%m%d", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str}")


def format_ib_date(dt: datetime) -> str:
    """Format datetime for IB API."""
    return dt.strftime("%Y%m%d %H:%M:%S")


def date_range_chunks(
    start_date: str,
    end_date: str,
    chunk_days: int = 365
) -> Iterator[Tuple[datetime, datetime]]:
    """
    Split date range into chunks for IB API limits.
    
    IB limits historical data requests by duration based on bar size.
    This splits large ranges into manageable chunks.
    """
    start = parse_date(start_date)
    end = parse_date(end_date)
    
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def get_duration_str(start: datetime, end: datetime, bar_size: str) -> str:
    """
    Calculate duration string for IB API based on date range and bar size.
    
    Returns appropriate duration string like "1 Y", "6 M", "30 D", etc.
    """
    days = (end - start).days + 1
    
    # IB duration limits by bar size
    if "secs" in bar_size:
        # Max 1800 seconds of data for second bars
        return f"{min(days, 1)} D"
    elif "min" in bar_size:
        # Max 1 day for 1-min, more for larger
        if bar_size == "1 min":
            return f"{min(days, 1)} D"
        else:
            return f"{min(days, 30)} D"
    elif "hour" in bar_size:
        return f"{min(days, 365)} D"
    elif bar_size == "1 day":
        if days > 365:
            years = days // 365
            return f"{years} Y"
        return f"{days} D"
    elif bar_size == "1 week":
        return f"{min(days, 365 * 5)} D"
    elif bar_size == "1 month":
        return f"{min(days, 365 * 10)} D"
    
    return f"{days} D"


def market_open_close(date: datetime, tz: str = "US/Eastern") -> Tuple[datetime, datetime]:
    """Get market open and close times for a given date."""
    eastern = pytz.timezone(tz)
    
    market_open = eastern.localize(datetime(date.year, date.month, date.day, 9, 30))
    market_close = eastern.localize(datetime(date.year, date.month, date.day, 16, 0))
    
    return market_open, market_close
