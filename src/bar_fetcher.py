"""Historical bar data fetcher."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

from ib_insync import Contract, util

from .client import IBClient
from .models import Bar, BarDataResult, FetchRequest
from .utils import (
    date_range_chunks,
    format_ib_date,
    get_duration_str,
    parse_date,
)


class BarFetcher:
    """
    Fetch historical bar data from IBKR.
    
    Handles:
    - Multiple bar sizes (1 sec to 1 month)
    - Date range chunking for API limits
    - Automatic retries
    - Data validation
    
    Usage:
        client = IBClient()
        client.connect()
        
        fetcher = BarFetcher(client)
        result = fetcher.fetch(
            symbol="TSLA",
            start_date="2024-01-01",
            end_date="2024-12-31",
            bar_size="1 day"
        )
        
        df = result.to_dataframe()
    """
    
    # Max duration per request by bar size (in days)
    MAX_DURATION_DAYS = {
        "1 secs": 1,
        "5 secs": 1,
        "10 secs": 1,
        "15 secs": 1,
        "30 secs": 1,
        "1 min": 1,
        "2 mins": 2,
        "3 mins": 7,
        "5 mins": 7,
        "10 mins": 14,
        "15 mins": 14,
        "20 mins": 30,
        "30 mins": 30,
        "1 hour": 30,
        "2 hours": 60,
        "3 hours": 60,
        "4 hours": 60,
        "8 hours": 120,
        "1 day": 365,
        "1 week": 365 * 2,
        "1 month": 365 * 5,
    }
    
    def __init__(self, client: IBClient):
        self.client = client
    
    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
        exchange: str = "SMART",
        currency: str = "USD",
        sec_type: str = "STK",
    ) -> BarDataResult:
        """
        Fetch historical bars for a symbol.
        
        Args:
            symbol: Ticker symbol (e.g., "TSLA", "AAPL")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            bar_size: Bar size (e.g., "1 day", "5 mins", "1 hour")
            what_to_show: Data type ("TRADES", "MIDPOINT", "BID", "ASK")
            use_rth: Use regular trading hours only
            exchange: Exchange (default "SMART")
            currency: Currency (default "USD")
            sec_type: Security type (default "STK")
            
        Returns:
            BarDataResult with bars and metadata
        """
        request = FetchRequest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            bar_size=bar_size,
            what_to_show=what_to_show,
            use_rth=use_rth,
            exchange=exchange,
            currency=currency,
            sec_type=sec_type,
        )
        
        return self.fetch_request(request)
    
    def fetch_request(self, request: FetchRequest) -> BarDataResult:
        """Fetch bars based on FetchRequest."""
        request.validate()
        
        if not self.client.is_connected:
            return BarDataResult(
                symbol=request.symbol,
                bar_size=request.bar_size,
                start_date=request.start_date,
                end_date=request.end_date,
                what_to_show=request.what_to_show,
                error="Not connected to IB",
            )
        
        # Create and qualify contract
        try:
            contract = self.client.make_stock_contract(
                request.symbol,
                request.exchange,
                request.currency,
            )
            contract = self.client.qualify_contract(contract)
        except Exception as e:
            return BarDataResult(
                symbol=request.symbol,
                bar_size=request.bar_size,
                start_date=request.start_date,
                end_date=request.end_date,
                what_to_show=request.what_to_show,
                error=f"Contract error: {e}",
            )
        
        # Fetch bars in chunks if needed
        all_bars: List[Bar] = []
        max_days = self.MAX_DURATION_DAYS.get(request.bar_size, 30)
        
        for chunk_start, chunk_end in date_range_chunks(
            request.start_date, 
            request.end_date, 
            chunk_days=max_days
        ):
            chunk_bars = self._fetch_chunk(
                contract=contract,
                end_date=chunk_end,
                duration_days=(chunk_end - chunk_start).days + 1,
                bar_size=request.bar_size,
                what_to_show=request.what_to_show,
                use_rth=request.use_rth,
            )
            
            if chunk_bars:
                all_bars.extend(chunk_bars)
        
        # Sort by time and remove duplicates
        all_bars.sort(key=lambda b: b.time)
        seen_times = set()
        unique_bars = []
        for bar in all_bars:
            if bar.time not in seen_times:
                seen_times.add(bar.time)
                unique_bars.append(bar)
        
        return BarDataResult(
            symbol=request.symbol,
            bar_size=request.bar_size,
            start_date=request.start_date,
            end_date=request.end_date,
            what_to_show=request.what_to_show,
            bars=unique_bars,
            fetch_time=datetime.now(),
        )
    
    def _fetch_chunk(
        self,
        contract: Contract,
        end_date: datetime,
        duration_days: int,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
    ) -> List[Bar]:
        """Fetch a single chunk of bars."""
        try:
            # Format end datetime for IB (end of day)
            end_dt = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59)
            end_str = format_ib_date(end_dt)
            
            # Build duration string
            duration_str = f"{duration_days} D"
            
            print(f"  Fetching {contract.symbol}: {duration_str} ending {end_date.date()} ({bar_size})...")
            
            # Request historical data
            ib_bars = self.client.ib.reqHistoricalData(
                contract,
                endDateTime=end_str,
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1,  # Return as datetime
            )
            
            if not ib_bars:
                print(f"    No bars returned")
                return []
            
            print(f"    Got {len(ib_bars)} bars")
            
            # Convert to our Bar model
            bars = []
            for ib_bar in ib_bars:
                bar = Bar(
                    time=ib_bar.date,
                    open=ib_bar.open,
                    high=ib_bar.high,
                    low=ib_bar.low,
                    close=ib_bar.close,
                    volume=int(ib_bar.volume),
                    wap=float(ib_bar.average) if hasattr(ib_bar, 'average') else 0.0,
                    bar_count=int(ib_bar.barCount) if hasattr(ib_bar, 'barCount') else 0,
                )
                bars.append(bar)
            
            return bars
            
        except Exception as e:
            print(f"    Error fetching chunk: {e}")
            return []
    
    def fetch_multiple(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> dict[str, BarDataResult]:
        """
        Fetch bars for multiple symbols.
        
        Returns:
            Dict mapping symbol -> BarDataResult
        """
        results = {}
        
        for symbol in symbols:
            print(f"\nFetching {symbol}...")
            result = self.fetch(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                bar_size=bar_size,
                what_to_show=what_to_show,
                use_rth=use_rth,
            )
            results[symbol] = result
            
            # Small delay between requests
            util.sleep(0.5)
        
        return results
