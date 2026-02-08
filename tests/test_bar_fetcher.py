"""Tests for bar_fetcher module."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Bar, BarDataResult, FetchRequest
from src.utils import parse_date, date_range_chunks


class TestBar:
    def test_bar_creation(self):
        bar = Bar(
            time=datetime(2024, 1, 15, 9, 30),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000,
        )
        
        assert bar.open == 100.0
        assert bar.high == 105.0
        assert bar.low == 99.0
        assert bar.close == 103.0
        assert bar.volume == 1000000
    
    def test_bar_to_dict(self):
        bar = Bar(
            time=datetime(2024, 1, 15),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000,
        )
        
        d = bar.to_dict()
        
        assert "time" in d
        assert "open" in d
        assert d["close"] == 103.0


class TestBarDataResult:
    def test_empty_result(self):
        result = BarDataResult(
            symbol="TSLA",
            bar_size="1 day",
            start_date="2024-01-01",
            end_date="2024-01-31",
            what_to_show="TRADES",
        )
        
        assert len(result) == 0
        assert not result  # bool is False when empty
    
    def test_result_with_bars(self):
        bars = [
            Bar(datetime(2024, 1, 15), 100, 105, 99, 103, 1000),
            Bar(datetime(2024, 1, 16), 103, 108, 102, 106, 1200),
        ]
        
        result = BarDataResult(
            symbol="TSLA",
            bar_size="1 day",
            start_date="2024-01-01",
            end_date="2024-01-31",
            what_to_show="TRADES",
            bars=bars,
        )
        
        assert len(result) == 2
        assert result  # bool is True when has bars
    
    def test_to_dataframe(self):
        bars = [
            Bar(datetime(2024, 1, 15), 100, 105, 99, 103, 1000),
            Bar(datetime(2024, 1, 16), 103, 108, 102, 106, 1200),
        ]
        
        result = BarDataResult(
            symbol="TSLA",
            bar_size="1 day",
            start_date="2024-01-01",
            end_date="2024-01-31",
            what_to_show="TRADES",
            bars=bars,
        )
        
        df = result.to_dataframe()
        
        assert len(df) == 2
        assert "symbol" in df.columns
        assert df["symbol"].iloc[0] == "TSLA"


class TestFetchRequest:
    def test_valid_request(self):
        request = FetchRequest(
            symbol="TSLA",
            start_date="2024-01-01",
            end_date="2024-12-31",
            bar_size="1 day",
        )
        
        # Should not raise
        request.validate()
    
    def test_invalid_bar_size(self):
        request = FetchRequest(
            symbol="TSLA",
            start_date="2024-01-01",
            end_date="2024-12-31",
            bar_size="invalid",
        )
        
        with pytest.raises(ValueError):
            request.validate()
    
    def test_invalid_what_to_show(self):
        request = FetchRequest(
            symbol="TSLA",
            start_date="2024-01-01",
            end_date="2024-12-31",
            what_to_show="INVALID",
        )
        
        with pytest.raises(ValueError):
            request.validate()


class TestUtils:
    def test_parse_date_iso(self):
        dt = parse_date("2024-01-15")
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
    
    def test_parse_date_compact(self):
        dt = parse_date("20240115")
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
    
    def test_date_range_chunks_small(self):
        chunks = list(date_range_chunks("2024-01-01", "2024-01-15", chunk_days=30))
        
        assert len(chunks) == 1
        assert chunks[0][0].day == 1
        assert chunks[0][1].day == 15
    
    def test_date_range_chunks_large(self):
        chunks = list(date_range_chunks("2024-01-01", "2024-12-31", chunk_days=100))
        
        assert len(chunks) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
