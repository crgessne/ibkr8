#!/usr/bin/env python
"""CLI script to fetch historical bars from IBKR."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.client import IBClient
from src.bar_fetcher import BarFetcher


def main():
    parser = argparse.ArgumentParser(
        description="Fetch historical bar data from Interactive Brokers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch daily bars for TSLA
  python fetch_bars.py --symbol TSLA --start 2024-01-01 --end 2024-12-31

  # Fetch 5-minute bars
  python fetch_bars.py --symbol AAPL --start 2024-06-01 --end 2024-06-30 --bar-size "5 mins"

  # Fetch multiple symbols
  python fetch_bars.py --symbol TSLA AAPL MSFT --start 2024-01-01 --end 2024-03-31

  # Save to CSV
  python fetch_bars.py --symbol TSLA --start 2024-01-01 --end 2024-12-31 --output data/tsla_daily.csv

Bar sizes:
  1 secs, 5 secs, 10 secs, 15 secs, 30 secs
  1 min, 2 mins, 3 mins, 5 mins, 10 mins, 15 mins, 20 mins, 30 mins
  1 hour, 2 hours, 3 hours, 4 hours, 8 hours
  1 day, 1 week, 1 month
        """
    )
    
    parser.add_argument(
        "--symbol", "-s",
        nargs="+",
        required=True,
        help="Stock symbol(s) to fetch (e.g., TSLA AAPL)"
    )
    parser.add_argument(
        "--start", "-S",
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", "-E",
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--bar-size", "-b",
        default="1 day",
        help="Bar size (default: '1 day')"
    )
    parser.add_argument(
        "--what-to-show", "-w",
        default="TRADES",
        choices=["TRADES", "MIDPOINT", "BID", "ASK", "BID_ASK"],
        help="Data type (default: TRADES)"
    )
    parser.add_argument(
        "--use-rth",
        action="store_true",
        default=True,
        help="Use regular trading hours only (default: True)"
    )
    parser.add_argument(
        "--include-extended",
        action="store_true",
        help="Include extended hours data"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (CSV or Parquet based on extension)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="IB TWS/Gateway host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=7497,
        help="IB TWS/Gateway port (default: 7497)"
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=10,
        help="IB client ID (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Handle RTH flag
    use_rth = not args.include_extended
    
    # Connect to IB
    client = IBClient(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
    )
    
    if not client.connect():
        print("Failed to connect to IB. Make sure TWS/Gateway is running.")
        sys.exit(1)
    
    try:
        fetcher = BarFetcher(client)
        
        if len(args.symbol) == 1:
            # Single symbol
            symbol = args.symbol[0]
            print(f"\n{'='*60}")
            print(f"Fetching {symbol} from {args.start} to {args.end}")
            print(f"Bar size: {args.bar_size}, Data: {args.what_to_show}")
            print(f"{'='*60}\n")
            
            result = fetcher.fetch(
                symbol=symbol,
                start_date=args.start,
                end_date=args.end,
                bar_size=args.bar_size,
                what_to_show=args.what_to_show,
                use_rth=use_rth,
            )
            
            if result.error:
                print(f"\nError: {result.error}")
                sys.exit(1)
            
            df = result.to_dataframe()
            
            print(f"\n{'='*60}")
            print(f"Fetched {len(result)} bars for {symbol}")
            print(f"{'='*60}")
            
            if not df.empty:
                print(f"\nFirst 5 bars:")
                print(df.head())
                print(f"\nLast 5 bars:")
                print(df.tail())
                print(f"\nSummary:")
                print(f"  Date range: {df.index.min()} to {df.index.max()}")
                print(f"  Total bars: {len(df)}")
                print(f"  Volume: {df['volume'].sum():,}")
            
            # Save output
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                if output_path.suffix == ".parquet":
                    result.to_parquet(args.output)
                else:
                    result.to_csv(args.output)
                
                print(f"\nSaved to: {args.output}")
        
        else:
            # Multiple symbols
            print(f"\n{'='*60}")
            print(f"Fetching {len(args.symbol)} symbols from {args.start} to {args.end}")
            print(f"Bar size: {args.bar_size}, Data: {args.what_to_show}")
            print(f"{'='*60}")
            
            results = fetcher.fetch_multiple(
                symbols=args.symbol,
                start_date=args.start,
                end_date=args.end,
                bar_size=args.bar_size,
                what_to_show=args.what_to_show,
                use_rth=use_rth,
            )
            
            print(f"\n{'='*60}")
            print("Results Summary:")
            print(f"{'='*60}")
            
            for symbol, result in results.items():
                if result.error:
                    print(f"  {symbol}: ERROR - {result.error}")
                else:
                    print(f"  {symbol}: {len(result)} bars")
                
                # Save each symbol if output specified
                if args.output and not result.error:
                    output_path = Path(args.output)
                    suffix = output_path.suffix
                    stem = output_path.stem
                    parent = output_path.parent
                    
                    symbol_output = parent / f"{stem}_{symbol}{suffix}"
                    symbol_output.parent.mkdir(parents=True, exist_ok=True)
                    
                    if suffix == ".parquet":
                        result.to_parquet(str(symbol_output))
                    else:
                        result.to_csv(str(symbol_output))
                    
                    print(f"    Saved to: {symbol_output}")
    
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
