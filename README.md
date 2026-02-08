# ibkr8

A clean Python library for fetching historical bar data from Interactive Brokers (IBKR) using `ib_insync`.

## Project Structure

```
ibkr8/
├── src/                    # Source code
│   ├── __init__.py
│   ├── client.py          # IBKR connection management
│   ├── bar_fetcher.py     # Historical bar data fetching
│   ├── models.py          # Data models
│   └── utils.py           # Utility functions
├── tests/                  # Unit tests
├── config/                 # Configuration files
├── data/                   # Downloaded data (gitignored)
├── scripts/               # CLI scripts
└── requirements.txt
```

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
# Fetch daily bars for TSLA from 2024-01-01 to 2024-12-31
python scripts/fetch_bars.py --symbol TSLA --start 2024-01-01 --end 2024-12-31 --bar-size "1 day"

# Fetch 5-minute bars
python scripts/fetch_bars.py --symbol AAPL --start 2024-06-01 --end 2024-06-30 --bar-size "5 mins"
```

### Python API

```python
from src.client import IBClient
from src.bar_fetcher import BarFetcher

# Connect to IBKR
client = IBClient(host="127.0.0.1", port=7497, client_id=1)
client.connect()

# Fetch bars
fetcher = BarFetcher(client)
bars = fetcher.fetch(
    symbol="TSLA",
    start_date="2024-01-01",
    end_date="2024-12-31",
    bar_size="1 day",
    what_to_show="TRADES"
)

# Convert to DataFrame
df = bars.to_dataframe()
print(df)

client.disconnect()
```

## Configuration

Create `config/settings.json`:

```json
{
  "host": "127.0.0.1",
  "port": 7497,
  "client_id": 1,
  "timeout": 60
}
```

## Supported Bar Sizes

- `1 secs`, `5 secs`, `10 secs`, `15 secs`, `30 secs`
- `1 min`, `2 mins`, `3 mins`, `5 mins`, `10 mins`, `15 mins`, `20 mins`, `30 mins`
- `1 hour`, `2 hours`, `3 hours`, `4 hours`, `8 hours`
- `1 day`, `1 week`, `1 month`

## Requirements

- Python 3.9+
- Interactive Brokers TWS or IB Gateway running
- ib_insync
