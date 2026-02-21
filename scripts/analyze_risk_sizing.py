
import pandas as pd
import numpy as np
from pathlib import Path

DATA_FILE = Path("data/tsla_5min_10years.csv")
RISK_PCT = 1.0
STOP_ATR_MULT = 1.5
CAPITAL = 1_000_000

def main():
    df = pd.read_csv(DATA_FILE)
    df["datetime"] = pd.to_datetime(df["time"], utc=True)
    df["year"] = df["datetime"].dt.year
    df = df[df["year"] == 2024].copy()
    
    # Calc ATR
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    
    # Filter for valid ATR
    df = df.dropna(subset=["atr"])
    df = df[df["atr"] > 0]
    
    # Calculate required shares and notional for 1% risk
    risk_dollars = CAPITAL * (RISK_PCT / 100.0) # 10,000
    
    # Stop distance = 1.5 * ATR
    # Risk per share = Stop Distance
    df["stop_dist"] = df["atr"] * STOP_ATR_MULT
    df["req_shares"] = (risk_dollars / df["stop_dist"]).astype(int)
    df["req_notional"] = df["req_shares"] * df["close"]
    
    total = len(df)
    over_cap = (df["req_notional"] > CAPITAL).sum()
    under_cap = total - over_cap
    
    print(f"Analysis for Year 2024")
    print(f"Total bars: {total}")
    print(f"Bars where 1% Risk requires > $1M Notional: {over_cap} ({over_cap/total*100:.1f}%)")
    print(f"Bars where 1% Risk fits in $1M: {under_cap} ({under_cap/total*100:.1f}%)")
    
    print("\nRequest Statistics:")
    print(df[["close", "atr", "req_shares", "req_notional"]].describe())

if __name__ == "__main__":
    main()
