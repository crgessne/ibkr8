import pandas as pd
import glob
import os

os.chdir(r"C:\Users\Administrator\ibkr8")
files = sorted(glob.glob("data/trades_realized_*072022*.csv"))
print(f"Found {len(files)} files")
for f in files:
    df = pd.read_csv(f)
    stop = f.split("stop")[1].split("_sel")[0]
    if 'shares' in df.columns:
        mn, mx, avg = df['shares'].min(), df['shares'].max(), df['shares'].mean()
        uniq = df['shares'].nunique()
        print(f"stop={stop:>5s}  n={len(df):5d}  shares: min={mn} max={mx} avg={avg:.1f} unique={uniq}")
    else:
        print(f"stop={stop:>5s}  n={len(df):5d}  NO shares column")
