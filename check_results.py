import pandas as pd

df = pd.read_csv('data/streaming_sim_results.csv')
print(f'Trades: {len(df)}')
print(f'Total P&L: ${df["pnl"].sum():,.2f}')
print(f'Unique symbols: {df["symbol"].nunique()}')
print(f'Sample symbols: {sorted(df["symbol"].unique())[:15]}')
print(f'\nWin rate: {(df["pnl"] > 0).mean():.1%}')
print(f'Avg win: ${df[df["pnl"] > 0]["pnl"].mean():.2f}')
print(f'Avg loss: ${df[df["pnl"] < 0]["pnl"].mean():.2f}')
