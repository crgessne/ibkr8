# filepath: c:\Users\Administrator\ibkr8\scripts\analyze_stop_width.py
"""
Analyze minimum stop width needed to revert to VWAP without getting stopped out.

For each bar where price is extended from VWAP:
1. Track the MAX adverse excursion (MAE) before either:
   - Price reverts to VWAP (win)
   - End of day (no reversion)
2. Determine what stop width (in ATR) would have been needed

This answers: "How wide does my stop need to be to capture the reversion?"
"""
import pandas as pd
import numpy as np
import sys

print('=== STOP WIDTH ANALYSIS FOR VWAP REVERSION ===\n')
sys.stdout.flush()

# Load data
df = pd.read_csv('C:/Users/Administrator/ibkr8/data/tsla_5min_10years_indicators.csv')
df['time'] = pd.to_datetime(df['time'], utc=True)
df.set_index('time', inplace=True)
df['date'] = df.index.date
print(f'Loaded {len(df):,} bars')
sys.stdout.flush()

# Analyze each potential entry
print('Computing MAE and reversion stats...')
sys.stdout.flush()

results = []
dates = df['date'].unique()

for i, date in enumerate(dates):
    if i % 500 == 0:
        print(f'  Day {i}/{len(dates)}')
        sys.stdout.flush()
    
    day_mask = df['date'] == date
    day_df = df[day_mask]
    day_indices = day_df.index.tolist()
    
    for j, idx in enumerate(day_indices[:-1]):  # Skip last bar
        entry_close = day_df.loc[idx, 'close']
        entry_vwap = day_df.loc[idx, 'vwap']
        entry_atr = day_df.loc[idx, 'atr']
        entry_vwap_width_atr = day_df.loc[idx, 'vwap_width_atr']
        
        if pd.isna(entry_atr) or entry_atr <= 0 or pd.isna(entry_vwap_width_atr):
            continue
        
        # Only analyze bars with meaningful extension
        if entry_vwap_width_atr < 0.5:
            continue
            
        future = day_df.iloc[j+1:]
        
        is_long = entry_close < entry_vwap  # Below VWAP = long trade
        
        # Track bar-by-bar to find MAE before reversion or EOD
        mae_price = entry_close  # Max adverse excursion price
        reverted = False
        reversion_bar = None
        
        for k, (fut_idx, fut_row) in enumerate(future.iterrows()):
            if is_long:
                # Long trade: adverse = price going lower
                if fut_row['low'] < mae_price:
                    mae_price = fut_row['low']
                # Check reversion
                if fut_row['high'] >= entry_vwap:
                    reverted = True
                    reversion_bar = k + 1
                    break
            else:
                # Short trade: adverse = price going higher
                if fut_row['high'] > mae_price:
                    mae_price = fut_row['high']
                # Check reversion
                if fut_row['low'] <= entry_vwap:
                    reverted = True
                    reversion_bar = k + 1
                    break
        
        # Calculate MAE in ATR units
        if is_long:
            mae_atr = (entry_close - mae_price) / entry_atr  # Positive = adverse
        else:
            mae_atr = (mae_price - entry_close) / entry_atr  # Positive = adverse
        
        results.append({
            'date': date,
            'time': idx,
            'entry_close': entry_close,
            'entry_vwap': entry_vwap,
            'entry_atr': entry_atr,
            'vwap_width_atr': entry_vwap_width_atr,
            'is_long': is_long,
            'reverted': reverted,
            'reversion_bar': reversion_bar,
            'mae_price': mae_price,
            'mae_atr': mae_atr,  # How much price went against us (in ATR)
        })

print(f'Computed {len(results):,} trade scenarios')
sys.stdout.flush()

# Convert to DataFrame
rdf = pd.DataFrame(results)

# === ANALYSIS ===
print('\n' + '='*60)
print('ANALYSIS: What stop width is needed to capture VWAP reversion?')
print('='*60)

# Filter to trades that DID revert
reverted = rdf[rdf['reverted'] == True]
not_reverted = rdf[rdf['reverted'] == False]

print(f'\nTotal scenarios: {len(rdf):,}')
print(f'  Reverted to VWAP: {len(reverted):,} ({100*len(reverted)/len(rdf):.1f}%)')
print(f'  Did NOT revert: {len(not_reverted):,} ({100*len(not_reverted)/len(rdf):.1f}%)')

# For trades that reverted, what was the MAE?
print('\n--- MAE FOR TRADES THAT REVERTED TO VWAP ---')
print('(Minimum stop width needed to capture the win)')
print(f'\nOverall MAE stats (in ATR):')
print(f'  Mean MAE: {reverted["mae_atr"].mean():.2f} ATR')
print(f'  Median MAE: {reverted["mae_atr"].median():.2f} ATR')
print(f'  75th percentile: {reverted["mae_atr"].quantile(0.75):.2f} ATR')
print(f'  90th percentile: {reverted["mae_atr"].quantile(0.90):.2f} ATR')
print(f'  95th percentile: {reverted["mae_atr"].quantile(0.95):.2f} ATR')
print(f'  Max MAE: {reverted["mae_atr"].max():.2f} ATR')

# By VWAP zone
print('\n--- MAE BY VWAP EXTENSION ZONE (trades that reverted) ---')
zones = [
    (0.5, 1.0),
    (1.0, 1.5),
    (1.5, 2.0),
    (2.0, 2.5),
    (2.5, 3.0),
    (3.0, 4.0),
    (4.0, 5.0),
]

print(f'\n{"Zone":>12} | {"N":>7} | {"Mean MAE":>9} | {"Median":>8} | {"75th":>8} | {"90th":>8} | {"95th":>8}')
print('-' * 80)

for lo, hi in zones:
    zone_mask = (reverted['vwap_width_atr'] >= lo) & (reverted['vwap_width_atr'] < hi)
    zone_df = reverted[zone_mask]
    
    if len(zone_df) < 10:
        continue
    
    print(f'{lo:.1f}-{hi:.1f} ATR | {len(zone_df):>7,} | {zone_df["mae_atr"].mean():>8.2f} | '
          f'{zone_df["mae_atr"].median():>7.2f} | {zone_df["mae_atr"].quantile(0.75):>7.2f} | '
          f'{zone_df["mae_atr"].quantile(0.90):>7.2f} | {zone_df["mae_atr"].quantile(0.95):>7.2f}')

# What stop width would have captured X% of winners?
print('\n--- STOP WIDTH TO CAPTURE X% OF WINNING TRADES ---')
print('(For trades that did revert to VWAP)')

stop_widths = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

print(f'\n{"Zone":>12} | {"N":>6}', end='')
for sw in stop_widths:
    print(f' | {sw:.1f}ATR', end='')
print()
print('-' * 100)

for lo, hi in zones:
    zone_mask = (reverted['vwap_width_atr'] >= lo) & (reverted['vwap_width_atr'] < hi)
    zone_df = reverted[zone_mask]
    
    if len(zone_df) < 10:
        continue
    
    print(f'{lo:.1f}-{hi:.1f} ATR | {len(zone_df):>6,}', end='')
    
    for sw in stop_widths:
        # What % of trades would NOT have been stopped out with this stop width?
        captured = (zone_df['mae_atr'] <= sw).mean() * 100
        print(f' | {captured:>5.1f}%', end='')
    print()

# Win rate accounting for stop-outs
print('\n--- EFFECTIVE WIN RATE WITH VARIOUS STOP WIDTHS ---')
print('(Considering trades that would have been stopped out)')

print(f'\n{"Zone":>12} | {"Raw WR":>7}', end='')
for sw in stop_widths:
    print(f' | {sw:.1f}ATR', end='')
print()
print('-' * 100)

for lo, hi in zones:
    zone_mask = (rdf['vwap_width_atr'] >= lo) & (rdf['vwap_width_atr'] < hi)
    zone_df = rdf[zone_mask]
    
    if len(zone_df) < 50:
        continue
    
    raw_wr = zone_df['reverted'].mean() * 100
    print(f'{lo:.1f}-{hi:.1f} ATR | {raw_wr:>6.1f}%', end='')
    
    for sw in stop_widths:
        # Win = reverted AND mae <= stop_width
        wins = ((zone_df['reverted'] == True) & (zone_df['mae_atr'] <= sw)).sum()
        # Loss = stopped out OR (didn't revert AND not stopped)
        # Actually: if mae > sw, we're stopped out (loss)
        # If mae <= sw and reverted, we win
        # If mae <= sw and not reverted... EOD exit (could be win or loss depending on exit price)
        # For simplicity: win = reverted and not stopped
        total = len(zone_df)
        eff_wr = 100 * wins / total if total > 0 else 0
        print(f' | {eff_wr:>5.1f}%', end='')
    print()

# Optimal stop width analysis
print('\n--- OPTIMAL STOP WIDTH BY ZONE ---')
print('(Maximizing: win_rate * avg_RR, assuming target = vwap_width)')

for lo, hi in zones:
    zone_mask = (rdf['vwap_width_atr'] >= lo) & (rdf['vwap_width_atr'] < hi)
    zone_df = rdf[zone_mask]
    
    if len(zone_df) < 100:
        continue
    
    best_ev = -999
    best_sw = None
    best_wr = None
    
    avg_target = zone_df['vwap_width_atr'].mean()  # Average profit potential in ATR
    
    for sw in np.arange(0.5, 5.1, 0.25):
        # Win = reverted and not stopped
        wins = ((zone_df['reverted'] == True) & (zone_df['mae_atr'] <= sw)).sum()
        losses = len(zone_df) - wins  # Either stopped or didn't revert
        
        wr = wins / len(zone_df)
        rr = avg_target / sw  # R:R ratio
        
        # Expected value per trade (in R units)
        ev = wr * rr - (1 - wr) * 1  # Win: +RR, Loss: -1R
        
        if ev > best_ev:
            best_ev = ev
            best_sw = sw
            best_wr = wr
    
    if best_sw:
        print(f'{lo:.1f}-{hi:.1f} ATR: Best stop = {best_sw:.2f} ATR '
              f'(WR={best_wr*100:.1f}%, RR={avg_target/best_sw:.2f}, EV={best_ev:.3f}R)')

print('\nDone!')
