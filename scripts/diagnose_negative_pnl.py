"""
DIAGNOSTIC: Why is every configuration producing negative P&L?

Analyzes recent trade files and the underlying data to find the root causes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np

DATA_DIR = Path("data")

def load_latest_trades(stop_atr):
    """Load the most recent trades file for a given stop level."""
    # Try realized trades first (from the latest run)
    pattern = f"trades_realized_y2024_stop{stop_atr}_seltop_5000_kregressor_*.csv"
    files = sorted(DATA_DIR.glob(pattern))
    if files:
        return pd.read_csv(files[-1]), files[-1].name
    # Fall back to regular trades
    pattern = f"trades_y2024_stop{stop_atr}_seltop_5000_kregressor_*.csv"
    files = sorted(DATA_DIR.glob(pattern))
    if files:
        return pd.read_csv(files[-1]), files[-1].name
    return None, None

def analyze_underlying_data():
    """Analyze the raw data to understand base rates."""
    print("=" * 80)
    print("PART 1: UNDERLYING DATA ANALYSIS")
    print("=" * 80)
    
    df = pd.read_csv(DATA_DIR / "tsla_5min_10years.csv")
    
    # Handle time column (same as master_pipeline)
    if 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['time'], utc=True)
        df['date'] = df['datetime'].dt.date
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.date
    
    # Compute ATR
    high = df['high']; low = df['low']; close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    # Compute VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    pv = typical_price * df['volume']
    df['vwap'] = df.groupby('date').apply(
        lambda g: (pv.loc[g.index].cumsum() / df.loc[g.index, 'volume'].cumsum())
    ).reset_index(level=0, drop=True)
    
    # VWAP distance metrics
    df['vwap_width_atr'] = abs(df['close'] - df['vwap']) / df['atr']
    df['is_long_setup'] = df['close'] < df['vwap']
    
    # Generate labels for key stops
    from label_generator import LabelConfig, generate_labels
    config = LabelConfig(stop_atrs=[0.25, 0.5, 0.75, 1.0, 1.5])
    df = generate_labels(df, config)
    
    # Filter to 2024 test year
    df['year'] = df['datetime'].dt.year
    df_2024 = df[df['year'] == 2024].copy()
    
    print(f"\nTotal bars: {len(df):,}")
    print(f"2024 bars:  {len(df_2024):,}")
    print(f"2024 trading days: {df_2024['date'].nunique()}")
    
    # VWAP distance analysis
    if 'vwap_width_atr' in df_2024.columns:
        vd = df_2024['vwap_width_atr'].dropna()
        print(f"\n--- vwap_width_atr distribution (2024) ---")
        print(f"  Count: {len(vd):,}")
        print(f"  Mean:  {vd.mean():.3f}")
        print(f"  Median:{vd.median():.3f}")
        for pct in [10, 25, 50, 75, 90, 95]:
            print(f"  P{pct:02d}:   {vd.quantile(pct/100):.3f}")
        
        # What fraction of bars have tiny reward?
        for thresh in [0.1, 0.25, 0.5, 1.0]:
            pct_below = (vd < thresh).mean() * 100
            print(f"  vwap_width_atr < {thresh}: {pct_below:.1f}%")
    
    # vwap_width (raw dollars) 
    if 'vwap' in df_2024.columns and 'close' in df_2024.columns:
        vwap_dist_dollars = (df_2024['vwap'] - df_2024['close']).abs()
        print(f"\n--- |close - vwap| in DOLLARS (2024) ---")
        print(f"  Mean:   ${vwap_dist_dollars.mean():.2f}")
        print(f"  Median: ${vwap_dist_dollars.median():.2f}")
        for pct in [10, 25, 50, 75, 90]:
            print(f"  P{pct:02d}:    ${vwap_dist_dollars.quantile(pct/100):.2f}")
        
        # Per 100 shares
        print(f"\n--- Reward per 100 shares (|close - vwap| * 100) ---")
        print(f"  Mean:   ${vwap_dist_dollars.mean() * 100:.0f}")
        print(f"  Median: ${vwap_dist_dollars.median() * 100:.0f}")
        for pct in [10, 25]:
            print(f"  P{pct:02d}:    ${vwap_dist_dollars.quantile(pct/100) * 100:.0f}")
        
        # Cost comparison
        cost_rt = 2 * (0.005 + 0.01) * 100  # round-trip cost per 100 shares
        print(f"  Round-trip cost (100 shares): ${cost_rt:.0f}")
        pct_reward_lt_cost = (vwap_dist_dollars * 100 < cost_rt).mean() * 100
        print(f"  FRACTION WHERE REWARD < COST: {pct_reward_lt_cost:.1f}%")
    
    # ATR analysis
    if 'atr' in df_2024.columns:
        atr = df_2024['atr'].dropna()
        print(f"\n--- ATR distribution (2024) ---")
        print(f"  Mean:   ${atr.mean():.2f}")
        print(f"  Median: ${atr.median():.2f}")
    
    # Base rate: what % of bars are is_long_setup?
    if 'is_long_setup' in df_2024.columns:
        pct_long = df_2024['is_long_setup'].mean() * 100
        print(f"\n--- Direction split ---")
        print(f"  Long setup:  {pct_long:.1f}%")
        print(f"  Short setup: {100 - pct_long:.1f}%")
    
    # Label base rates (oracle win rate WITHOUT any model)
    print(f"\n--- ORACLE BASE RATES (no model, all bars) for 2024 ---")
    for stop in [0.25, 0.5, 0.75, 1.0, 1.5]:
        col = f"label_s{stop}".replace(".", "_")
        if col in df_2024.columns:
            labels = df_2024[col].dropna()
            wr = labels.mean() * 100
            print(f"  Stop {stop}: base win rate = {wr:.1f}% ({len(labels):,} bars)")
    
    # How many bars per day on average?
    bars_per_day = df_2024.groupby('date').size()
    print(f"\n--- Bars per day ---")
    print(f"  Mean:   {bars_per_day.mean():.0f}")
    print(f"  Min:    {bars_per_day.min()}")
    print(f"  Max:    {bars_per_day.max()}")
    
    return df_2024

def analyze_trade_file(stop_atr, df_raw=None):
    """Deep dive into a specific stop level's trades."""
    trades, fname = load_latest_trades(stop_atr)
    if trades is None:
        print(f"\n  [SKIP] No trades file found for stop={stop_atr}")
        return None
    
    print(f"\n{'='*80}")
    print(f"STOP = {stop_atr} ATR  |  File: {fname}")
    print(f"{'='*80}")
    
    n = len(trades)
    print(f"  Total trades: {n:,}")
    
    if n == 0:
        return None
    
    # --- Exit reason breakdown ---
    if 'exit_reason' in trades.columns:
        print(f"\n  --- Exit Reason Breakdown ---")
        for reason in ['stop', 'vwap', 'eod']:
            mask = trades['exit_reason'] == reason
            cnt = mask.sum()
            pct = cnt / n * 100
            avg_pnl = trades.loc[mask, 'net_pnl'].mean() if cnt > 0 else 0
            total_pnl = trades.loc[mask, 'net_pnl'].sum() if cnt > 0 else 0
            print(f"    {reason:5s}: {cnt:5d} ({pct:5.1f}%)  avg_net=${avg_pnl:8.0f}  total_net=${total_pnl:10.0f}")
    
    # --- P&L breakdown ---
    print(f"\n  --- P&L Summary ---")
    print(f"    Total gross: ${trades['gross_pnl'].sum():,.0f}")
    print(f"    Total costs: ${trades['costs'].sum():,.0f}")
    print(f"    Total net:   ${trades['net_pnl'].sum():,.0f}")
    print(f"    Avg net/trade: ${trades['net_pnl'].mean():.2f}")
    
    # --- Win rate ---
    wins = (trades['net_pnl'] > 0).sum()
    losses = (trades['net_pnl'] <= 0).sum()
    wr = wins / n * 100
    print(f"    Win rate (net > 0): {wr:.1f}% ({wins} wins, {losses} losses)")
    
    avg_win = trades.loc[trades['net_pnl'] > 0, 'net_pnl'].mean() if wins > 0 else 0
    avg_loss = trades.loc[trades['net_pnl'] <= 0, 'net_pnl'].mean() if losses > 0 else 0
    print(f"    Avg win:  ${avg_win:.2f}")
    print(f"    Avg loss: ${avg_loss:.2f}")
    if avg_loss != 0:
        print(f"    Win/Loss ratio: {abs(avg_win/avg_loss):.2f}")
    
    # --- Reward/Risk distribution ---
    if 'per_trade_rr' in trades.columns:
        print(f"\n  --- Reward/Risk (entry-bar vwap_width_atr / stop_atr) ---")
        rr = trades['per_trade_rr']
        print(f"    Mean RR:   {rr.mean():.3f}")
        print(f"    Median RR: {rr.median():.3f}")
        for thresh in [0.25, 0.5, 1.0, 2.0, 3.0]:
            pct = (rr < thresh).mean() * 100
            print(f"    RR < {thresh}: {pct:.1f}%")
    
    # --- vwap_dist_atr distribution of entries ---
    if 'vwap_dist_atr' in trades.columns:
        vd = trades['vwap_dist_atr']
        print(f"\n  --- vwap_dist_atr at entry ---")
        print(f"    Mean:   {vd.mean():.3f}")
        print(f"    Median: {vd.median():.3f}")
        for pct in [10, 25, 50, 75, 90]:
            print(f"    P{pct:02d}:    {vd.quantile(pct/100):.3f}")
    
    # --- $ reward at entry (distance to VWAP in dollars per 100 shares) ---
    if 'entry_price' in trades.columns and df_raw is not None:
        # Reconstruct $ reward
        pass  # We'd need the VWAP at entry which isn't in trades; use vwap_dist_atr * atr * shares
    
    # --- Win rate BY R:R bucket ---
    if 'per_trade_rr' in trades.columns and 'exit_reason' in trades.columns:
        print(f"\n  --- Win Rate by R:R Bucket ---")
        buckets = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 100)]
        for lo, hi in buckets:
            mask = (trades['per_trade_rr'] >= lo) & (trades['per_trade_rr'] < hi)
            cnt = mask.sum()
            if cnt > 0:
                wins_b = (trades.loc[mask, 'net_pnl'] > 0).sum()
                wr_b = wins_b / cnt * 100
                avg_pnl_b = trades.loc[mask, 'net_pnl'].mean()
                total_pnl_b = trades.loc[mask, 'net_pnl'].sum()
                vwap_pct = (trades.loc[mask, 'exit_reason'] == 'vwap').mean() * 100
                print(f"    RR [{lo:.1f}, {hi:.1f}): n={cnt:5d}, WR={wr_b:5.1f}%, "
                      f"avg_net=${avg_pnl_b:7.0f}, total=${total_pnl_b:10.0f}, vwap_exit={vwap_pct:.0f}%")
    
    # --- Direction analysis ---
    if 'is_long' in trades.columns:
        print(f"\n  --- Direction Analysis ---")
        for d, label in [(1, 'LONG'), (0, 'SHORT')]:
            mask = trades['is_long'] == d
            cnt = mask.sum()
            if cnt > 0:
                pnl = trades.loc[mask, 'net_pnl'].sum()
                wr_d = (trades.loc[mask, 'net_pnl'] > 0).mean() * 100
                print(f"    {label}: n={cnt:5d}, total_net=${pnl:10.0f}, WR={wr_d:.1f}%")
    
    # --- Trades per day distribution ---
    if 'entry_datetime' in trades.columns:
        trades['entry_dt'] = pd.to_datetime(trades['entry_datetime'])
        trades['entry_date'] = trades['entry_dt'].dt.date
        tpd = trades.groupby('entry_date').size()
        print(f"\n  --- Trades Per Day ---")
        print(f"    Mean:   {tpd.mean():.1f}")
        print(f"    Median: {tpd.median():.0f}")
        print(f"    Max:    {tpd.max()}")
        print(f"    Days with >5 trades: {(tpd > 5).sum()}")
        print(f"    Days with >10 trades: {(tpd > 10).sum()}")
        
        # Daily P&L
        daily_pnl = trades.groupby('entry_date')['net_pnl'].sum()
        print(f"\n  --- Daily P&L ---")
        print(f"    Profitable days: {(daily_pnl > 0).sum()} / {len(daily_pnl)}")
        print(f"    Best day:  ${daily_pnl.max():,.0f}")
        print(f"    Worst day: ${daily_pnl.min():,.0f}")
        print(f"    Mean daily: ${daily_pnl.mean():,.0f}")
    
    # --- THE SMOKING GUN: What's the $ reward for VWAP-touch wins? ---
    if 'exit_reason' in trades.columns:
        vwap_wins = trades[trades['exit_reason'] == 'vwap']
        stops = trades[trades['exit_reason'] == 'stop']
        eods = trades[trades['exit_reason'] == 'eod']
        
        print(f"\n  --- SMOKING GUN: VWAP-touch wins vs Stops ---")
        if len(vwap_wins) > 0:
            print(f"    VWAP wins ({len(vwap_wins):,}):")
            print(f"      Avg gross:  ${vwap_wins['gross_pnl'].mean():.2f}")
            print(f"      Avg net:    ${vwap_wins['net_pnl'].mean():.2f}")
            print(f"      Avg risk $: ${vwap_wins['risk_dollars'].mean():.2f}")
            # How many VWAP wins have net < 0 (reward < costs)?
            vwap_but_net_neg = (vwap_wins['net_pnl'] <= 0).sum()
            pct = vwap_but_net_neg / len(vwap_wins) * 100
            print(f"      VWAP touch but net_pnl <= 0: {vwap_but_net_neg} ({pct:.1f}%) ← reward < costs!")
        
        if len(stops) > 0:
            print(f"    Stops ({len(stops):,}):")
            print(f"      Avg gross:  ${stops['gross_pnl'].mean():.2f}")
            print(f"      Avg net:    ${stops['net_pnl'].mean():.2f}")
            print(f"      Avg risk $: ${stops['risk_dollars'].mean():.2f}")
        
        if len(eods) > 0:
            print(f"    EOD exits ({len(eods):,}):")
            print(f"      Avg gross:  ${eods['gross_pnl'].mean():.2f}")
            print(f"      Avg net:    ${eods['net_pnl'].mean():.2f}")
            # EOD: what fraction end up as losses?
            eod_loss_pct = (eods['net_pnl'] < 0).mean() * 100
            print(f"      EOD loss rate: {eod_loss_pct:.1f}%")
    
    return trades


def compute_theoretical_edge():
    """
    What WOULD the strategy need to be profitable?
    
    For a stop of S (ATR), cost C ($/trade), and 100 shares:
    - Loss per stop = S * ATR * 100 + C
    - Win per VWAP touch = dist_to_vwap * 100 - C
    - Breakeven win rate = Loss / (Win + Loss)
    
    Let's compute for typical TSLA 2024 parameters.
    """
    print(f"\n{'='*80}")
    print("THEORETICAL EDGE ANALYSIS")
    print(f"{'='*80}")
    
    # Typical 2024 values
    atr = 4.0  # typical ATR
    cost_rt = 2 * (0.005 + 0.01) * 100  # $3 per round trip
    
    print(f"  Assumptions: ATR=${atr:.1f}, cost/RT=${cost_rt:.0f}, 100 shares")
    
    for stop_atr in [0.25, 0.5, 0.75, 1.0, 1.5]:
        stop_dollars = stop_atr * atr * 100  # risk per trade
        
        # For different VWAP distances
        print(f"\n  Stop = {stop_atr} ATR (risk/trade = ${stop_dollars:.0f}):")
        for vwap_atr in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
            reward = vwap_atr * atr * 100  # gross reward
            reward_net = reward - cost_rt
            loss_total = stop_dollars + cost_rt
            
            if reward_net <= 0:
                print(f"    vwap_dist={vwap_atr:.2f} ATR: IMPOSSIBLE (reward ${reward:.0f} < cost ${cost_rt:.0f})")
                continue
            
            breakeven_wr = loss_total / (reward_net + loss_total) * 100
            rr = reward_net / loss_total
            print(f"    vwap_dist={vwap_atr:.2f} ATR: reward_net=${reward_net:.0f}, BE_WR={breakeven_wr:.1f}%, net_RR={rr:.2f}")


def check_bar_level_overtrading():
    """
    THE BIG QUESTION: Are we entering EVERY qualifying bar, or once per 'setup'?
    
    In a real strategy you'd enter ONCE when price diverges from VWAP, not at 
    every single 5-min bar while it stays away from VWAP.
    """
    print(f"\n{'='*80}")
    print("BAR-LEVEL OVERTRADING ANALYSIS")
    print(f"{'='*80}")
    
    # Load a trade file
    trades, fname = load_latest_trades(0.5)
    if trades is None:
        print("  No trades found for stop=0.5")
        return
    
    trades['entry_dt'] = pd.to_datetime(trades['entry_datetime'])
    trades['exit_dt'] = pd.to_datetime(trades['exit_datetime'])
    trades['entry_date'] = trades['entry_dt'].dt.date
    
    # Even with flat-to-flat, how many trades per day are we taking?
    tpd = trades.groupby('entry_date').size()
    print(f"\n  File: {fname}")
    print(f"  Total trades: {len(trades):,}")
    print(f"  Trading days: {len(tpd)}")
    print(f"  Trades/day: mean={tpd.mean():.1f}, median={tpd.median():.0f}, max={tpd.max()}")
    
    # Duration of trades
    trades['duration_bars'] = trades['exit_bar_index'] - trades['entry_bar_index']
    print(f"\n  Trade duration (in bars):")
    print(f"    Mean:   {trades['duration_bars'].mean():.1f}")
    print(f"    Median: {trades['duration_bars'].median():.0f}")
    print(f"    Min:    {trades['duration_bars'].min()}")
    print(f"    Max:    {trades['duration_bars'].max()}")
    
    # How many 1-bar trades?
    one_bar = (trades['duration_bars'] == 1).sum()
    two_bar = (trades['duration_bars'] <= 2).sum()
    print(f"    1-bar trades: {one_bar} ({one_bar/len(trades)*100:.1f}%)")
    print(f"    <=2-bar trades: {two_bar} ({two_bar/len(trades)*100:.1f}%)")
    
    # Show a sample day
    busiest_date = tpd.idxmax()
    day_trades = trades[trades['entry_date'] == busiest_date].sort_values('entry_dt')
    print(f"\n  --- Busiest day: {busiest_date} ({len(day_trades)} trades) ---")
    for _, t in day_trades.iterrows():
        print(f"    Entry: {t['entry_dt'].strftime('%H:%M')} -> "
              f"Exit: {t['exit_dt'].strftime('%H:%M')} | "
              f"{'L' if t['is_long'] else 'S'} | "
              f"exit={t['exit_reason']:4s} | "
              f"net=${t['net_pnl']:8.0f} | "
              f"RR={t.get('per_trade_rr', 0):.2f}")


def main():
    print("=" * 80)
    print("ROOT CAUSE DIAGNOSIS: WHY IS EVERY CONFIG NEGATIVE P&L?")
    print("=" * 80)
    
    # 1. Analyze raw data
    df_2024 = analyze_underlying_data()
    
    # 2. Theoretical edge analysis
    compute_theoretical_edge()
    
    # 3. Analyze trades for key stop levels
    for stop in [0.25, 0.5, 1.0, 1.5]:
        analyze_trade_file(stop, df_2024)
    
    # 4. Check overtrading
    check_bar_level_overtrading()
    
    # 5. Final verdict
    print(f"\n{'='*80}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*80}")
    print("""
    Key questions to answer from the above data:
    
    1. REWARD < COST: What fraction of entries have reward (|close - vwap|) < $3 cost?
       If this is >50%, the strategy is structurally unprofitable for those trades.
    
    2. BAR-LEVEL OVERTRADING: Are we entering 5-10+ trades per day? 
       Each trade pays costs. Real discretionary traders enter once per setup.
    
    3. EOD EXITS: How many trades neither hit VWAP nor stop, and exit at EOD at a loss?
       These are "slow bleed" losses that EV-in-R analysis misses.
    
    4. RR DISTRIBUTION: The per_trade_rr is vwap_width_atr/stop_atr.
       If median RR < 1.0, you need >50% win rate to break even (before costs).
    
    5. STOP ASYMMETRY: Wider stops lose more per loss. 
       Stop=1.5 ATR: each stop loss = 1.5 * ATR * 100 = ~$600.
       Even at 45% WR, if avg win is only $200-300, net is deeply negative.
    """)


if __name__ == "__main__":
    main()
