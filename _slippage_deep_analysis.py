"""
Deep analysis of slippage sensitivity and what it takes to be profitable.
Answers the key question: at what slippage does the strategy break even?
"""
import pandas as pd
import numpy as np
import os

os.chdir(r'C:\Users\Administrator\ibkr8')

# ── 1. Load realized trades ($0.01 slippage model) ──────────────────────────
df = pd.read_csv('data/trades_realized_y2024_stop0.75_selprob_0.50_kclassifier_20260218_192628.csv')
print(f"Loaded {len(df)} trades | Win rate: {df['win'].mean():.1%}")
print(f"Columns: {df.columns.tolist()}")
print()

# ── 2. Understand trade structure ────────────────────────────────────────────
print("=== TRADE STRUCTURE ===")
for c in ['vwap_dist_atr', 'stop_dist', 'shares', 'risk_dollars', 'net_pnl', 'gross_pnl']:
    if c in df.columns:
        print(f"  {c:20s}: mean={df[c].mean():.4f}  median={df[c].median():.4f}  p25={df[c].quantile(0.25):.4f}  p75={df[c].quantile(0.75):.4f}")
print()

# ── 3. Derive target distance in $ ──────────────────────────────────────────
# vwap_dist_atr is the outcome variable = distance price moved toward VWAP in ATR units
# At stop=0.75 ATR: stop_dist = 0.75 * ATR  => ATR = stop_dist / 0.75
# target_dist_$ = vwap_dist_atr * ATR  (but target is the entry-bar VWAP so it's actually stop_dist/0.75 * vwap_dist_atr? Let's check)
if 'stop_dist' in df.columns and 'vwap_dist_atr' in df.columns:
    atr_est = df['stop_dist'] / 0.75
    target_dist = df['vwap_dist_atr'] * atr_est
    print("=== TARGET DISTANCE IN $ ===")
    print(f"  Avg ATR: ${atr_est.mean():.3f}  Median ATR: ${atr_est.median():.3f}")
    print(f"  Avg target_dist: ${target_dist.mean():.3f}  Median target_dist: ${target_dist.median():.3f}")
    print(f"  Avg stop_dist:   ${df['stop_dist'].mean():.3f}  Median stop_dist: ${df['stop_dist'].median():.3f}")
    print(f"  R:R ratio (avg): {(target_dist / df['stop_dist']).mean():.3f}")
    print(f"  R:R ratio (median): {(target_dist / df['stop_dist']).median():.3f}")
    print(f"  % trades with target < $0.25: {(target_dist < 0.25).mean():.1%}")
    print(f"  % trades with target < $0.50: {(target_dist < 0.50).mean():.1%}")
    print(f"  % trades with target < $1.00: {(target_dist < 1.00).mean():.1%}")
    df['target_dist'] = target_dist
    df['atr_est'] = atr_est
    df['rr_ratio'] = target_dist / df['stop_dist']
    print()

# ── 4. Slippage sensitivity: simulate different slippage levels ──────────────
print("=== SLIPPAGE SENSITIVITY (stop=0.75 ATR) ===")
print(f"{'Slip/share':>12} {'Net P&L':>12} {'P&L/trade':>12} {'Break-even':>12}")

for slip in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
    if 'net_pnl' in df.columns and 'shares' in df.columns:
        # Original net_pnl was computed with $0.01 slippage
        # Reverse: gross_pnl = net_pnl + 0.01*2*shares (entry+exit slip)
        # Then apply new slippage
        orig_slip = 0.01
        if 'gross_pnl' in df.columns:
            gross = df['gross_pnl']
        else:
            gross = df['net_pnl'] + orig_slip * 2 * df['shares']
        
        new_net = gross - slip * 2 * df['shares']
        total = new_net.sum()
        per_trade = new_net.mean()
        print(f"  ${slip:.2f}         ${total:>12,.0f}   ${per_trade:>9,.1f}/trade")

print()

# ── 5. Slippage break-even per share ────────────────────────────────────────
if 'gross_pnl' in df.columns and 'shares' in df.columns:
    total_gross = df['gross_pnl'].sum()
    total_shares_roundtrip = (df['shares'] * 2).sum()
    be_slip = total_gross / total_shares_roundtrip
    print(f"=== BREAK-EVEN SLIPPAGE ===")
    print(f"  Total gross P&L: ${total_gross:,.0f}")
    print(f"  Total round-trip shares: {total_shares_roundtrip:,.0f}")
    print(f"  Break-even slippage: ${be_slip:.4f}/share")
    print()

# ── 6. Analyze which trades are most slippage-sensitive ─────────────────────
print("=== SLIPPAGE IMPACT BY TRADE SIZE ===")
if 'shares' in df.columns:
    df['slip_cost_per_rtr'] = df['shares'] * 2 * 0.01  # at $0.01
    df['slip_pct_of_gross'] = df['slip_cost_per_rtr'] / df['gross_pnl'].abs()
    
    print(f"  Avg shares per trade: {df['shares'].mean():.0f}")
    print(f"  Median shares: {df['shares'].median():.0f}")
    print(f"  p90 shares: {df['shares'].quantile(0.90):.0f}")
    print(f"  p99 shares: {df['shares'].quantile(0.99):.0f}")
    print(f"  At $0.01: avg slip cost ${df['slip_cost_per_rtr'].mean():.2f}/trade")
    print(f"  At $0.05: avg slip cost ${df['shares'].mean()*2*0.05:.2f}/trade")
    print()

# ── 7. What if we only traded high R:R setups? ──────────────────────────────
if 'rr_ratio' in df.columns:
    print("=== FILTER BY R:R RATIO (at $0.05 slippage) ===")
    print(f"{'Min R:R':>8} {'N trades':>10} {'Win rate':>10} {'Net P&L @0.05':>15} {'P&L/trade':>12}")
    
    for rr_min in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        sub = df[df['rr_ratio'] >= rr_min].copy()
        if len(sub) == 0:
            continue
        if 'gross_pnl' in df.columns:
            gross = sub['gross_pnl']
        else:
            gross = sub['net_pnl'] + 0.01 * 2 * sub['shares']
        net05 = gross - 0.05 * 2 * sub['shares']
        total = net05.sum()
        per_t = net05.mean()
        wr = sub['win'].mean()
        print(f"  >={rr_min:.2f}   {len(sub):>10,}   {wr:>10.1%}   ${total:>13,.0f}   ${per_t:>9,.1f}/trade")
    print()

# ── 8. What if we only traded large target distances? ──────────────────────
if 'target_dist' in df.columns:
    print("=== FILTER BY TARGET DISTANCE (at $0.05 slippage) ===")
    print(f"{'Min target':>12} {'N trades':>10} {'Win rate':>10} {'Net P&L @0.05':>15} {'P&L/trade':>12}")
    
    for tgt_min in [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]:
        sub = df[df['target_dist'] >= tgt_min].copy()
        if len(sub) == 0:
            continue
        if 'gross_pnl' in df.columns:
            gross = sub['gross_pnl']
        else:
            gross = sub['net_pnl'] + 0.01 * 2 * sub['shares']
        net05 = gross - 0.05 * 2 * sub['shares']
        total = net05.sum()
        per_t = net05.mean()
        wr = sub['win'].mean()
        print(f"  >=${tgt_min:.2f}        {len(sub):>10,}   {wr:>10.1%}   ${total:>13,.0f}   ${per_t:>9,.1f}/trade")
    print()

# ── 9. What is the actual label definition? ──────────────────────────────────
print("=== LABEL ANALYSIS ===")
if 'label' in df.columns:
    print(f"  label distribution: {df['label'].value_counts().to_dict()}")
if 'win' in df.columns:
    print(f"  win distribution: {df['win'].value_counts().to_dict()}")

# ── 10. Check if there's a minimum profit requirement ─────────────────────
print()
print("=== DIAGNOSIS: WHY DOES $0.05 KILL PERFORMANCE? ===")
if 'shares' in df.columns and 'gross_pnl' in df.columns:
    slip_delta = 0.04  # from $0.01 to $0.05
    slip_impact = slip_delta * 2 * df['shares'].sum()
    print(f"  Extra slippage cost ($0.01 -> $0.05): ${slip_impact:,.0f}")
    print(f"  = ${slip_impact/len(df):,.0f}/trade extra cost")
    print(f"  Original gross P&L: ${df['gross_pnl'].sum():,.0f}")
    print(f"  Net at $0.01: ${(df['gross_pnl'] - 0.01*2*df['shares']).sum():,.0f}")
    print(f"  Net at $0.05: ${(df['gross_pnl'] - 0.05*2*df['shares']).sum():,.0f}")
    print()
    print("  CONCLUSION: Very large position sizes make the strategy extremely sensitive")
    print(f"  Avg shares: {df['shares'].mean():.0f} => each extra $0.01/share = ${df['shares'].mean()*2*0.01:.2f}/trade")

print("\nDone.")
