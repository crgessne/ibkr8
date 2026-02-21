"""Pull today's executions from IBKR and show P&L, slippage, open positions."""
import sys, os
sys.path.insert(0, r'C:\Users\Administrator\ibkr8')
os.chdir(r'C:\Users\Administrator\ibkr8')

out = open(r'C:\Users\Administrator\ibkr8\_analyze_live_trades_out.txt', 'w', encoding='utf-8')

def log(msg=''):
    out.write(str(msg) + '\n')
    out.flush()

PORT = 7497   # paper; change to 7496 for live
MODE = 'PAPER'

try:
    from ib_insync import IB
    import pandas as pd
    import numpy as np
    from datetime import date

    TODAY = date.today().isoformat()

    ib = IB()
    ib.connect('127.0.0.1', PORT, clientId=20)
    log(f'Connected to IBKR {MODE} port {PORT}')
    log(f'Accounts: {ib.managedAccounts()}')
    log(f'Date: {TODAY}')

    # ── Fills (execution + commission in one call) ────────────────────
    log('\n=== ALL FILLS (with P&L) ===')
    fills = ib.fills()
    log(f'Total fills returned: {len(fills)}')

    rows = []
    for f in fills:
        ex  = f.execution
        cr  = f.commissionReport
        ctr = f.contract
        dt  = ex.time  # datetime object
        rows.append({
            'date':         dt.date().isoformat() if dt else '',
            'time':         dt.strftime('%H:%M:%S') if dt else '',
            'symbol':       ctr.symbol,
            'side':         ex.side,
            'shares':       int(ex.shares),
            'fill_px':      float(ex.price),
            'commission':   float(cr.commission)  if cr else 0.0,
            'realized_pnl': float(cr.realizedPNL) if cr else float('nan'),
        })

    if not rows:
        log('No fills found.')
    else:
        df = pd.DataFrame(rows)
        # Filter to today
        tdf = df[df['date'] == TODAY].copy()
        log(f'Total fills ever:  {len(df)}')
        log(f"Today's fills:     {len(tdf)}")

        if len(tdf) == 0:
            log(f'\nNo fills today. Most recent 20 fills:')
            log(df.tail(20).to_string(index=False))
        else:
            log('\n' + tdf.to_string(index=False))

            # Summary — only rows where IB reported realized P&L
            IBKR_NAN = 1.7976931348623157e+308
            pnl_df = tdf[
                tdf['realized_pnl'].notna() &
                (tdf['realized_pnl'].abs() < IBKR_NAN / 2)
            ]

            log('\n=== TODAY SUMMARY ===')
            log(f'  Fills:               {len(tdf)}')
            log(f'  Realized P&L rows:   {len(pnl_df)}')
            if len(pnl_df) > 0:
                gross = pnl_df['realized_pnl'].sum()
                comm  = tdf['commission'].sum()
                log(f'  Gross realized P&L:  ${gross:>12,.2f}')
                log(f'  Total commission:    ${comm:>12,.2f}')
                log(f'  Net (P&L - comm):    ${gross - comm:>12,.2f}')

            # Per-symbol breakdown
            log('\n=== BY SYMBOL ===')
            for sym, g in tdf.groupby('symbol'):
                pnl_g = g[g['realized_pnl'].notna() & (g['realized_pnl'].abs() < IBKR_NAN / 2)]
                gross_sym = pnl_g['realized_pnl'].sum()
                comm_sym  = g['commission'].sum()
                log(f'  {sym:8s}  fills={len(g):3d}  '
                    f'gross_pnl=${gross_sym:>10,.2f}  '
                    f'comm=${comm_sym:>7,.2f}  '
                    f'net=${gross_sym - comm_sym:>10,.2f}')

            # Per-fill detail with slippage
            log('\n=== PER-FILL DETAIL ===')
            for _, r in tdf.iterrows():
                pnl_str = f'  pnl=${r["realized_pnl"]:+,.2f}' if abs(r['realized_pnl']) < IBKR_NAN / 2 else ''
                log(f'  {r["time"]}  {r["side"]:4s}  {r["shares"]:5d} {r["symbol"]:8s}'
                    f'  @ ${r["fill_px"]:.4f}'
                    f'  comm=${r["commission"]:.2f}'
                    f'{pnl_str}')

    # ── Open positions ────────────────────────────────────────────────
    log('\n=== OPEN POSITIONS ===')
    ib.reqPositions()
    ib.sleep(0.5)
    for p in ib.positions():
        log(f'  {p.contract.symbol:8s}  {p.position:+.0f} shares  '
            f'avg_cost=${p.avgCost:.4f}  account={p.account}')
    if not ib.positions():
        log('  (none)')

    # ── Open orders ───────────────────────────────────────────────────
    log('\n=== OPEN ORDERS ===')
    for t in ib.openTrades():
        o, s = t.order, t.orderStatus
        log(f'  {t.contract.symbol:8s}  {o.action:4s}  qty={o.totalQuantity:.0f}'
            f'  type={o.orderType}  lmt={getattr(o,"lmtPrice","-")}'
            f'  aux={getattr(o,"auxPrice","-")}  status={s.status}  filled={s.filled}')
    if not ib.openTrades():
        log('  (none)')

    # ── Account ───────────────────────────────────────────────────────
    log('\n=== ACCOUNT ===')
    want = {'NetLiquidation','TotalCashValue','UnrealizedPnL','RealizedPnL',
            'GrossPositionValue','DailyPnL'}
    for row in ib.accountSummary():
        if row.tag in want:
            log(f'  {row.tag:25s} = {row.value:>15s} {row.currency}')

    ib.disconnect()
    log('\nDone.')

except Exception as exc:
    import traceback
    log(f'\nERROR: {exc}')
    log(traceback.format_exc())

out.close()
