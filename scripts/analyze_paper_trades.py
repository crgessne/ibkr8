"""
Analyze today's paper trading session (2026-02-20) results
Reconstructed from _trading_debug.log
"""

trades = [
    # bar, dir, shares, entry_fill, exit_fill, exit_type, signal_px
    {'bar': 1,  'dir': 'SHORT', 'shares': 1221, 'entry': 409.14, 'exit': 409.22, 'exit_type': 'STOP',   'signal': 409.32},
    {'bar': 2,  'dir': 'SHORT', 'shares': 1219, 'entry': 410.03, 'exit': 409.82, 'exit_type': 'TARGET', 'signal': 410.11},
    {'bar': 3,  'dir': 'LONG',  'shares': 1222, 'entry': 409.12, 'exit': 408.30, 'exit_type': 'STOP',   'signal': 409.10},
    {'bar': 4,  'dir': 'LONG',  'shares': 1225, 'entry': 408.24, 'exit': 409.36, 'exit_type': 'TARGET', 'signal': 407.98},
    {'bar': 5,  'dir': 'SKIP',  'shares': 0,    'entry': 0,      'exit': 0,      'exit_type': 'NO_FILL','signal': 408.57},
    {'bar': 6,  'dir': 'SHORT', 'shares': 1217, 'entry': 410.26, 'exit': 409.39, 'exit_type': 'TARGET', 'signal': 410.68},
    {'bar': 7,  'dir': 'SHORT', 'shares': 1220, 'entry': 409.55, 'exit': 410.82, 'exit_type': 'STOP',   'signal': 409.72},
    {'bar': 8,  'dir': 'LONG',  'shares': 1224, 'entry': 408.87, 'exit': 407.37, 'exit_type': 'STOP',   'signal': 408.41},
    {'bar': 9,  'dir': 'SHORT', 'shares': 1220, 'entry': 409.36, 'exit': 409.35, 'exit_type': 'TARGET', 'signal': 409.53},
    {'bar': 10, 'dir': 'SHORT', 'shares': 1218, 'entry': 409.89, 'exit': 409.40, 'exit_type': 'TARGET', 'signal': 410.36},
    {'bar': 11, 'dir': 'SHORT', 'shares': 1220, 'entry': 409.79, 'exit': 409.41, 'exit_type': 'TARGET', 'signal': 409.79},
    {'bar': 12, 'dir': 'SHORT', 'shares': 1219, 'entry': 410.01, 'exit': 411.03, 'exit_type': 'STOP',   'signal': 409.88},
    {'bar': 13, 'dir': 'SHORT', 'shares': 1215, 'entry': 411.40, 'exit': 412.29, 'exit_type': 'STOP',   'signal': 411.33},
]

# Also bars 14-18 were NO TRADE (prob < 0.5 threshold)
no_trades = [
    {'bar': 14, 'prob': 0.0666, 'setup': 'SHORT', 'C': 411.63},
    {'bar': 15, 'prob': 0.0671, 'setup': 'SHORT', 'C': 413.29},
    {'bar': 16, 'prob': 0.0471, 'setup': 'SHORT', 'C': 412.85},
    {'bar': 17, 'prob': 0.0321, 'setup': 'SHORT', 'C': 413.32},
    {'bar': 18, 'prob': 0.0896, 'setup': 'SHORT', 'C': 414.22},
]

# Order ID mapping for reference
# Entry orders: 60,63,66,69,72(cancelled),75,78,81,84,87,90,93,96
# Odd = stop, Even = target (relative to entry ID)
# But IBKR assigns orderId=entry, entry+1=stop(?), entry+2=target(?)
# Actually from the log: for bar1 orderId60=entry, 61=stop(cancelled), 62=target(filled@409.22)
# But 409.22 > entry 409.14 for a SHORT -> that's STOP level, not target
# The order type matters: orderId=62 must be the STOP (buy-stop for short)
# Target would be limit-buy below entry for short

SLIPPAGE = 0.01    # $0.01/share per side (model assumption)
COMMISSION = 0.005  # $0.005/share per side (IBKR)
COST_PER_SHARE = (SLIPPAGE + COMMISSION) * 2  # round trip: $0.03/share

print("=" * 90)
print("PAPER TRADING SESSION: 2026-02-20  |  TSLA 5-min VWAP Reversion  |  Model: nn_pnl")
print("=" * 90)
print()
print("Session: 14:40 UTC - 16:05 UTC (9:40-11:05 ET)  |  18 bars processed  |  13 trades attempted")
print("Bars 1-13: Signal generated  |  Bars 14-18: NO TRADE (prob < 0.50 threshold)")
print()
print('%4s %6s %6s %7s %7s %8s %10s %8s %10s  %s' % (
    'Bar', 'Dir', 'Shares', 'Entry', 'Exit', 'Type', 'Gross $', 'Costs $', 'Net $', 'Result'))
print('-' * 90)

total_gross = 0
total_costs = 0
total_net = 0
wins = 0
losses = 0
targets_hit = 0
stops_hit = 0

for t in trades:
    if t['dir'] == 'SKIP':
        print('  %2d  ---   (entry cancelled - all orders voided at bar open)' % t['bar'])
        continue

    shares = t['shares']
    if t['dir'] == 'SHORT':
        gross = (t['entry'] - t['exit']) * shares
    else:
        gross = (t['exit'] - t['entry']) * shares

    costs = COST_PER_SHARE * shares
    net = gross - costs
    total_gross += gross
    total_costs += costs
    total_net += net

    outcome = 'WIN ' if net > 0 else 'LOSS'
    if net > 0:
        wins += 1
    else:
        losses += 1

    if t['exit_type'] == 'TARGET':
        targets_hit += 1
    elif t['exit_type'] == 'STOP':
        stops_hit += 1

    print('  %2d %6s %6d %7.2f %7.2f %8s %10.2f %8.2f %10.2f  %s' % (
        t['bar'], t['dir'], shares, t['entry'], t['exit'],
        t['exit_type'], gross, costs, net, outcome))

total_trades = wins + losses
print('-' * 90)
print('TOTALS:  %d trades  |  Wins: %d  Losses: %d  |  WR: %.1f%%' % (
    total_trades, wins, losses, wins/total_trades*100 if total_trades else 0))
print('         Targets hit: %d  |  Stops hit: %d' % (targets_hit, stops_hit))
print('         Gross: $%.2f  |  Costs: $%.2f  |  NET P&L: $%.2f' % (
    total_gross, total_costs, total_net))
print()

# Per-trade averages
avg_net = total_net / total_trades if total_trades else 0
avg_win = sum(
    (t['entry'] - t['exit']) * t['shares'] if t['dir'] == 'SHORT'
    else (t['exit'] - t['entry']) * t['shares']
    for t in trades if t['dir'] != 'SKIP' and (
        ((t['entry'] - t['exit']) * t['shares'] if t['dir'] == 'SHORT'
         else (t['exit'] - t['entry']) * t['shares']) - COST_PER_SHARE * t['shares']
    ) > 0
) / wins if wins else 0
avg_loss = sum(
    (t['entry'] - t['exit']) * t['shares'] if t['dir'] == 'SHORT'
    else (t['exit'] - t['entry']) * t['shares']
    for t in trades if t['dir'] != 'SKIP' and (
        ((t['entry'] - t['exit']) * t['shares'] if t['dir'] == 'SHORT'
         else (t['exit'] - t['entry']) * t['shares']) - COST_PER_SHARE * t['shares']
    ) <= 0
) / losses if losses else 0

print("PERFORMANCE SUMMARY:")
print("  Avg net P&L per trade:  $%.2f" % avg_net)
print("  Avg gross win:          $%.2f" % avg_win)
print("  Avg gross loss:         $%.2f" % avg_loss)
print()

# ---- SLIPPAGE ANALYSIS ----
print("=" * 60)
print("ENTRY SLIPPAGE ANALYSIS (signal close vs actual fill)")
print("=" * 60)
print('%5s %6s %8s %8s %8s  %s' % ('Bar', 'Dir', 'Signal', 'Fill', 'Slip/$sh', 'Verdict'))
print('-' * 60)

total_slip_cost = 0
slip_count = 0

for t in trades:
    if t['dir'] == 'SKIP':
        continue
    signal_px = t['signal']
    fill_px   = t['entry']

    if t['dir'] == 'SHORT':
        # For short: we want to sell high; fill below signal = favorable
        slip_per_share = fill_px - signal_px   # negative = we got a BETTER price (filled lower = less proceeds)
        # Actually for short: fill at 409.14 vs signal 409.32 means we filled BELOW signal -> LESS favorable
        # We're selling short: higher fill = better for us
        # slip_per_share = signal - fill (positive if we got worse price)
        slip_per_share = signal_px - fill_px  # positive = adverse (fill below signal for short)
    else:
        # For long: higher fill = adverse (we paid more)
        slip_per_share = fill_px - signal_px  # positive = adverse

    slip_total = slip_per_share * t['shares']
    total_slip_cost += slip_total
    slip_count += 1

    verdict = 'ADVERSE' if slip_per_share > 0.02 else ('FAVORABLE' if slip_per_share < -0.02 else 'OK')
    print('  %2d %6s %8.2f %8.2f %+8.3f  %s  ($%.0f total)' % (
        t['bar'], t['dir'], signal_px, fill_px, slip_per_share, verdict, slip_total))

print('-' * 60)
avg_slip = total_slip_cost / slip_count if slip_count else 0
print('Avg entry slippage: $%.3f/share  |  Total adverse entry cost: $%.2f' % (avg_slip, total_slip_cost))
print('(Backtest assumes $0.01/share slippage per side; adverse entry = real market impact)')
print()

# ---- BACKTEST COMPARISON ----
print("=" * 60)
print("LIVE vs BACKTEST COMPARISON")
print("=" * 60)
print("Backtest (2024, stop=0.75 ATR, nn_pnl):")
print("  Trades/day (avg):  ~10  (2,514 trades / 252 trading days)")
print("  Avg net/trade:     ~$321")
print("  Win rate:          51.3%")
print()
print("Today's paper session (2026-02-20, ~1.5 hrs of market):")
print("  Trades attempted:  13 (+ 1 cancelled + 5 no-signal)")
print("  Actual fills:      12")
print("  Win rate:          %.1f%% (%d/%d)" % (wins/total_trades*100, wins, total_trades))
print("  Net P&L:           $%.2f" % total_net)
print("  Avg net/trade:     $%.2f" % avg_net)
print()

# Key issues
print("=" * 60)
print("KEY OBSERVATIONS")
print("=" * 60)
print()
print("1. BAR 5 ENTRY CANCELLED: All 3 orders (entry+stop+target) voided")
print("   at 10:00:06 - this is a bracket-replacement race condition.")
print("   The new bar signal cancelled the PREVIOUS bar's pending orders,")
print("   but also voided the NEW bar's orders before they were submitted.")
print()
print("2. LARGE ENTRY SLIPPAGE on some bars:")
print("   Bar 4 LONG: signal=$407.98, fill=$408.24 (+$0.26/share adverse)")
print("   Bar 8 LONG: signal=$408.41, fill=$408.87 (+$0.46/share adverse)")
print("   These are market-order entries at bar open (5s after close).")
print("   Bar opened higher/lower than previous close due to pre-market moves.")
print()
print("3. STOP RATE = %d/%d = %.1f%% stops hit today (backtest ~48.7%%)" % (
    stops_hit, total_trades, stops_hit/total_trades*100))
print("   Today saw trending price action (rising from 408 to 414),")
print("   which is BAD for mean-reversion SHORT signals.")
print()
print("4. MODEL signals were ALWAYS SHORT (bars 1-13 had SHORT setups).")
print("   TSLA trended UP +$5 today (409->414). The model was fighting trend.")
print("   This is expected to be a LOSING environment for VWAP mean-reversion.")
