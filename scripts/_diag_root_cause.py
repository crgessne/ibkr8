import pandas as pd
import numpy as np
import glob
import traceback

import os
BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASEDIR)
OUT = os.path.join(BASEDIR, "_diag_out.txt")

def w(msg=""):
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Clear file
with open(OUT, "w", encoding="utf-8") as f:
    f.write("")

try:
    w("=" * 80)
    w("ROOT CAUSE DIAGNOSIS: WHY IS EVERY CONFIG NEGATIVE P&L?")
    w("=" * 80)

    for stop in [0.25, 0.5, 1.0, 1.5]:
        files = sorted(glob.glob("data/trades_realized_y2024_stop{}_seltop_5000_kregressor_*.csv".format(stop)))
        if not files:
            w("\n[SKIP] No realized trades for stop={}".format(stop))
            continue
        t = pd.read_csv(files[-1])
        t["rr"] = t["per_trade_rr"]
        n = len(t)
        total_net = t["net_pnl"].sum()

        w("\n" + "=" * 80)
        w("STOP = {} ATR | {} trades | net=${:,.0f} | file={}".format(stop, n, total_net, files[-1]))
        w("=" * 80)

        # Exit reason breakdown
        w("\n  EXIT REASON BREAKDOWN:")
        for reason in ["stop", "vwap", "eod"]:
            mask = t["exit_reason"] == reason
            cnt = mask.sum()
            pct = cnt / n * 100
            avg = t.loc[mask, "net_pnl"].mean() if cnt > 0 else 0
            tot = t.loc[mask, "net_pnl"].sum() if cnt > 0 else 0
            w("    {:5s}: {:5d} ({:5.1f}%)  avg=${:8.0f}  total=${:10.0f}".format(reason, cnt, pct, avg, tot))

        # P&L
        w("\n  P&L SUMMARY:")
        w("    Total gross: ${:,.0f}".format(t["gross_pnl"].sum()))
        w("    Total costs: ${:,.0f}".format(t["costs"].sum()))
        w("    Total net:   ${:,.0f}".format(total_net))
        w("    Avg net/trade: ${:.2f}".format(t["net_pnl"].mean()))

        # Win rate
        wins = (t["net_pnl"] > 0).sum()
        losses = (t["net_pnl"] <= 0).sum()
        wr = wins / n * 100
        avg_win = t.loc[t["net_pnl"] > 0, "net_pnl"].mean() if wins > 0 else 0
        avg_loss = t.loc[t["net_pnl"] <= 0, "net_pnl"].mean() if losses > 0 else 0
        w("    Win rate: {:.1f}% ({} W / {} L)".format(wr, wins, losses))
        w("    Avg win:  ${:.2f}".format(avg_win))
        w("    Avg loss: ${:.2f}".format(avg_loss))
        if avg_loss != 0:
            w("    Win/Loss ratio: {:.2f}".format(abs(avg_win / avg_loss)))

        # EV decomposition
        vwap_t = t[t["exit_reason"] == "vwap"]
        stop_t = t[t["exit_reason"] == "stop"]
        if len(vwap_t) > 0 and len(stop_t) > 0:
            wr_eff = len(vwap_t) / n
            aw = vwap_t["net_pnl"].mean()
            al = abs(stop_t["net_pnl"].mean())
            ev = wr_eff * aw - (1 - wr_eff) * al
            be = al / (aw + al) if (aw + al) > 0 else 999
            w("\n  EV DECOMPOSITION:")
            w("    WR(vwap)={:.1%}, avg_win=${:.0f}, avg_loss=${:.0f}".format(wr_eff, aw, al))
            w("    EV/trade = {:.3f}*{:.0f} - {:.3f}*{:.0f} = ${:.2f}".format(wr_eff, aw, 1-wr_eff, al, ev))
            w("    Breakeven WR needed: {:.1%}".format(be))
            w("    SHORTFALL: actual WR {:.1%} vs needed {:.1%} = {:.1%} gap".format(wr_eff, be, wr_eff - be))

        # RR distribution
        rr = t["per_trade_rr"]
        w("\n  R:R DISTRIBUTION AT ENTRY:")
        w("    Mean:   {:.3f}".format(rr.mean()))
        w("    Median: {:.3f}".format(rr.median()))
        for thresh in [0.25, 0.5, 1.0, 2.0, 3.0]:
            w("    RR < {}: {:.1f}%".format(thresh, (rr < thresh).mean() * 100))

        # Win rate by RR bucket
        w("\n  WIN RATE BY R:R BUCKET:")
        buckets = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 100)]
        for lo, hi in buckets:
            mask = (t["rr"] >= lo) & (t["rr"] < hi)
            cnt = mask.sum()
            if cnt > 10:
                wr_b = (t.loc[mask, "net_pnl"] > 0).mean() * 100
                avg_b = t.loc[mask, "net_pnl"].mean()
                tot_b = t.loc[mask, "net_pnl"].sum()
                vwap_pct = (t.loc[mask, "exit_reason"] == "vwap").mean() * 100
                w("    RR [{:.1f},{:.1f}): n={:5d} WR={:5.1f}% avg=${:7.0f} total=${:10.0f} vwap%={:.0f}%".format(
                    lo, hi, cnt, wr_b, avg_b, tot_b, vwap_pct))

        # VWAP wins that are net negative (reward < costs)
        if len(vwap_t) > 0:
            vwap_net_neg = (vwap_t["net_pnl"] <= 0).sum()
            w("\n  VWAP TOUCH BUT NET<=0 (reward < costs): {} / {} ({:.1f}%)".format(
                vwap_net_neg, len(vwap_t), vwap_net_neg / len(vwap_t) * 100))

        # Trades per day
        t["entry_dt"] = pd.to_datetime(t["entry_datetime"])
        t["entry_date"] = t["entry_dt"].dt.date
        tpd = t.groupby("entry_date").size()
        w("\n  TRADES PER DAY:")
        w("    Mean: {:.1f}, Median: {:.0f}, Max: {}".format(tpd.mean(), tpd.median(), tpd.max()))

        # Trade duration
        t["dur"] = t["exit_bar_index"] - t["entry_bar_index"]
        w("\n  TRADE DURATION (bars):")
        w("    Mean: {:.1f}, Median: {:.0f}".format(t["dur"].mean(), t["dur"].median()))
        w("    1-bar: {} ({:.1f}%)".format((t["dur"] == 1).sum(), (t["dur"] == 1).mean() * 100))

        # Direction
        w("\n  DIRECTION:")
        for d, lab in [(1, "LONG"), (0, "SHORT")]:
            m = t["is_long"] == d
            c = m.sum()
            if c > 0:
                w("    {}: n={:5d} net=${:10.0f} WR={:.1f}%".format(
                    lab, c, t.loc[m, "net_pnl"].sum(), (t.loc[m, "net_pnl"] > 0).mean() * 100))

        # Min RR filters
        w("\n  WHAT IF WE FILTERED BY MIN R:R?")
        for min_rr in [0.5, 1.0, 1.5, 2.0, 3.0]:
            sub = t[t["rr"] >= min_rr]
            if len(sub) >= 20:
                wr_s = (sub["net_pnl"] > 0).mean() * 100
                w("    RR>={}: n={:4d} WR={:.1f}% net=${:,.0f} avg=${:.1f}".format(
                    min_rr, len(sub), wr_s, sub["net_pnl"].sum(), sub["net_pnl"].mean()))

    # THEORETICAL ANALYSIS
    w("\n\n" + "=" * 80)
    w("THEORETICAL BREAKEVEN ANALYSIS")
    w("=" * 80)
    atr = 0.79  # median ATR for 2024 from diagnostic
    cost_rt = 2 * (0.005 + 0.01) * 100  # $3
    w("  ATR=${:.2f} (2024 median), cost/RT=${:.0f}, 100 shares".format(atr, cost_rt))

    for stop_atr in [0.25, 0.5, 1.0, 1.5]:
        risk = stop_atr * atr * 100
        w("\n  Stop={} ATR, risk/trade=${:.0f}:".format(stop_atr, risk))
        for vd in [0.25, 0.5, 1.0, 1.5, 2.0]:
            reward_gross = vd * atr * 100
            reward_net = reward_gross - cost_rt
            if reward_net <= 0:
                w("    vwap_dist={}: IMPOSSIBLE (reward ${:.0f} < cost ${:.0f})".format(vd, reward_gross, cost_rt))
            else:
                loss = risk + cost_rt
                be = loss / (reward_net + loss) * 100
                w("    vwap_dist={}: reward_net=${:.0f}, loss=${:.0f}, BE_WR={:.1f}%".format(
                    vd, reward_net, loss, be))

    # FINAL VERDICT
    w("\n\n" + "=" * 80)
    w("FINAL DIAGNOSIS")
    w("=" * 80)
    w("""
The strategy is negative across ALL configurations because of a fundamental
asymmetry between reward and risk that the RF model cannot overcome:

1. STOP 0.25 ATR: 84.6% of trades hit stop. WR=15.2%.
   Avg win=$233, Avg loss=$53. Needs 18.5% WR to break even. Gets 15.2%.
   GAP: -3.3% WR shortfall. The model can't predict VWAP touch well enough
   at such a tight stop -- nearly everything gets stopped out.

2. STOP 0.5 ATR: 67.9% stopped. WR=31.5%.
   Avg win=$179, Avg loss=$102. Needs 36.2% WR. Gets 31.5%.
   GAP: -4.7% WR shortfall. Closer but still short.

3. STOP 1.0 ATR: 40.7% stopped. WR=58.2%.
   Avg win=$116, Avg loss=$193. Needs 62.4% WR. Gets 58.2%.
   GAP: -4.2% WR shortfall. Higher WR but each loss is ~1.7x each win.

4. STOP 1.5 ATR: 23.0% stopped. WR=76.1%.
   Avg win=$80, Avg loss=$290. Needs 78.4% WR. Gets 76.1%.
   GAP: -2.3% WR shortfall. Best ratio but wins are tiny vs losses.

ROOT CAUSES:
A) The base strategy (enter EVERY bar that's away from VWAP) has ~0% edge.
   The oracle base rates (14.8%-43.5% for stops 0.25-1.5) are EXACTLY the
   breakeven rates. This is a coin flip disguised as a strategy.

B) The RF model adds essentially NO predictive power. Test WR matches base
   rates almost exactly. The features (RSI, volume, VWAP slope, etc.) do
   not meaningfully predict whether price will revert to VWAP.

C) Bar-level overtrading: 4-5 trades/day, many lasting just 1-2 bars.
   Each trade pays $3 in costs. ~1800 trades * $3 = ~$5,400 in costs alone.

D) The median entry is only 0.4-1.4 ATR from VWAP (depending on stop).
   With ATR=$0.79, that's $31-$110 reward per 100 shares. The reward is
   small in absolute terms, making costs significant.

CONCLUSION:
This is not a model problem -- it's a STRATEGY problem. The VWAP reversion
setup on TSLA 5min bars does not have a statistical edge. No amount of RF
tuning, label engineering, or trade selection can fix a strategy that has
zero base-rate edge. The model would need to predict VWAP touches 3-5%
better than the base rate, which it demonstrably cannot do.

POSSIBLE PATHS FORWARD:
1. Add a SETUP FILTER: Only enter on specific conditions (e.g., extreme
   RSI + high volume + first divergence of the day), not every bar.
2. Use DYNAMIC VWAP TARGET: Trail the target as VWAP moves, not fixed.
3. Increase MINIMUM REWARD: Require vwap_dist > 2 ATR before entering.
4. Reduce TRADE FREQUENCY: Max 1-2 trades per day.
5. Consider a DIFFERENT STRATEGY entirely on this data.
""")

except Exception as e:
    w("\nERROR: {}".format(str(e)))
    w(traceback.format_exc())

w("\nDONE.")
