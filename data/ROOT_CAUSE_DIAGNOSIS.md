# ROOT CAUSE: Why Every Configuration Produces Negative P&L

## Executive Summary

**The VWAP reversion strategy on TSLA 5-min bars is fundamentally unprofitable.** This is not a model problem, feature engineering problem, or hyperparameter problem. The base strategy has **zero edge** — and the RF model adds essentially no predictive power on top of it.

---

## The Smoking Gun: Oracle Base Rate vs Breakeven Win Rate

The **oracle base rate** is the win rate you'd get if you entered EVERY bar (no model at all).
The **breakeven WR** is the minimum win rate needed to make $0 net, given the asymmetric payoff.

| Stop (ATR) | Oracle Base Rate | Breakeven WR | Gap | Verdict |
|------------|-----------------|--------------|-----|---------|
| 0.25 | 14.8% | 18.6% | **-3.8%** | Underwater even with perfect info |
| 0.50 | 25.2% | 36.2% | **-11.0%** | Far underwater |
| 1.00 | 37.6% | 62.4% | **-24.8%** | Massively underwater |
| 1.50 | 43.5% | 78.4% | **-34.9%** | Hopeless |

**At every stop level, the oracle (all-bars) win rate is BELOW breakeven.** This means even if you had a perfect classifier that perfectly predicted which bars will touch VWAP, you still couldn't overcome the payoff asymmetry at wider stops — because the wins that DO happen are too small relative to the losses.

The RF model's actual test win rate matches the oracle base rate almost exactly (within 1-2%), confirming it adds ~zero predictive edge.

---

## EV Decomposition by Stop Level (Actual Trades, 2024)

### Stop 0.25 ATR (2,514 trades)
- Exit: 84.6% stop, 15.2% vwap, 0.2% eod
- Avg win: **$233** | Avg loss: **$53** | Win/Loss ratio: 4.39
- EV/trade = 0.152 × $233 − 0.848 × $53 = **−$9.73**
- WR shortfall: 15.2% actual vs 18.6% needed = **−3.4%**
- Net P&L: **−$22,711**

### Stop 0.50 ATR (1,814 trades)
- Exit: 67.9% stop, 31.5% vwap, 0.6% eod
- Avg win: **$179** | Avg loss: **$102** | Win/Loss ratio: 1.78
- EV/trade = 0.315 × $179 − 0.685 × $102 = **−$12.99**
- WR shortfall: 31.5% vs 36.2% = **−4.6%**
- Net P&L: **−$19,783**

### Stop 1.00 ATR (1,453 trades)
- Exit: 40.7% stop, 58.2% vwap, 1.1% eod
- Avg win: **$116** | Avg loss: **$193** | Win/Loss ratio: 0.64
- EV/trade = 0.582 × $116 − 0.418 × $193 = **−$12.83**
- WR shortfall: 58.2% vs 62.4% = **−4.2%**
- Net P&L: **−$15,236**

### Stop 1.50 ATR (1,600 trades)
- Exit: 23.0% stop, 76.1% vwap, 0.9% eod
- Avg win: **$80** | Avg loss: **$290** | Win/Loss ratio: 0.32
- EV/trade = 0.761 × $80 − 0.239 × $290 = **−$8.44**
- WR shortfall: 76.1% vs 78.4% = **−2.3%**
- Net P&L: **−$10,159**

---

## Why "EV in R" Looked Positive But $ P&L Is Negative

The earlier EV-in-R metric used `median_vwap_dist_atr / stop_atr` as the headline R:R. For stop 0.25, with median vwap_dist ~1.5 ATR, that gives R:R = 6:1, so even a 15% WR gives "positive EV in R":

```
EV_R = 0.15 × 6.0 − 0.85 × 1.0 = +0.05R  (looks positive!)
```

But this is **misleading** because:
1. The median reward ($233) and loss ($53) at stop 0.25 are in real dollars, not R-units
2. Costs ($3/trade) eat into the tiny per-trade edge
3. The R:R metric ignores that most high-RR trades (RR > 5) have only 10% WR — still below breakeven

---

## Why Min-RR Filters Don't Help

| Stop 0.5 ATR | Trades | WR | Net P&L | Avg P&L |
|--------------|--------|-----|---------|---------|
| No filter | 1,814 | 32.1% | −$19,783 | −$10.91 |
| RR ≥ 0.5 | 1,769 | 31.2% | −$19,621 | −$11.09 |
| RR ≥ 1.0 | 1,520 | 26.4% | −$20,165 | −$13.27 |
| RR ≥ 2.0 | 1,043 | 20.9% | −$15,236 | −$14.61 |
| RR ≥ 3.0 | 619 | 16.6% | −$9,130 | −$14.75 |

**Higher RR requirements actually make avg P&L WORSE**, because entries far from VWAP are less likely to reach VWAP. The higher reward-per-win is offset by a proportionally lower win rate. The net effect is zero — it's a self-canceling filter.

---

## Contributing Factors

### 1. Bar-Level Overtrading (4-5 trades/day)
- Flat-to-flat simulation correctly prevents overlapping positions
- But the strategy still enters 4-5 trades/day, paying $3 costs each = ~$15/day × 252 days = **~$3,800/year in costs alone**
- 35-64% of trades are only 1-bar duration (entered and stopped in 5 minutes)

### 2. Shorts Significantly Worse Than Longs
At every stop level, SHORT trades lose much more than LONG:
- Stop 0.5: LONG net = −$1,615 (WR 39.9%) vs SHORT net = −$18,168 (WR 30.2%)
- Stop 1.0: LONG net = −$2,904 (WR 66.0%) vs SHORT net = −$12,332 (WR 53.6%)

This suggests TSLA has a long bias in 2024, making short VWAP-reversion trades structurally disadvantaged.

### 3. Costs Relative to Reward
- Median ATR in 2024: **$0.79**
- At stop 1.5 ATR: median entry is only 0.27 ATR from VWAP = **$21 reward per 100 shares**
- Round-trip cost: $3.00 = **14% of gross reward eaten by costs**

---

## Conclusion

The VWAP reversion strategy as currently defined ("enter at close of any bar where price ≠ VWAP, target = fixed entry-bar VWAP, stop = N×ATR") is a **zero-edge strategy** on TSLA 5-minute data. The market already prices in VWAP reversion — the base-rate probability of touching VWAP is almost exactly the breakeven rate at every stop width. The RF model cannot improve on this because the features available do not contain information about future VWAP touches beyond what the base rate already reflects.

## Actionable Paths Forward

1. **Setup-Based Entry**: Instead of entering every bar, define a specific "setup" (e.g., >2 ATR from VWAP + RSI extreme + volume spike + first divergence of session). Test if this subset has a higher WR than the base rate.

2. **Dynamic VWAP Target**: Use rolling VWAP as target (not fixed at entry-bar VWAP). VWAP naturally moves toward price intraday, which could increase WR.

3. **Long-Only**: Drop short entries entirely given the clear TSLA long bias.

4. **Max 1-2 Trades/Day**: Reduce cost drag by only taking the "best" setup per day.

5. **Different Instrument**: Test on an instrument with stronger mean-reversion properties (e.g., SPY, which has stronger VWAP magnet effects).

6. **Different Strategy**: Consider momentum/breakout strategies which may have an actual edge on a high-beta stock like TSLA.
