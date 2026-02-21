# 📊 VWAP Reversion Strategy - Complete Analysis

**Generated**: 2026-02-07 17:46:15

**Dataset**: tsla_5min_10years.csv
**Test Period**: 2024+ (40,285 bars, 2.11 years)
**Bars per year**: ~19,092
**Features Used**: 18
**Position Size**: 100 shares @ $250/share

---

## 📌 Summary Tables

### Top 10 Strategies (by EV, RF≥0.50)

| Stop (ATR) | R:R | Win Rate | EV (R) | Trades (test) | % Filtered | Net P&L (test) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 6.05 | 25.6% | +0.802 | 17,403 | 56.8% | $820,078 |
| 0.35 | 4.32 | 33.7% | +0.793 | 17,453 | 56.7% | $1,158,708 |
| 0.40 | 3.78 | 37.0% | +0.770 | 17,955 | 55.4% | $1,329,438 |
| 0.50 | 3.02 | 43.2% | +0.740 | 18,131 | 55.0% | $1,622,813 |
| 0.60 | 2.52 | 48.3% | +0.702 | 18,177 | 54.9% | $1,858,488 |
| 0.75 | 2.02 | 54.0% | +0.628 | 18,472 | 54.2% | $2,120,819 |
| 1.00 | 1.51 | 60.9% | +0.529 | 18,712 | 53.6% | $2,416,886 |
| 1.25 | 1.21 | 64.8% | +0.433 | 19,257 | 52.2% | $2,546,677 |
| 1.50 | 1.01 | 66.4% | +0.334 | 20,884 | 48.2% | $2,549,176 |

### Baseline (No RF Filter, threshold=0.00)

| Stop (ATR) | R:R | Win Rate | EV (R) | Trades (test) | Net P&L (test) |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 6.05 | 15.3% | +0.079 | 40,293 | $77,905 |
| 0.35 | 4.32 | 20.1% | +0.069 | 40,293 | $122,646 |
| 0.40 | 3.78 | 22.3% | +0.067 | 40,293 | $149,880 |
| 0.50 | 3.02 | 26.2% | +0.056 | 40,293 | $160,119 |
| 0.60 | 2.52 | 29.6% | +0.041 | 40,293 | $125,092 |
| 0.75 | 2.02 | 33.7% | +0.016 | 40,293 | $-3,312 |
| 1.00 | 1.51 | 38.8% | -0.026 | 40,293 | $-379,998 |
| 1.25 | 1.21 | 42.4% | -0.063 | 40,293 | $-912,069 |
| 1.50 | 1.01 | 45.2% | -0.093 | 40,293 | $-1,525,674 |

---

## 🔑 Top 15 Features (by importance)

Based on 0.25 ATR stop model:

 1. `vwap_width_atr`
 2. `price_to_vwap_atr`
 3. `bars_from_vwap`
 4. `crossed_vwap`
 5. `rel_vol`
 6. `vwap_slope`
 7. `bar_range_atr`
 8. `vwap_slope_5`
 9. `vwap_helping`
10. `vol_at_extension`
11. `bar_count`
12. `rsi`
13. `close_position`
14. `wap`
15. `rsi_slope`

---

## 🎯 Key Findings & Recommendations

### Best Strategies

- **Best EV (RF≥0.50):** 0.25 ATR | EV=+0.802R | WR=25.6% | R:R=6.05:1
- **Best Net P&L (test, RF≥0.50):** 1.50 ATR | Net P&L=$2,549,176 | Trades=20,884

### Recommended (for scaling tables)

Using **0.50 ATR** as the recommended stop for consistent projection tables (falls back to best-EV stop if unavailable).

- **Recommended stop:** 0.50 ATR
- **RF threshold:** 0.50
- **Win rate:** 43.2%
- **R:R:** 3.02:1
- **EV:** +0.740R
- **Net P&L (test period):** $1,622,813 across 18,131 trades
- **Estimated trades/year:** ~8,593
- **Estimated net P&L/year (100 shares):** $769,106

### Capital & Execution Assumptions

- **Capital per trade (notional):** $25,000 (100 shares × $250)
- **Costs per round trip:** commission+slippage = $3.00
- **Price assumption:** Projections use **AVG_ENTRY_PRICE=$250** for risk sizing and notional.

### Position Scaling (Recommended Strategy)

| Shares | Net P&L / Year | Notional / Trade |
|---:|---:|---:|
| 1 | $7,691 | $250 |
| 10 | $76,911 | $2,500 |
| 25 | $192,276 | $6,250 |
| 50 | $384,553 | $12,500 |
| 100 | $769,106 | $25,000 |
| 200 | $1,538,211 | $50,000 |
| 500 | $3,845,529 | $125,000 |

### Summary

The RF filter improves expected value by selecting higher-quality VWAP reversion setups. The most consistently important signals are VWAP distance (in ATR terms), VWAP slope/dynamics, and volume/momentum context.

### Next Steps

1. Validate stability of feature importance and EV across multiple random seeds and walk-forward splits.
2. Re-estimate selection rate (trades per bar) from the actual test set and remove the static 0.432 shortcut.
3. Integrate explicit capital constraints (max concurrent trades / margin) into the P&L projection.
4. Add slippage sensitivity (e.g., 1–5 cents/share) and verify robustness.

