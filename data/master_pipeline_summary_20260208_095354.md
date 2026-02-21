# 📊 VWAP Reversion Strategy - Complete Analysis

**Generated**: 2026-02-08 09:54:08

**Dataset**: tsla_5min_10years.csv
**Test Period**: 2024+ (40,293 eligible bars, 2.10 years)
**Eligible bars per year**: ~19,213
**Features Used**: 18
**Position Size**: 100 shares @ $400/share

---

## 📌 Summary Tables

### Top 10 Strategies (by EV, RF≥0.50)

| Stop (ATR) | R:R | Win Rate | EV (R) | Trades (test) | % Filtered | Net P&L (test) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 6.05 | 25.6% | +0.802 | 17,403 | 56.8% | $1,343,450 |
| 0.35 | 4.32 | 33.7% | +0.793 | 17,453 | 56.7% | $1,885,349 |
| 0.40 | 3.78 | 37.0% | +0.770 | 17,955 | 55.4% | $2,159,420 |
| 0.50 | 3.02 | 43.2% | +0.740 | 18,131 | 55.0% | $2,629,137 |
| 0.60 | 2.52 | 48.3% | +0.702 | 18,177 | 54.9% | $3,006,299 |
| 0.75 | 2.02 | 54.0% | +0.628 | 18,472 | 54.2% | $3,426,560 |
| 1.00 | 1.51 | 60.9% | +0.529 | 18,712 | 53.6% | $3,900,700 |
| 1.25 | 1.21 | 64.8% | +0.433 | 19,257 | 52.2% | $4,109,345 |
| 1.50 | 1.01 | 66.4% | +0.334 | 20,884 | 48.2% | $4,116,273 |

### Baseline (No RF Filter, threshold=0.00)

| Stop (ATR) | R:R | Win Rate | EV (R) | Trades (test) | Net P&L (test) |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 6.05 | 15.3% | +0.079 | 40,293 | $197,176 |
| 0.35 | 4.32 | 20.1% | +0.069 | 40,293 | $268,761 |
| 0.40 | 3.78 | 22.3% | +0.067 | 40,293 | $312,335 |
| 0.50 | 3.02 | 26.2% | +0.056 | 40,293 | $328,718 |
| 0.60 | 2.52 | 29.6% | +0.041 | 40,293 | $272,674 |
| 0.75 | 2.02 | 33.7% | +0.016 | 40,293 | $67,229 |
| 1.00 | 1.51 | 38.8% | -0.026 | 40,293 | $-535,469 |
| 1.25 | 1.21 | 42.4% | -0.063 | 40,293 | $-1,386,783 |
| 1.50 | 1.01 | 45.2% | -0.093 | 40,293 | $-2,368,551 |

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
- **Best Net P&L (test, RF≥0.50):** 1.50 ATR | Net P&L=$4,116,273 | Trades=20,884

### Recommended (for scaling tables)

Scaling/projection tables use the **max net P&L** strategy at **RF≥0.50**.

- **Recommended stop:** 1.50 ATR
- **RF threshold:** 0.50
- **Win rate:** 66.4%
- **R:R:** 1.01:1
- **EV:** +0.334R
- **Net P&L (test period):** $4,116,273 across 20,884 trades
- **Estimated trades/year:** ~9,958
- **Estimated net P&L/year (100 shares):** $1,962,753

### Results by Year (Recommended Strategy, RF≥0.50)

| Year | Trades | Win Rate | EV (R) | Net P&L |
|---:|---:|---:|---:|---:|
| 2024 | 9,764 | 66.0% | +0.326 | $1,877,868 |
| 2025 | 10,150 | 66.8% | +0.342 | $2,051,097 |
| 2026 | 970 | 66.1% | +0.327 | $187,309 |

### Capital & Execution Assumptions

- **Capital per trade (notional):** $40,000 (100 shares × $400)
- **Costs per round trip:** commission+slippage = $3.00
- **Price assumption:** Projections use **AVG_ENTRY_PRICE=$400** for risk sizing and notional.

### Position Scaling (Recommended Strategy)

| Shares | Net P&L / Year | Notional / Trade |
|---:|---:|---:|
| 1 | $19,628 | $400 |
| 10 | $196,275 | $4,000 |
| 25 | $490,688 | $10,000 |
| 50 | $981,377 | $20,000 |
| 100 | $1,962,753 | $40,000 |
| 200 | $3,925,506 | $80,000 |
| 500 | $9,813,765 | $200,000 |

### Summary

Max-P&L selection tends to move toward wider stops because larger stop widths reduce stop-outs and increase win rate, even as R:R compresses. This choice is objective-dependent: max P&L is not the same as max EV(R) per trade.

### Next Steps

1. Confirm max-P&L stability via walk-forward resampling (to avoid overfitting stop width to one period).
2. Add drawdown/volatility stats so P&L can be compared on a risk-adjusted basis.
3. Re-estimate selection rate (trades per bar) from the actual test set and remove the static 0.432 shortcut.
4. Integrate explicit capital constraints (max concurrent trades / margin) into the P&L projection.

---

## 🔁 Walk-Forward Resampling (Optional)

Fixed stop (Option A): evaluate the same stop width across sequential yearly test folds.

| Train Years | Test Year | Test Samples | Trades | Win Rate | EV (R) | Net P&L |
|---:|---:|---:|---:|---:|---:|---:|
| 2016-2021 | 2022 | 78,762 | 37,309 | 67.2% | +0.349 | $7,695,111 |
| 2016-2022 | 2023 | 59,471 | 30,395 | 66.4% | +0.333 | $5,979,245 |
| 2016-2023 | 2024 | 40,293 | 20,993 | 66.3% | +0.332 | $4,115,601 |
| 2016-2024 | 2025 | 20,997 | 10,804 | 67.6% | +0.358 | $2,286,798 |

