# 📊 VWAP Reversion Strategy - Complete Analysis

**Generated**: 2026-02-08 10:19:32

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
| 0.25 | 6.05 | 25.6% | +0.802 | 17,403 | 56.8% | $338,575 |
| 0.35 | 4.32 | 33.7% | +0.793 | 17,453 | 56.7% | $490,199 |
| 0.40 | 3.78 | 37.0% | +0.770 | 17,955 | 55.4% | $565,855 |
| 0.50 | 3.02 | 43.2% | +0.740 | 18,131 | 55.0% | $696,995 |
| 0.60 | 2.52 | 48.3% | +0.702 | 18,177 | 54.9% | $802,501 |
| 0.75 | 2.02 | 54.0% | +0.628 | 18,472 | 54.2% | $919,537 |
| 1.00 | 1.51 | 60.9% | +0.529 | 18,712 | 53.6% | $1,051,778 |
| 1.25 | 1.21 | 64.8% | +0.433 | 19,257 | 52.2% | $1,109,022 |
| 1.50 | 1.01 | 66.4% | +0.334 | 20,884 | 48.2% | $1,107,447 |

### Baseline (No RF Filter, threshold=0.00)

| Stop (ATR) | R:R | Win Rate | EV (R) | Trades (test) | Net P&L (test) |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 6.05 | 15.3% | +0.079 | 40,293 | $-31,824 |
| 0.35 | 4.32 | 20.1% | +0.069 | 40,293 | $-11,780 |
| 0.40 | 3.78 | 22.3% | +0.067 | 40,293 | $421 |
| 0.50 | 3.02 | 26.2% | +0.056 | 40,293 | $5,008 |
| 0.60 | 2.52 | 29.6% | +0.041 | 40,293 | $-10,684 |
| 0.75 | 2.02 | 33.7% | +0.016 | 40,293 | $-68,209 |
| 1.00 | 1.51 | 38.8% | -0.026 | 40,293 | $-236,964 |
| 1.25 | 1.21 | 42.4% | -0.063 | 40,293 | $-475,332 |
| 1.50 | 1.01 | 45.2% | -0.093 | 40,293 | $-750,227 |

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
- **Best Net P&L (test, RF≥0.50):** 1.25 ATR | Net P&L=$1,109,022 | Trades=19,257

### Recommended (for scaling tables)

Scaling/projection tables use the **max net P&L** strategy at **RF≥0.50**.

- **Recommended stop:** 1.25 ATR
- **RF threshold:** 0.50
- **Win rate:** 64.8%
- **R:R:** 1.21:1
- **EV:** +0.433R
- **Net P&L (test period):** $1,109,022 across 19,257 trades
- **Estimated trades/year:** ~9,182
- **Estimated net P&L/year (100 shares):** $528,812

### Results by Year (Recommended Strategy, RF≥0.50)

| Year | Trades | Win Rate | EV (R) | Net P&L |
|---:|---:|---:|---:|---:|
| 2024 | 8,984 | 64.6% | +0.428 | $433,206 |
| 2025 | 9,390 | 65.0% | +0.436 | $815,139 |
| 2026 | 883 | 65.5% | +0.446 | $68,068 |

### Capital & Execution Assumptions

- **Capital per trade (notional):** $40,000 (100 shares × $400)
- **Costs per round trip:** commission+slippage = $3.00
- **Price assumption:** Projections use **AVG_ENTRY_PRICE=$400** for risk sizing and notional.

### Position Scaling (Recommended Strategy)

| Shares | Net P&L / Year | Notional / Trade |
|---:|---:|---:|
| 1 | $5,288 | $400 |
| 10 | $52,881 | $4,000 |
| 25 | $132,203 | $10,000 |
| 50 | $264,406 | $20,000 |
| 100 | $528,812 | $40,000 |
| 200 | $1,057,624 | $80,000 |
| 500 | $2,644,061 | $200,000 |

### Summary

Max-P&L selection tends to move toward wider stops because larger stop widths reduce stop-outs and increase win rate, even as R:R compresses. This choice is objective-dependent: max P&L is not the same as max EV(R) per trade.

### Next Steps

1. Confirm max-P&L stability via walk-forward resampling (to avoid overfitting stop width to one period).
2. Add drawdown/volatility stats so P&L can be compared on a risk-adjusted basis.
3. Re-estimate selection rate (trades per bar) from the actual test set and remove the static 0.432 shortcut.
4. Integrate explicit capital constraints (max concurrent trades / margin) into the P&L projection.

