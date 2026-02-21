# 📊 VWAP Reversion Strategy - Complete Analysis

**Generated**: 2026-02-08 11:01:39

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
| 0.25 | 6.05 | 25.6% | +0.802 | 17,403 | 56.8% | $528,391 |
| 0.35 | 4.32 | 33.7% | +0.793 | 17,453 | 56.7% | $739,607 |
| 0.40 | 3.78 | 37.0% | +0.770 | 17,955 | 55.4% | $849,422 |
| 0.50 | 3.02 | 43.2% | +0.740 | 18,131 | 55.0% | $1,035,702 |
| 0.60 | 2.52 | 48.3% | +0.702 | 18,177 | 54.9% | $1,168,354 |
| 0.75 | 2.02 | 54.0% | +0.628 | 18,472 | 54.2% | $1,326,815 |
| 1.00 | 1.51 | 60.9% | +0.529 | 18,712 | 53.6% | $1,507,371 |
| 1.25 | 1.21 | 64.8% | +0.433 | 19,257 | 52.2% | $1,582,482 |
| 1.50 | 1.01 | 66.4% | +0.334 | 20,884 | 48.2% | $1,583,733 |

### Baseline (No RF Filter, threshold=0.00)

| Stop (ATR) | R:R | Win Rate | EV (R) | Trades (test) | Net P&L (test) |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 6.05 | 15.3% | +0.079 | 40,293 | $158,541 |
| 0.35 | 4.32 | 20.1% | +0.069 | 40,293 | $232,879 |
| 0.40 | 3.78 | 22.3% | +0.067 | 40,293 | $277,297 |
| 0.50 | 3.02 | 26.2% | +0.056 | 40,293 | $335,000 |
| 0.60 | 2.52 | 29.6% | +0.041 | 40,293 | $352,164 |
| 0.75 | 2.02 | 33.7% | +0.016 | 40,293 | $342,356 |
| 1.00 | 1.51 | 38.8% | -0.026 | 40,293 | $226,987 |
| 1.25 | 1.21 | 42.4% | -0.063 | 40,293 | $15,165 |
| 1.50 | 1.01 | 45.2% | -0.093 | 40,293 | $-242,427 |

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

