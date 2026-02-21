# RF VWAP Reversion Strategy - Analysis Results

**Generated**: 2026-02-11 17:13:01

**Dataset**: tsla_5min_10years.csv
**Test Period Definition**: year >= 2024 (aggregate across all years >= TEST_YEAR)
**Win Definition**: label
**P&L Definition**: label_rr

**Capital Constraint**: $1,000,000
**Features Used**: 18
**Position Size**: 100 shares
**Commission**: $0.005/share
**Slippage**: $0.01/share
**Min R:R Filter**: none

## Top Strategies (RF >= 0.50, Capital-Constrained)

| Stop ATR | R:R | WR(selected) | WR(label) | EV | Signals | Executed | % Exec | Net P&L | Return % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 6.05 | 21.3% | 21.3% | +0.504 | 22,962 | 22,962 | 100.0% | $-180,737 | -18.1% |
| 0.40 | 3.78 | 29.7% | 29.7% | +0.419 | 23,278 | 23,278 | 100.0% | $-212,427 | -21.2% |
| 0.35 | 4.32 | 27.1% | 27.1% | +0.440 | 23,230 | 23,230 | 100.0% | $-213,106 | -21.3% |
| 0.50 | 3.02 | 33.7% | 33.7% | +0.356 | 23,802 | 23,802 | 100.0% | $-223,273 | -22.3% |
| 0.60 | 2.52 | 36.8% | 36.8% | +0.297 | 24,084 | 24,084 | 100.0% | $-240,154 | -24.0% |
| 0.75 | 2.02 | 39.5% | 39.5% | +0.191 | 25,516 | 25,516 | 100.0% | $-292,976 | -29.3% |
| 1.00 | 1.51 | 44.0% | 44.0% | +0.105 | 25,173 | 25,173 | 100.0% | $-362,043 | -36.2% |
| 1.25 | 1.21 | 46.7% | 46.7% | +0.032 | 24,974 | 24,974 | 100.0% | $-473,183 | -47.3% |
| 1.50 | 1.01 | 48.4% | 48.4% | -0.028 | 24,599 | 24,599 | 100.0% | $-611,815 | -61.2% |

## Top Features

1. `vwap_width_atr`
2. `price_to_vwap_atr`
3. `bars_from_vwap`
4. `wap`
5. `bar_count`
6. `vwap_slope`
7. `rel_vol`
8. `vwap_slope_5`
9. `vol_at_extension`
10. `bar_range_atr`
11. `vwap_helping`
12. `close_position`
13. `crossed_vwap`
14. `rsi`
15. `vol_ratio`