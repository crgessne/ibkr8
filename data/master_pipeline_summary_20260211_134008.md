# RF VWAP Reversion Strategy - Analysis Results

**Generated**: 2026-02-11 13:40:08

**Dataset**: tsla_5min_10years.csv
**Test Period Definition**: year >= 2024 (aggregate across all years >= TEST_YEAR)
**Win Definition**: label
**P&L Definition**: label_rr

**Capital Constraint**: $1,000,000 (realistic capital-constrained simulation)
**Features Used**: 18
**Position Size**: 100 shares
**Commission**: $0.005/share
**Slippage**: $0.01/share

## Top Strategies (RF >= 0.50, Capital-Constrained @ $1M)

| Stop ATR | R:R | WR(selected) | WR(label) | EV | Signals | Executed | % Exec | Net P&L | Return % | Max Pos |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 1.51 | 60.9% | 60.9% | +0.529 | 18,712 | 18,712 | 100.0% | $-159,096 | -15.9% | 0 |
| 0.75 | 2.02 | 54.0% | 54.0% | +0.628 | 18,472 | 18,472 | 100.0% | $-159,436 | -15.9% | 0 |
| 0.60 | 2.52 | 48.3% | 48.3% | +0.702 | 18,177 | 18,177 | 100.0% | $-164,608 | -16.5% | 0 |
| 0.50 | 3.02 | 43.2% | 43.2% | +0.740 | 18,131 | 18,131 | 100.0% | $-166,489 | -16.6% | 0 |
| 0.25 | 6.05 | 25.6% | 25.6% | +0.802 | 17,403 | 17,403 | 100.0% | $-168,171 | -16.8% | 0 |
| 0.35 | 4.32 | 33.7% | 33.7% | +0.793 | 17,453 | 17,453 | 100.0% | $-175,500 | -17.5% | 0 |
| 0.40 | 3.78 | 37.0% | 37.0% | +0.770 | 17,955 | 17,955 | 100.0% | $-176,543 | -17.7% | 0 |
| 1.25 | 1.21 | 64.8% | 64.8% | +0.433 | 19,257 | 19,257 | 100.0% | $-195,697 | -19.6% | 0 |
| 1.50 | 1.01 | 66.4% | 66.4% | +0.334 | 20,884 | 20,884 | 100.0% | $-272,008 | -27.2% | 0 |

## Top Features

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

