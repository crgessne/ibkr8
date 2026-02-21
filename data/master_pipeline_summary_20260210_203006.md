# RF VWAP Reversion Strategy - Analysis Results

**Generated**: 2026-02-10 20:30:06

**Dataset**: tsla_5min_10years.csv
**Test Period Definition**: year >= 2024 (aggregate across all years >= TEST_YEAR)
**Win Definition**: realized_net_pnl
**P&L Definition**: realized_path

**Capital Constraint**: $1,000,000 (realistic capital-constrained simulation)
**Features Used**: 18
**Position Size**: 100 shares
**Commission**: $0.005/share
**Slippage**: $0.01/share

## Top Strategies (RF >= 0.50, Capital-Constrained @ $1M)

| Stop ATR | R:R | WR(selected) | WR(label) | EV | Signals | Executed | % Exec | Net P&L | Return % | Max Pos |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 1.51 | 59.5% | 60.9% | +0.495 | 18,712 | 18,712 | 100.0% | $-140,497 | -14.0% | 19 |
| 0.75 | 2.02 | 52.7% | 54.0% | +0.589 | 18,472 | 18,472 | 100.0% | $-150,497 | -15.0% | 16 |
| 1.25 | 1.21 | 63.6% | 64.8% | +0.405 | 19,257 | 19,257 | 100.0% | $-157,880 | -15.8% | 22 |
| 0.60 | 2.52 | 47.1% | 48.3% | +0.657 | 18,177 | 18,177 | 100.0% | $-159,412 | -15.9% | 14 |
| 0.50 | 3.02 | 42.2% | 43.2% | +0.697 | 18,131 | 18,131 | 100.0% | $-162,778 | -16.3% | 12 |
| 0.25 | 6.05 | 25.0% | 25.6% | +0.759 | 17,403 | 17,403 | 100.0% | $-166,785 | -16.7% | 8 |
| 0.35 | 4.32 | 32.8% | 33.7% | +0.747 | 17,453 | 17,453 | 100.0% | $-173,918 | -17.4% | 9 |
| 0.40 | 3.78 | 36.1% | 37.0% | +0.727 | 17,955 | 17,955 | 100.0% | $-174,015 | -17.4% | 10 |
| 1.50 | 1.01 | 65.5% | 66.4% | +0.316 | 20,884 | 20,883 | 100.0% | $-186,137 | -18.6% | 28 |

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

