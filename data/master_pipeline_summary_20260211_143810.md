# RF VWAP Reversion Strategy - Analysis Results

**Generated**: 2026-02-11 14:38:10

**Dataset**: tsla_5min_10years.csv
**Test Period Definition**: year >= 2024 (aggregate across all years >= TEST_YEAR)
**Win Definition**: label
**P&L Definition**: label_rr

**Capital Constraint**: $1,000,000 (realistic capital-constrained simulation)
**Features Used**: 18
**Position Size**: 100 shares
**Commission**: $0.005/share
**Slippage**: $0.01/share
**Min R:R Filter**: 2.00 (reject trades with vwap_dist/stop < 2.00)

## Top Strategies (RF >= 0.50, Capital-Constrained @ $1M)

| Stop ATR | R:R | WR(selected) | WR(label) | EV | Signals | Executed | % Exec | Net P&L | Return % | Max Pos |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.60 | 2.52 | 30.7% | 30.7% | +0.082 | 1,714 | 1,714 | 100.0% | $-6,740 | -0.7% | 0 |
| 0.50 | 3.02 | 28.8% | 28.8% | +0.160 | 4,093 | 4,093 | 100.0% | $-18,132 | -1.8% | 0 |
| 0.35 | 4.32 | 25.2% | 25.2% | +0.338 | 7,501 | 7,501 | 100.0% | $-34,518 | -3.5% | 0 |
| 0.40 | 3.78 | 25.9% | 25.9% | +0.236 | 6,574 | 6,574 | 100.0% | $-40,316 | -4.0% | 0 |
| 0.25 | 6.05 | 20.7% | 20.7% | +0.458 | 10,314 | 10,314 | 100.0% | $-55,669 | -5.6% | 0 |

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

