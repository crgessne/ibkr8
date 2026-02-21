# VWAP Reversion Strategy - Expected Value Analysis

**Date:** February 6, 2026  
**Data:** TSLA 5-min bars, 10 years (2015-2026)  
**Train:** 2015-2023, **Test:** 2024-2026

## Key Finding: Positive EV Setups Exist

The critical insight: **R:R matters more than raw win rate**. At 2:1 R:R, you only need 33% win rate to break even.

---

## +EV Setups Found (Test Data 2024-2026)

| Zone (ATR) | Stop (ATR) | R:R | Breakeven WR | Raw WR | Raw EV | RF≥0.5 WR | RF≥0.5 EV | N Trades |
|------------|------------|-----|--------------|--------|--------|-----------|-----------|----------|
| **0.5-1.0** | **0.35** | **2.14:1** | 31.8% | 34.1% | **+0.072R** | 37.1% | **+0.166R** | 2,043 |
| **0.5-1.0** | **0.40** | **1.88:1** | 34.8% | 36.6% | **+0.053R** | 40.1% | **+0.151R** | 1,940 |
| **0.5-1.0** | **0.50** | **1.50:1** | 40.0% | 41.7% | **+0.043R** | 44.4% | **+0.109R** | 2,432 |
| 1.0-1.5 | 0.60 | 2.08:1 | 32.4% | 30.4% | -0.062R | 32.9% | **+0.015R** | 2,187 |

### Best Setup
- **Zone:** 0.5-1.0 ATR from VWAP
- **Stop:** 0.35-0.40 ATR  
- **R:R:** ~2:1
- **RF Threshold:** ≥0.5
- **Expected EV:** +0.15 to +0.17R per trade
- **Sample Size:** ~2,000 trades in 2-year test period

---

## Negative EV Setups (Avoid)

| Zone (ATR) | Stop (ATR) | R:R | Raw WR | Raw EV | RF≥0.5 WR | RF≥0.5 EV |
|------------|------------|-----|--------|--------|-----------|-----------|
| 1.5-2.0 | 2.0 | 0.88:1 | 28.6% | -0.464R | 33.3% | -0.376R |
| 1.5-2.0 | 1.0 | 1.75:1 | 24.4% | -0.330R | 28.9% | -0.206R |
| 2.0-2.5 | 1.0 | 2.25:1 | 15.3% | -0.504R | 23.0% | -0.252R |
| 2.5-3.0 | 1.0 | 2.75:1 | 8.0% | -0.698R | 12.2% | -0.541R |

**Why these fail:** Wider stops destroy R:R; farther zones have very low base win rates.

---

## Top RF Features (0.5-1.0 ATR Zone, 0.4 ATR Stop)

| Feature | Importance | Description |
|---------|------------|-------------|
| vol_at_extension | 0.056 | Relative volume at current bar |
| rel_vol | 0.054 | Current volume vs 20-bar average |
| wap | 0.050 | Weighted average price (IBKR) |
| bar_range_atr | 0.034 | Current bar range / ATR |
| dist_to_bb_upper | 0.033 | Distance to upper Bollinger Band |
| vwap_slope | 0.033 | VWAP slope (linreg) |
| vol_trend_3 | 0.032 | 3-bar volume trend |
| rsi | 0.030 | RSI(14) |
| rsi_extremity | 0.029 | |RSI - 50| |
| momentum_divergence_5 | 0.029 | Price/RSI divergence |

**Key insight:** Volume features dominate - `vol_at_extension` and `rel_vol` are top predictors.

---

## Trade Logic Summary

**Entry Criteria:**
1. Price is 0.5-1.0 ATR from VWAP
2. RF probability ≥ 0.5

**Trade Setup:**
- **Long:** Price below VWAP → Buy, target VWAP, stop = entry - 0.4*ATR
- **Short:** Price above VWAP → Sell, target VWAP, stop = entry + 0.4*ATR

**Win Condition:** Price touches VWAP before stop hit, before EOD

**Exit:** At VWAP (target) or stop

---

## Important Notes

1. **vwap_width_atr EXCLUDED from RF features** - prevents RF from learning "closer = better"
2. **Labels computed dynamically** based on stop width parameter
3. **Test data is out-of-sample** (2024-2026, trained on 2015-2023)
4. **R:R calculated as:** (zone midpoint) / (stop width in ATR)
5. **EV formula:** `WR × R - (1 - WR)`

---

## Script Reference

```
python rf_dynamic_label.py --stop_atr 0.4 --zone_min 0.5 --zone_max 1.0
```

Arguments:
- `--stop_atr`: Stop width in ATR units (default: 2.0)
- `--zone_min`: Minimum VWAP distance in ATR (default: 1.5)
- `--zone_max`: Maximum VWAP distance in ATR (default: 2.0)
- `--test_year`: First year of test set (default: 2024)
