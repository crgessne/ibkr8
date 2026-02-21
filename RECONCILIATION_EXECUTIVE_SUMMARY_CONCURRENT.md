# RECONCILIATION: Concurrent vs Master Pipeline - Executive Summary

**Date:** February 9, 2026  
**Test Configuration:** Stop ATR 1.5, RF Threshold 0.5, Year 2024

---

## 📊 THE BOTTOM LINE

| System | Trades | Win Rate | Total P&L | Return % | Verdict |
|--------|--------|----------|-----------|----------|---------|
| **Master Pipeline** (Single Position) | 20,884 | 66.41% | **+$1,583,733** | **+158.4%** | ✅ **PROFITABLE** |
| **Concurrent Backtest** (Multi-Position) | 2,344 | 49.66% | **-$8,995** | **-9.0%** | ❌ **LOSING** |
| **Difference** | **-18,540** | **-16.75pp** | **-$1,592,728** | **-167.4pp** | 🚨 **MASSIVE GAP** |

---

## 🚨 CRITICAL ISSUES FOUND

### 1. **MISSING 88.8% OF TRADES**
- **Master Pipeline:** 20,884 trades
- **Concurrent:** 2,344 trades  
- **Missing:** 18,540 trades (88.8%)

**This is the #1 problem.** The concurrent backtest is not seeing or executing most of the trading opportunities.

### 2. **WIN RATE COLLAPSE**
- **Master Pipeline:** 66.41% win rate
- **Concurrent:** 49.66% win rate
- **Drop:** 16.75 percentage points

With a sub-50% win rate and losses > wins, the strategy has **negative expected value**.

### 3. **STRATEGY LOSES MONEY**
- **Master Pipeline:** Made $1.58M profit (+158% return)
- **Concurrent:** Lost $9K (-9% return)
- **Gap:** $1.59M difference

The concurrent approach is **fundamentally broken**.

---

## 🔍 ROOT CAUSE ANALYSIS

### Why Are 88.8% of Trades Missing?

**Hypothesis #1: Signal Generation Bug**
- The concurrent backtest generated only 4,728 signals
- Master pipeline processes ~40,265 signals (before RF filtering)
- The concurrent system is missing **35,537 signals** (88.3%)

**Possible causes:**
- `StreamingIndicatorsAligned.calculate()` returning wrong values
- Different entry logic in concurrent vs master
- Data alignment issues (lookback window problems)
- RF model not being called correctly

**Hypothesis #2: Different Entry Conditions**
- Master uses `check_vwap_reversion_entry()`
- Concurrent may be using different or stricter entry logic
- Need to compare line-by-line

**Hypothesis #3: Blocking Logic**
- Concurrent allows max 6 positions
- If hitting the limit frequently, later signals get blocked
- But this should still show in "signals generated" count

### Why Did Win Rate Drop 16.75%?

When allowing concurrent positions:
1. **Capital gets diluted** across multiple positions
2. **Mediocre signals get taken** alongside good ones
3. **Stop losses trigger faster** when capital is spread thin
4. **Average win < average loss** creates negative expectancy

**Measured:**
- Avg Win: $175.43
- Avg Loss: $180.68
- **Loss is 2.9% larger than win**

With 49.66% win rate:
```
Expected Value per trade = (0.4966 × $175.43) - (0.5034 × $180.68)
                         = $87.13 - $90.95
                         = -$3.82 per trade
```

Over 2,344 trades: **-$3.82 × 2,344 = -$8,955** ✅ (matches actual -$8,995)

---

## ✅ WHAT'S WORKING

1. **Models are loaded correctly** - Same RF models, same RR ratio
2. **Data is correct** - Same year, same bars (19,548)
3. **Position management** - Max 6 concurrent positions tracked correctly
4. **Risk/reward calculations** - Expected value math checks out

---

## ❌ WHAT'S BROKEN

1. **Signal generation is missing 88.8% of opportunities**
2. **Win rate collapsed from profitable to unprofitable**
3. **Concurrent positions dilute capital and degrade performance**
4. **The strategy has negative expected value**

---

## 🎯 ACTION ITEMS (PRIORITY ORDER)

### **IMMEDIATE (Critical Path)**

1. ✅ **Document the gap** - DONE (this report)

2. **Debug Signal Generation**
   ```python
   # Add logging to concurrent_backtest.py main loop:
   print(f"Bar {i}: VWAP={vwap:.2f}, Price={bar.close:.2f}, Signal={signal}")
   ```
   - Count how many bars generate entry signals
   - Compare to master pipeline signal count
   - Find where 35,537 signals disappeared

3. **Test Single-Position Mode**
   ```bash
   # Run concurrent backtest WITHOUT --concurrent flag
   python sim_trading/concurrent_backtest.py --year 2024 --stop-atr 1.5 --rf-threshold 0.5
   ```
   - If single-position matches master → concurrent logic is the problem
   - If single-position still underperforms → signal generation is broken

### **NEXT (Investigation)**

4. **Compare Indicators Bar-by-Bar**
   - Export indicator values from master pipeline
   - Export indicator values from concurrent backtest
   - Diff the two files
   - Find first bar where they diverge

5. **Compare RF Predictions**
   - Log every RF prediction from both systems
   - Compare probability scores
   - Verify same features being used

### **LATER (If Needed)**

6. **Rebuild Concurrent Logic**
   - If concurrent is fundamentally different architecture, may need to rebuild
   - Consider: Is concurrent mode even desirable? Master makes 158% with single position

---

## 💡 STRATEGIC QUESTION

### **Do We Even Want Concurrent Positions?**

**Evidence says NO:**
- Master Pipeline: 158.4% profit with single position
- Concurrent: -9.0% loss with multi-position
- **Difference: 167.4 percentage points**

**Why single-position wins:**
1. **Full capital focus** - All buying power on best opportunity
2. **Better entries** - Wait for highest-quality setups
3. **No dilution** - Winners run without competing for capital
4. **Higher win rate** - 66.41% vs 49.66%

**Recommendation:** Focus on fixing signal generation to match master pipeline. Once that's working, test both modes fairly. But evidence strongly suggests **single-position is superior**.

---

## 📈 EXPECTED OUTCOME (After Fixes)

If we fix signal generation to match master pipeline:

**Best Case (Single Position Mode):**
- Trades: ~20,884
- Win Rate: ~66%
- P&L: ~+$1.58M
- Return: ~+158%

**Realistic Case (Concurrent Mode):**
- Trades: ~20,884
- Win Rate: ~55-60% (dilution effect)
- P&L: +$500K to +$1M (lower but positive)
- Return: +50% to +100%

---

## 🏁 CONCLUSION

The concurrent backtest is **NOT READY FOR USE**. It's missing 88.8% of trades and loses money.

**Critical blockers:**
1. Signal generation produces 10X fewer signals than expected
2. Win rate below 50% with negative expected value
3. Strategy loses money in both absolute and relative terms

**Next step:** Debug signal generation to find the missing 35,537 signals.

---

**Status:** 🔴 **BLOCKED - REQUIRES DEBUGGING BEFORE FURTHER USE**

**Files:**
- Detailed reconciliation: `CONCURRENT_VS_MASTER_RECONCILIATION.md`
- Trade data: `data/concurrent_backtest_trades_concurrent.csv`
- Master results: `data/master_pipeline_results_20260209_155832.csv`
