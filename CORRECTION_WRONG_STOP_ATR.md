# CRITICAL CORRECTION: Wrong Stop ATR Comparison
**Date**: February 10, 2026  
**Status**: 🔴 ANALYSIS ERROR DETECTED

---

## 🚨 THE MISTAKE

We were comparing:
- **Master**: Stop ATR = **1.25**, RF threshold = 0.5 → 8,984 trades, 64.6% win rate
- **Concurrent**: Stop ATR = **1.5**, RF threshold = 0.5 → 4,728 trades, 47.9% win rate

**This is comparing DIFFERENT strategies!**

---

## 📊 CORRECT COMPARISON NEEDED

From master_pipeline_summary_20260208_120424.md:

### Master Pipeline 2024 Results (Stop ATR = 1.25, RF ≥ 0.50):
```
Year: 2024
Trades: 8,984
Win Rate: 64.6%
EV (R): +0.428
R:R: 1.21:1
Net P&L: $550,612
```

### Our Concurrent Test (Stop ATR = 1.5, RF ≥ 0.50):
```
Year: 2024
Trades: 4,728
Win Rate: 47.9%
R:R: 1.01:1
Net P&L: -$36,732
```

**We tested the WRONG configuration!**

---

## 🔧 CORRECTIVE ACTION

### Need to Run:
```bash
python sim_trading/concurrent_backtest.py \
    --year 2024 \
    --stop-atr 1.25 \  # ← CHANGE FROM 1.5 to 1.25
    --rf-threshold 0.5 \
    --capital 1000000 \
    --concurrent
```

### Expected Results:
- Trades: ~8,984 (match master)
- Win Rate: ~64.6% (match master)
- R:R: ~1.21 (match master)
- P&L: ~$550,000+ (match master)

---

## 💡 WHY THIS MATTERS

### Stop ATR = 1.25 vs 1.50
**1.25 ATR stop** is **tighter** than 1.5 ATR:
- Smaller stop loss distance
- Stops hit more easily
- BUT: Higher R:R ratio (1.21 vs 1.01)
- Result: Can be profitable at lower win rates

**Example with $3 ATR**:
```
Stop ATR = 1.25:
- Stop distance: 1.25 × $3 = $3.75
- Target distance: 1.21 × $3.75 = $4.54
- R:R = 1.21:1

Stop ATR = 1.5:
- Stop distance: 1.5 × $3 = $4.50
- Target distance: 1.01 × $4.50 = $4.55
- R:R = 1.01:1
```

Wider stops (1.5) have nearly 1:1 R:R, which requires 50% win rate.  
Tighter stops (1.25) have 1.21:1 R:R, which requires only 45% win rate!

---

## 🎯 ACTION PLAN

1. ✅ Identify the error (DONE)
2. ⏳ Run concurrent with --stop-atr 1.25
3. ⏳ Compare results with master's 8,984 trades
4. ⏳ Verify win rate matches 64.6%
5. ⏳ Verify P&L matches ~$550K

---

**Status**: Ready to run correct test.  
**ETA**: 2 minutes to run, then we can do proper comparison.
