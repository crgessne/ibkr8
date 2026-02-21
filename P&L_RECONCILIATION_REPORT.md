# P&L Reconciliation Report
## Master Pipeline vs Streaming Simulator

**Date**: February 9, 2026  
**Analysis**: Comparing theoretical (vectorized) vs realistic (streaming) P&L

---

## Executive Summary

| Metric | Master Pipeline | Streaming Simulator | Ratio |
|--------|----------------|---------------------|-------|
| **Net P&L** | **+$1,583,733** | **-$18,870** | **83.9x** |
| **Trades** | 20,884 | 724 | 28.8x |
| **Win Rate** | 66.4% | 47.8% | 1.4x |
| **Profit Factor** | N/A | 0.81 (losing) | N/A |

**Status**: ⚠️ **CRITICAL DISCREPANCY** - Streaming simulator achieving only **1.2%** of theoretical P&L

---

## Root Cause Analysis

### 1. **Fundamental Methodology Differences**

#### Master Pipeline (Theoretical)
```python
# Counts EVERY bar where RF probability >= 0.5
# Assumes perfect execution at ideal prices
Win P&L  = +risk × R:R ratio
Loss P&L = -risk
Risk = stop_atr × ATR × 100 shares
```

**Assumptions**:
- ✓ Perfect entry at VWAP reversion point
- ✓ Stop/target always hit (never in between)
- ✓ No position sizing constraints
- ✓ Can take every signal simultaneously
- ✓ Fixed R:R ratio per trade

#### Streaming Simulator (Realistic)
```python
# Simulates bar-by-bar execution
# Only one position at a time
P&L = (exit_price - entry_price) × shares - fees
```

**Reality Factors**:
- ✗ Entry/exit at actual bar prices (not ideal VWAP)
- ✗ Can only hold one position at a time
- ✗ Stops may be hit before target
- ✗ Market gaps and slippage
- ✗ Variable holding periods

---

## 2. **Critical Issues Identified**

### Issue A: Trade Count Mismatch (20,884 vs 724)

**Master Pipeline**: 20,884 "trades"
- Counts every bar with RF≥0.5 as a potential trade
- No exclusion for existing positions
- Assumes infinite capital to take all signals

**Streaming Simulator**: 724 actual trades
- **Only 3.5% of master pipeline signals were traded**
- Can only enter when no position exists
- Must wait for exit before next entry

**Impact**: Master pipeline overstates opportunity by **28.8x**

---

### Issue B: Win Rate Discrepancy (66.4% vs 47.8%)

**Theoretical (Master)**: 66.4% win rate
- Based on label outcomes (perfect hindsight)
- Assumes perfect entry/exit at VWAP ± target

**Actual (Streaming)**: 47.8% win rate
- Real entry prices (bar open/close)
- Stops hit more frequently
- Market noise and slippage reduce edge

**Impact**: Actual win rate is **18.6 percentage points lower**

---

### Issue C: P&L Calculation Method

**Master Pipeline**:
```python
# Fixed R:R-based calculation
if label == 1:  # Win
    pnl = stop_atr * atr * 100 * rr  # ~$229 risk × 1.01 R:R
else:  # Loss
    pnl = -stop_atr * atr * 100      # -$229 risk

# For 1.5 ATR stop:
# Avg win: ~$231
# Avg loss: -$229
# With 66.4% WR: +$75.83 per trade
```

**Streaming Simulator**:
```python
# Actual price-based calculation
pnl = (exit_price - entry_price) * shares - fees

# Observed:
# Avg win: $221.23
# Avg loss: -$250.50
# With 47.8% WR: -$26.06 per trade
```

**Key Difference**: 
- Master assumes symmetric wins/losses
- Reality: Losses are LARGER than wins ($250 vs $221)
- This indicates **stops are being hit harder than targets**

---

## 3. **Why Streaming Simulator Is Losing Money**

### Problem 1: Asymmetric Risk/Reward
```
Master assumes:    Win $231 / Loss -$229  (balanced)
Actual reality:    Win $221 / Loss -$250  (13% worse on losses)
```

**Cause**: Market moves against position faster than toward target

### Problem 2: Insufficient Win Rate
```
Breakeven WR needed with actual R:R:
  Required WR = Loss / (Win + Loss) 
              = 250 / (221 + 250) 
              = 53.1%
  
  Actual WR = 47.8%
  Shortfall = 5.3 percentage points
```

### Problem 3: Position Utilization
```
Signal opportunities: 20,884
Trades executed:         724 (3.5%)
Missed opportunities: 20,160 (96.5%)
```

**Cause**: Single position constraint + long holding periods

### Problem 4: Stop Placement
```
1.5 ATR stops are TOO WIDE:
- Ties up capital longer
- Allows larger losses
- Reduces number of trades
- Increases drawdown before exit
```

---

## 4. **Recommended Fixes**

### Fix 1: Tighten Stops (Immediate)
```python
# Current: 1.5 ATR stop → 66.4% WR (theoretical)
# Reality: 1.5 ATR → 47.8% WR (losing)

# Test tighter stops:
STOP_ATR = 0.75  # Master: 54% WR, +$1.3M P&L
# Expected streaming: ~45-48% WR (closer to breakeven)
```

### Fix 2: Add Profit Taking Logic
```python
def should_exit_profit(position, current_price, indicators):
    """Exit early if profit target partially hit"""
    unrealized = (current_price - position.entry_price) * position.quantity
    risk = position.stop_atr * indicators['atr'] * position.quantity
    
    # Take profit at 0.5R (50% of target)
    if unrealized >= risk * 0.5:
        return True
    return False
```

### Fix 3: Dynamic Position Sizing
```python
# Instead of fixed 100 shares, size by risk:
RISK_PER_TRADE = 500  # $500 risk per trade
shares = int(RISK_PER_TRADE / (stop_atr * atr))
shares = min(shares, 200)  # Max 200 shares
```

### Fix 4: Better Entry Filtering
```python
# Add momentum confirmation
def should_enter(signals, indicators):
    if not signals['rf_signal']:
        return False
    
    # Only enter if RSI confirms
    if indicators['is_long_setup']:
        return indicators['rsi'] < 40  # Oversold
    else:
        return indicators['rsi'] > 60  # Overbought
```

### Fix 5: Trailing Stop
```python
def update_trailing_stop(position, current_price, indicators):
    """Move stop to breakeven after 0.5R profit"""
    unrealized = (current_price - position.entry_price) * position.quantity
    risk = position.stop_atr * indicators['atr'] * position.quantity
    
    if unrealized >= risk * 0.5:
        # Move stop to breakeven
        position.stop_price = position.entry_price
```

---

## 5. **Immediate Action Plan**

### Phase 1: Diagnostic (Next)
```bash
# 1. Run streaming sim with multiple stop widths
python scripts/streaming_comparison.py --stops 0.5,0.75,1.0,1.25,1.5

# 2. Analyze actual entry/exit prices vs VWAP
python scripts/analyze_execution.py

# 3. Check if stops are being hit prematurely
python scripts/stop_analysis.py
```

### Phase 2: Strategy Improvements (This Week)
1. ✓ Implement profit-taking at 0.5R and 1.0R
2. ✓ Add trailing stop to breakeven after 0.5R
3. ✓ Test tighter stops (0.75 ATR target)
4. ✓ Add momentum confirmation filter
5. ✓ Implement dynamic position sizing

### Phase 3: Validation (Next Week)
1. Re-run streaming simulation with improvements
2. Target: Achieve 10-20% of theoretical P&L
3. Minimum acceptable: Positive P&L with Sharpe > 1.0

---

## 6. **Expected Outcomes After Fixes**

### Conservative Estimate
```
Target streaming performance:
- Win Rate: 52-55% (vs current 47.8%)
- Profit Factor: 1.3-1.5 (vs current 0.81)
- Net P&L: $50K-$150K (vs current -$19K)
- Sharpe Ratio: 1.5-2.0 (vs current -1.94)
- % of theoretical: 10-15% (vs current 1.2%)
```

### Optimistic Estimate (with all improvements)
```
- Win Rate: 56-58%
- Profit Factor: 1.8-2.2
- Net P&L: $200K-$400K
- Sharpe Ratio: 2.5-3.0
- % of theoretical: 20-25%
```

---

## 7. **Key Insights**

### ✅ What Master Pipeline Is Good For
- Strategy selection (which stop width?)
- Parameter optimization (RF threshold, features)
- Understanding theoretical edge (R:R analysis)
- Quick iteration on logic

### ✅ What Streaming Simulator Is Good For
- Realistic P&L expectations
- Risk management testing
- Position sizing optimization
- Live trading preparation
- Drawdown analysis

### ⚠️ Critical Understanding
```
Master Pipeline = "What's possible with perfect execution"
Streaming Sim   = "What's achievable in real trading"

Gap between them = Implementation skill + Market reality
```

---

## 8. **Next Steps**

1. **Immediate** (Today):
   - Create `streaming_comparison.py` to test multiple stops
   - Run with 0.5, 0.75, 1.0 ATR stops
   - Compare to master pipeline expectations

2. **Short-term** (This Week):
   - Implement profit-taking logic
   - Add trailing stops
   - Test momentum filters

3. **Medium-term** (Next 2 Weeks):
   - Optimize entry/exit logic
   - Implement dynamic position sizing
   - Achieve positive P&L in streaming

4. **Long-term** (Next Month):
   - Paper trade with Interactive Brokers
   - Validate assumptions with real data
   - Refine based on actual fills

---

## Conclusion

The 83.9x discrepancy between master pipeline and streaming simulator is **NOT a bug** - it reveals the **gap between theory and practice** in algorithmic trading.

**Current Status**: Strategy has theoretical edge but poor real-world execution  
**Root Cause**: Too-wide stops + insufficient win rate + no profit-taking  
**Solution**: Tighten stops, add profit-taking, improve entry filtering  
**Goal**: Achieve 10-20% of theoretical P&L with positive Sharpe ratio  

---

**Report Generated**: 2026-02-09  
**Analysis By**: Master Pipeline vs Streaming Simulator Comparison  
**Status**: Ready for Phase 1 Diagnostic Testing
