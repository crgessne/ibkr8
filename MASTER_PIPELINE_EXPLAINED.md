# Master Pipeline P&L Calculation - Deep Dive

## How Master Pipeline Calculates P&L

### TL;DR
**Master pipeline does NOT simulate actual trading.** It calculates **theoretical P&L** based on label outcomes using fixed risk/reward ratios.

---

## The Key Question: "Is this using actual stock price to trade?"

### Short Answer: **NO** ❌

The master pipeline:
- ✅ Uses actual stock prices to calculate indicators (VWAP, ATR, etc.)
- ✅ Uses actual stock prices in the label generation (to determine win/loss)
- ❌ Does NOT simulate buying/selling at those prices
- ❌ Does NOT track entry/exit prices for P&L
- ❌ Does NOT simulate order execution

### What It Actually Does

```python
# Master Pipeline P&L Calculation (from calculate_dollar_pnl function)

def calculate_dollar_pnl(y_actual, stop_atr, rr, atr_series, entry_price_series, slippage_per_share=0.02):
    """
    This function calculates THEORETICAL P&L based on:
    1. Label outcomes (y_actual = 1 for win, 0 for loss)
    2. Fixed risk/reward math
    """
    
    # For a 1.5 ATR stop with 1.01 R:R ratio:
    
    # THEORETICAL CALCULATION:
    risk_per_trade = stop_atr * atr_series * SHARES_PER_TRADE
    # Example: 1.5 × $10 ATR × 100 shares = $1,500 risk
    
    reward_per_trade = risk_per_trade * rr
    # Example: $1,500 × 1.01 = $1,515 reward
    
    # P&L based on label outcome (NOT actual price movement):
    if label == 1:  # Win
        pnl = +$1,515 (reward)
    else:  # Loss
        pnl = -$1,500 (risk)
    
    # Subtract costs
    net_pnl = pnl - commissions - slippage
```

---

## What This Means

### The Pipeline Assumes Perfect Execution

```python
# Example trade scenario:

# Bar X: Price = $400, VWAP = $395, ATR = $10
# Signal: RF probability = 0.65 (above 0.5 threshold)
# Setup: Price is $5 below VWAP → Long setup

# MASTER PIPELINE LOGIC:
# 1. Check label: Did price reach VWAP ($395) before hitting stop?
#    Label = 1 (yes, it reached VWAP)
# 
# 2. Calculate P&L using MATH, not actual prices:
#    Stop = $400 - (1.5 × $10) = $385
#    Target = $400 + (1.5 × $10 × 1.01) = $415.15
#    Risk = $15 × 100 shares = $1,500
#    Reward = $15.15 × 100 shares = $1,515
#    
#    Since label = 1: P&L = +$1,515 - fees
#
# 3. Add this to total P&L
#
# WHAT IT DOESN'T DO:
# ❌ Doesn't check if you could actually buy at $400
# ❌ Doesn't check if you could sell at $415.15
# ❌ Doesn't check the actual high/low prices on exit bar
# ❌ Doesn't consider slippage on entry/exit
# ❌ Doesn't consider if you'd be in another position
```

---

## The Critical Difference

### Master Pipeline (Theoretical):
```python
# For EVERY bar where RF ≥ 0.5:
if label == 1:
    pnl += reward  # Fixed amount based on R:R
else:
    pnl -= risk    # Fixed amount based on stop

# Result: 20,884 "trades" in 2024
# Net P&L: +$1,583,733
```

### Streaming Simulator (Realistic):
```python
# For each bar:
if has_position:
    # Check if stop or target hit based on actual high/low
    if current_price <= stop_price:
        exit_pnl = (stop_price - entry_price) * shares - fees
    elif current_price >= target_price:
        exit_pnl = (target_price - entry_price) * shares - fees
else:
    if signal and no_position:
        # Enter at actual bar open/close price
        entry_price = bar['close']
        entry_with_slippage = entry_price + slippage

# Result: 724 trades (only when not already in position)
# Net P&L: -$18,870
```

---

## Why The Huge Difference?

### 1. Trade Count Discrepancy
```
Master Pipeline: 20,884 potential signals
  - Counts EVERY bar where RF ≥ 0.5
  - Assumes you can take all signals simultaneously
  
Streaming Simulator: 724 actual trades
  - Can only hold ONE position at a time
  - While holding position, misses 96.5% of signals
```

### 2. P&L Calculation Method
```
Master Pipeline:
  Win = +$1,515 (fixed, based on R:R math)
  Loss = -$1,500 (fixed, based on stop math)
  Avg = +$75.83 per trade (with 66.4% WR)

Streaming Simulator:
  Win = $221.23 (actual price difference)
  Loss = -$250.50 (actual price difference)
  Avg = -$26.06 per trade (with 47.8% WR)
```

### 3. Entry/Exit Reality
```
Master Pipeline Assumes:
  ✓ Entry exactly at signal bar close price
  ✓ Exit exactly at target or stop price
  ✓ No gaps, no slippage, no bad fills
  ✓ Stop and target always hit precisely

Streaming Simulator Reality:
  ✗ Entry at next bar open (not signal bar close)
  ✗ Exit at bar where stop/target is hit
  ✗ Price may gap through stop
  ✗ Slippage on both entry and exit
```

---

## The Code That Reveals The Truth

### Master Pipeline (scripts/master_pipeline.py, line ~490)
```python
def calculate_dollar_pnl(y_actual, stop_atr, rr, atr_series, entry_price_series, slippage_per_share=0.02):
    """Calculate P&L in dollars, accounting for commissions and slippage."""
    
    # ⚠️ KEY LINE: P&L is based on LABELS, not actual price movement
    gross_pnl_per_trade = np.where(
        y_actual == 1,           # If label says "win"
        reward_per_trade,        # → Give fixed reward
        -risk_per_trade          # → Else give fixed loss
    )
    
    # This is NOT:
    #   pnl = (exit_price - entry_price) * shares
    # 
    # It's:
    #   pnl = label ? +reward : -risk
```

### The Label Generation (src/label_generator.py)
```python
# Labels are generated by looking forward:
# Label = 1 if price reaches target before stop
# Label = 0 if price hits stop before target

# This is used for TRAINING the model
# But master pipeline ALSO uses it for P&L calculation
# That's why it's "theoretical" - it assumes perfect execution
```

---

## What Master Pipeline IS Good For

✅ **Strategy Selection**
  - Which stop width has best theoretical edge?
  - Answer: 1.5 ATR stop has highest theoretical P&L

✅ **Parameter Optimization**
  - Which RF threshold filters best?
  - Which features are most important?

✅ **Understanding Edge**
  - Does the strategy have positive expectancy?
  - What's the theoretical R:R ratio?

✅ **Quick Iteration**
  - Test many configurations quickly
  - No need to simulate full order execution

---

## What Master Pipeline Is NOT Good For

❌ **Realistic P&L Expectations**
  - Don't expect $1.58M in real trading
  - Real P&L will be 10-30% of theoretical (at best)

❌ **Position Sizing**
  - Doesn't account for capital constraints
  - Assumes unlimited capital to take all signals

❌ **Risk Management Testing**
  - No drawdown simulation
  - No equity curve
  - No max loss limits

❌ **Live Trading Preparation**
  - Doesn't simulate actual order flow
  - Doesn't test execution logic
  - Doesn't handle position conflicts

---

## The Bottom Line

### Master Pipeline = "What's Possible"
- If you had perfect execution
- If you could take every signal
- If stops and targets always hit exactly
- **Result: $1,583,733 theoretical profit**

### Streaming Simulator = "What's Achievable"
- With realistic execution
- With position constraints
- With actual market conditions
- **Result: -$18,870 actual loss (needs improvement)**

---

## The Gap Is The Challenge

```
$1,583,733 (theoretical)
    ↓
    Gap = Implementation Reality
    ↓
-$18,870 (actual)
```

**That gap is your job to close through:**
1. Better entry/exit logic
2. Profit-taking rules
3. Tighter stops
4. Better signal filtering
5. Position management

**Realistic goal:** Achieve 10-20% of theoretical = $150K-$300K

---

## Conclusion

**No, the master pipeline is NOT using actual stock prices to trade.**

It's using:
- ✅ Actual prices for indicators
- ✅ Actual labels (win/loss outcomes)
- ❌ Theoretical R:R math for P&L
- ❌ No actual trade simulation

Think of it as:
- **Master Pipeline** = "Strategy has edge, here's the maximum possible"
- **Streaming Simulator** = "Can we actually capture that edge in reality?"

The answer so far: Not yet, but we know how to improve it.
