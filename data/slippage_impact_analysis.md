# 💸 Slippage Impact Analysis

**Generated**: 2026-02-08
**Strategy**: RF VWAP Reversion (1.25 ATR stop, RF≥0.50)

---

## Summary

This analysis shows how different slippage assumptions affect the P&L of the recommended strategy (1.25 ATR stop width with RF threshold ≥0.50).

**Key Finding**: With **0.02 slippage** (vs default 0.01), the recommended strategy shows:
- **Net P&L**: $1,543,968 (test period, 19,257 trades)
- **Cost per round trip**: $5.00 ($0.005 commission + $0.02 slippage × 2 × 100 shares)
- **Est. annual P&L**: $736,207

---

## Slippage Impact Breakdown

### Current Results (Slippage = $0.02/share)

From `master_pipeline_summary_20260208_120424.md`:

| Metric | Value |
|--------|-------|
| **Slippage per share** | $0.0200 |
| **Commission per share** | $0.0050 |
| **Total cost per share (round trip)** | $0.0500 |
| **Cost per trade (100 shares)** | $5.00 |
| **Test period net P&L** | $1,543,968 |
| **Trades (test)** | 19,257 |
| **Average P&L per trade** | $80.17 |
| **Win rate** | 64.8% |
| **Annual P&L estimate** | $736,207 |

### Cost Structure Analysis

For 100 shares per trade:
- **Entry**: $0.50 commission + $2.00 slippage = $2.50
- **Exit**: $0.50 commission + $2.00 slippage = $2.50
- **Round trip total**: $5.00

Total costs over test period:
- **19,257 trades × $5.00 = $96,285**

---

## Comparative Scenarios

### Scenario 1: Ultra-Low Slippage ($0.005/share)
*Ideal HFT conditions or limit orders with perfect fills*

- **Round trip cost**: $2.00 per 100 shares
- **Est. total costs**: $38,514 (19,257 trades)
- **Cost savings vs $0.02**: $57,771
- **Est. net P&L**: ~$1,601,739 (+3.7% vs current)
- **Annual P&L**: ~$763,923

### Scenario 2: Default Slippage ($0.01/share)
*IBKR Pro with good execution*

- **Round trip cost**: $3.00 per 100 shares
- **Est. total costs**: $57,771 (19,257 trades)
- **Cost savings vs $0.02**: $38,514
- **Est. net P&L**: ~$1,582,482 (+2.5% vs current)
- **Annual P&L**: ~$754,515

### Scenario 3: Current Analysis ($0.02/share)
*Conservative retail estimate*

- **Round trip cost**: $5.00 per 100 shares
- **Est. total costs**: $96,285
- **Est. net P&L**: $1,543,968 (baseline)
- **Annual P&L**: $736,207

### Scenario 4: High Slippage ($0.03/share)
*Market orders during volatile periods*

- **Round trip cost**: $7.00 per 100 shares
- **Est. total costs**: $134,799 (19,257 trades)
- **Cost increase vs $0.02**: $38,514
- **Est. net P&L**: ~$1,505,454 (-2.5% vs current)
- **Annual P&L**: ~$717,864

### Scenario 5: Extreme Slippage ($0.05/share)
*Very poor execution / high-frequency entries*

- **Round trip cost**: $11.00 per 100 shares
- **Est. total costs**: $211,827 (19,257 trades)
- **Cost increase vs $0.02**: $115,542
- **Est. net P&L**: ~$1,428,426 (-7.5% vs current)
- **Annual P&L**: ~$681,346

---

## Impact Summary Table

| Slippage/Share | Round Trip Cost | Total Costs | Est. Net P&L | Annual P&L | % Change |
|----------------|-----------------|-------------|--------------|------------|----------|
| $0.005 | $2.00 | $38,514 | $1,601,739 | $763,923 | +3.7% |
| $0.010 | $3.00 | $57,771 | $1,582,482 | $754,515 | +2.5% |
| **$0.020** | **$5.00** | **$96,285** | **$1,543,968** | **$736,207** | **0.0%** |
| $0.030 | $7.00 | $134,799 | $1,505,454 | $717,864 | -2.5% |
| $0.050 | $11.00 | $211,827 | $1,428,426 | $681,346 | -7.5% |

---

## Key Insights

### 1. **Slippage Matters, But Strategy Remains Profitable**
- Even at $0.05/share slippage (5× typical), strategy still generates $681k/year
- Strategy is robust across execution quality scenarios

### 2. **Diminishing Returns from Better Execution**
- Moving from $0.02 → $0.01 saves ~$38k/year (+2.5%)
- Moving from $0.01 → $0.005 saves ~$19k/year (+1.2%)
- Getting better fills has declining marginal benefit

### 3. **Scale of Costs**
- At current slippage ($0.02), costs = $96k over 2.1 years
- That's ~6% of gross P&L
- Costs are manageable relative to edge

### 4. **Breakeven Analysis**
- Strategy generates avg $80.17/trade net P&L at $0.02 slippage
- At $0.05 slippage, avg drops to ~$74.17/trade
- Would need slippage of ~$0.40/share to reach breakeven
- This gives huge margin of safety

### 5. **Realistic Expectations**
- **IBKR Pro**: $0.005-0.01/share typical
- **Market orders on TSLA**: $0.01-0.02/share
- **Poor timing (crossing spread)**: $0.02-0.05/share
- **Current assumption ($0.02)** is conservative but realistic

---

## Recommendations

### For Live Trading

1. **Use Limit Orders**
   - Join bid/ask rather than cross spread
   - Target $0.01 or better slippage
   - Adds ~$19k/year vs market orders

2. **Monitor Execution Quality**
   - Track actual fill prices vs signal prices
   - Measure realized slippage per trade
   - Adjust assumptions if real slippage exceeds $0.02

3. **Time Entries Carefully**
   - Avoid panic entries crossing full spread
   - Use limit orders with short duration
   - Cancel/repost if not filled quickly

4. **Test Different Slippage Models**
   - Run pipeline with `--slippage 0.01` (optimistic)
   - Run pipeline with `--slippage 0.03` (pessimistic)
   - Build confidence interval around P&L estimates

### Command Examples

```bash
# Optimistic (good execution)
python scripts/master_pipeline.py --slippage 0.01

# Conservative (current)
python scripts/master_pipeline.py --slippage 0.02

# Pessimistic (poor execution)
python scripts/master_pipeline.py --slippage 0.03

# With walk-forward validation
python scripts/master_pipeline.py --slippage 0.02 --walk-forward
```

---

## Bottom Line

**Slippage impact is material but manageable:**
- $0.01/share difference = ~$38k/year (~5% P&L impact)
- Strategy remains profitable even at 5× normal slippage
- Focus on limit orders to minimize slippage
- Current $0.02 assumption is appropriately conservative

**The edge is strong enough to overcome realistic execution costs.**
