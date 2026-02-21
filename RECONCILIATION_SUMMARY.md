# P&L Reconciliation - Summary & Next Steps

## Problem Identified

**Historic state**: Master pipeline showed large positive P&L while the streaming/concurrent simulator showed flat/negative results.

**Current state**: This gap is now largely explained and reproducible **apples-to-apples**.

- **Per-bar features + probabilities match 100%** (master vs concurrent)
- **Signals match** once filters/capital/EOD assumptions are aligned
- The remaining "win-rate gap" was primarily a **metric-definition mismatch**:
  - **Label win** = target (VWAP) touched before stop (forward scan)
  - **Net win** = trade is **net profitable after fees/slippage**

---

## Root Causes (What Actually Drove the Discrepancy)

### 1) Trade counting / execution constraints
- Master counts many more eligible signals
- Streaming/concurrent may execute fewer trades due to position/capital constraints
- Fix for apples-to-apples comparisons: run concurrent with master-like capital/position settings

### 2) End-of-day (EOD) assumption mismatch
- Labeling logic is **same-day only**
- Streaming originally held positions across days
- Fix: forced EOD flattening in concurrent

### 3) Long/short + P&L accounting mismatches (fixed)
- Concurrent was originally long-only and had an incorrect short P&L formula
- Fix: direction-aware stop/target logic + correct short net P&L

### 4) Win rate definition mismatch (the big one)
- **Master "win rate" (label semantics)** can be high even when many trades are **not net profitable after costs**
- Added master support for:
  - `--win-definition label`
  - `--win-definition net_pnl` (ATR-based reward mapping)
  - `--win-definition realized_net_pnl` (**realized path**, net after costs)

---

## What To Use Going Forward

### For reconciliation / parity checks
- Use concurrent backtester in **label mode** (forward scan) to match labels
- Use master `--win-definition label` to compare to label-mode win rates

### For realistic performance reporting
- Use **net-profitable win rate**:
  - Master: `--win-definition realized_net_pnl`
  - Concurrent simulate-mode: compute wins as `net_pnl > 0` if you want "real-world" win rate

---

## Next Steps

### Phase 1: Lock the reporting semantics
- Update reports/tables to always show both:
  - `WR(label)` (target touched first)
  - `WR(net)` (net profitable after costs)
- Ensure any plots/export filenames include the mode (label vs simulate) to avoid overwrites

### Phase 2: Strategy iteration (only after metrics are consistent)
- Profit-taking / partial exits
- Trailing stops / move-to-breakeven
- Improved entry filters (momentum/volatility regime)

---

## Status

- Reconciliation: **complete for signals/features and label-mode outcomes**
- Metric clarity: **resolved (label vs net profitable)**
- Master pipeline win definition: **supports realized-path net P&L wins**

**Recommended command (master, realistic win definition)**:
```powershell
python scripts\master_pipeline.py --win-definition realized_net_pnl
```
