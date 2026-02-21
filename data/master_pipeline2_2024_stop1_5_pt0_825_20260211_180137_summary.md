# master_pipeline2 summary

- Year (test): 2024
- Train years: [2016, 2024)
- Validation (Option A): 2023
- Stop (ATR): 1.5
- Proba threshold: 0.825 (auto(validation=2023))
- Sizing: 0.25% risk/trade
- Capital cap: 1,000,000
- Costs (round-trip): 2*(commission+slippage)*shares = 2*(0.005+0.01)*shares

## Semantics
- Model target: classification: P(net_pnl>0) under realized-path execution + costs
- P&L definition: realized_path_dollars
- Win definitions: WR(label)=target-first, WR(net)=net_pnl>0

## Model fit diagnostics (classification)
Metrics computed on `net_profitable = (net_pnl>0)` labels.
- AUC/logloss/Brier use probabilities; acc/confusion use threshold=0.5.

### train
- n: 156,359
- base_rate (P(y=1)): 0.4216
- AUC: 0.7813
- logloss: 0.5726
- brier: 0.1957
- acc@0.5: 0.7002
- confusion@0.5 (tn fp / fn tp): 64269 26173 / 20705 45212

### valid
- n: 19,428
- base_rate (P(y=1)): 0.5018
- AUC: 0.7543
- logloss: 0.6103
- brier: 0.2110
- acc@0.5: 0.6695
- confusion@0.5 (tn fp / fn tp): 5817 3862 / 2559 7190

### test
- n: 19,548
- base_rate (P(y=1)): 0.5163
- AUC: 0.6750
- logloss: 0.6433
- brier: 0.2268
- acc@0.5: 0.6123
- confusion@0.5 (tn fp / fn tp): 5037 4419 / 3160 6932

## Results
- Trades executed: 935
- WR(label): 88.88%
- WR(net): 88.88%
- Gross P&L: $45,951.35
- Net P&L: $3,074.45

## Exit reasons
- target: 831
- stop: 102
- eod: 2
## Test-year results by probability threshold
Test year = 2024. Backtest uses the same realized-path execution + costs + capital cap.

| Threshold | Executed | WR(net) | WR(label) | Gross P&L | Net P&L | Return % |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 6,670 | 65.43% | 64.98% | $-423,638.30 | $-734,777.21 | -73.478% |
| 0.55 | 6,221 | 68.45% | 68.75% | $-322,415.81 | $-616,934.48 | -61.693% |
| 0.60 | 5,719 | 71.50% | 71.94% | $-253,298.86 | $-530,212.84 | -53.021% |
| 0.65 | 5,068 | 74.78% | 75.34% | $-190,860.16 | $-443,260.30 | -44.326% |
| 0.70 | 4,393 | 77.76% | 78.26% | $-137,974.38 | $-362,406.27 | -36.241% |
| 0.75 | 3,580 | 80.47% | 80.87% | $-36,210.11 | $-221,852.69 | -22.185% |
| 0.80 | 1,980 | 86.31% | 86.31% | $68,054.90 | $-34,503.76 | -3.450% |
