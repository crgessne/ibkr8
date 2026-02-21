# master_pipeline2 summary

- Year (test): 2024
- Train years: [2016, 2024)
- Validation (Option A): 2023
- Stop (ATR): 1.5
- Proba threshold: 0.85 (auto(validation=2023))
- Shares/trade: 100
- Capital cap: 1,000,000
- Costs (round-trip): 2*(commission+slippage)*shares = 2*(0.005+0.01)*100

## Semantics
- Model target: classification: P(net_pnl>0) under realized-path execution + costs
- P&L definition: realized_path_dollars
- Win definitions: WR(label)=target-first, WR(net)=net_pnl>0

## Results
- Trades executed: 0
- WR(label): nan%
- WR(net): nan%
- Gross P&L: $0.00
- Net P&L: $0.00

## Exit reasons
- target: 0
- stop: 0
- eod: 0