# Master pipeline summary (20260212_114944)

- Test year: 2024
- Model kind: regressor
- Regression target: realized_net_pnl
- Selection mode: threshold (top_n=5000)
- Label mode: touch_vwap
- min_rr (post-filter): 0.0
- Slippage: 0.01

## Recommended stop_atr (by max total_net_pnl)

- stop_atr = **0.25**

## Results (sorted by total_net_pnl)

|   stop_atr | model_kind   | selection   |   rf_threshold | regression_target   |   n_trades |   n_trades_executed |   win_rate |      rr |   total_net_pnl |   total_return_pct |   avg_net_pnl_per_trade |   avg_risk_dollars |
|-----------:|:-------------|:------------|---------------:|:--------------------|-----------:|--------------------:|-----------:|--------:|----------------:|-------------------:|------------------------:|-------------------:|
|       0.25 | regressor    | top_5000    |            nan | realized_net_pnl    |       4150 |                4150 |   0.148193 | 6.0471  |        -42799.8 |           -4.27998 |                -10.3132 |            33.2358 |
|       0.35 | regressor    | top_5000    |            nan | realized_net_pnl    |       4150 |                4150 |   0.196867 | 4.31936 |        -52146.7 |           -5.21467 |                -12.5655 |            46.5301 |
|       0.4  | regressor    | top_5000    |            nan | realized_net_pnl    |       4150 |                4150 |   0.221928 | 3.77944 |        -55466.6 |           -5.54666 |                -13.3654 |            53.1773 |
|       0.5  | regressor    | top_5000    |            nan | realized_net_pnl    |       4150 |                4150 |   0.256867 | 3.02355 |        -59399.6 |           -5.93996 |                -14.3131 |            66.4716 |
|       0.6  | regressor    | top_5000    |            nan | realized_net_pnl    |       4150 |                4150 |   0.29253  | 2.51962 |        -62870.8 |           -6.28708 |                -15.1496 |            79.7659 |
|       0.75 | regressor    | top_5000    |            nan | realized_net_pnl    |       4150 |                4150 |   0.333735 | 2.0157  |        -70882.5 |           -7.08825 |                -17.0801 |            99.7074 |
|       1    | regressor    | top_5000    |            nan | realized_net_pnl    |       4150 |                4150 |   0.387711 | 1.51177 |        -92668.7 |           -9.26687 |                -22.3298 |           132.943  |
|       1.25 | regressor    | top_5000    |            nan | realized_net_pnl    |       4150 |                4150 |   0.421446 | 1.20942 |       -126317   |          -12.6317  |                -30.4378 |           166.179  |
|       1.5  | regressor    | top_5000    |            nan | realized_net_pnl    |       4150 |                4150 |   0.453494 | 1.00785 |       -149206   |          -14.9206  |                -35.9533 |           199.415  |
