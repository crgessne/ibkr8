# Master pipeline summary (20260212_094444)

- Test year: 2024
- Model kind: regressor
- Regression target: net_r
- Selection mode: top (top_n=5000)
- Label mode: touch_vwap
- min_rr (post-filter): 0.0
- Slippage: 0.01

## Recommended stop_atr (by max total_net_pnl)

- stop_atr = **0.75**

## Results (sorted by total_net_pnl)

|   stop_atr | model_kind   | selection   |   rf_threshold | regression_target   |   n_trades |   n_trades_executed |   win_rate |      rr |   total_net_pnl |   total_return_pct |   avg_net_pnl_per_trade |   avg_risk_dollars |
|-----------:|:-------------|:------------|---------------:|:--------------------|-----------:|--------------------:|-----------:|--------:|----------------:|-------------------:|------------------------:|-------------------:|
|       0.75 | regressor    | top_5000    |            nan | net_r               |       5000 |                1502 |     0.4248 | 2.0157  |        -9718.06 |          -0.971806 |                -6.47008 |           153.814  |
|       1.5  | regressor    | top_5000    |            nan | net_r               |       5000 |                1600 |     0.6738 | 1.00785 |       -10158.9  |          -1.01589  |                -6.34931 |           286.124  |
|       0.6  | regressor    | top_5000    |            nan | net_r               |       5000 |                1715 |     0.3534 | 2.51962 |       -13339.3  |          -1.33393  |                -7.778   |           122.001  |
|       1    | regressor    | top_5000    |            nan | net_r               |       5000 |                1453 |     0.5214 | 1.51177 |       -15235.8  |          -1.52358  |               -10.4857  |           195.965  |
|       1.25 | regressor    | top_5000    |            nan | net_r               |       5000 |                1371 |     0.5776 | 1.20942 |       -15311    |          -1.5311   |               -11.1678  |           249.497  |
|       0.5  | regressor    | top_5000    |            nan | net_r               |       5000 |                1814 |     0.2858 | 3.02355 |       -19783.3  |          -1.97833  |               -10.9059  |           101.816  |
|       0.25 | regressor    | top_5000    |            nan | net_r               |       5000 |                2514 |     0.1476 | 6.0471  |       -22710.8  |          -2.27108  |                -9.03373 |            50.963  |
|       0.35 | regressor    | top_5000    |            nan | net_r               |       5000 |                2206 |     0.2054 | 4.31936 |       -25571.1  |          -2.55711  |               -11.5916  |            67.9553 |
|       0.4  | regressor    | top_5000    |            nan | net_r               |       5000 |                2147 |     0.222  | 3.77944 |       -25752    |          -2.5752   |               -11.9944  |            78.0403 |
