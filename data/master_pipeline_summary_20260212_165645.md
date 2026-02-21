# Master pipeline summary (20260212_165645)

- Test year: 2024
- Model kind: regressor
- Regression target: net_pnl
- Selection mode: threshold (top_n=100)
- Label mode: touch_vwap
- min_rr (post-filter): 0.0
- Slippage: 0.01

## Recommended stop_atr (by max total_net_pnl)

- stop_atr = **0.60**

## Results (sorted by total_net_pnl)

|   stop_atr | model_kind   | selection   |   rf_threshold | regression_target   |   n_trades |   n_trades_executed |   win_rate |      rr |   total_net_pnl |   total_return_pct |   avg_net_pnl_per_trade |   avg_risk_dollars |
|-----------:|:-------------|:------------|---------------:|:--------------------|-----------:|--------------------:|-----------:|--------:|----------------:|-------------------:|------------------------:|-------------------:|
|       0.6  | regressor    | top_100     |            nan | net_pnl             |        100 |                 100 |       0.29 | 2.51962 |       3915.52   |         0.391552   |               39.1552   |           180.602  |
|       0.35 | regressor    | top_100     |            nan | net_pnl             |        100 |                 100 |       0.18 | 4.31936 |       3512.83   |         0.351283   |               35.1283   |            98.2223 |
|       0.5  | regressor    | top_100     |            nan | net_pnl             |        100 |                 100 |       0.25 | 3.02355 |       2922.11   |         0.292211   |               29.2211   |           138.801  |
|       0.75 | regressor    | top_100     |            nan | net_pnl             |        100 |                 100 |       0.31 | 2.0157  |        254.621  |         0.0254621  |                2.54621  |           221.382  |
|       0.4  | regressor    | top_100     |            nan | net_pnl             |        100 |                 100 |       0.16 | 3.77944 |        150.675  |         0.0150675  |                1.50675  |           109.248  |
|       0.25 | regressor    | top_100     |            nan | net_pnl             |        100 |                 100 |       0.11 | 6.0471  |        -58.2946 |        -0.00582946 |               -0.582946 |            73.8464 |
|       1    | regressor    | top_100     |            nan | net_pnl             |        100 |                 100 |       0.31 | 1.51177 |      -3754.16   |        -0.375416   |              -37.5416   |           270.483  |
|       1.25 | regressor    | top_100     |            nan | net_pnl             |        100 |                 100 |       0.32 | 1.20942 |      -7764.54   |        -0.776454   |              -77.6454   |           335.804  |
|       1.5  | regressor    | top_100     |            nan | net_pnl             |        100 |                 100 |       0.3  | 1.00785 |     -10329.2    |        -1.03292    |             -103.292    |           364.971  |
