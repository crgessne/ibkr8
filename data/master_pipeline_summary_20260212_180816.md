# Master pipeline summary (20260212_180816)

- Test year: 2024
- Model kind: regressor
- Regression target: net_pnl
- Selection mode: threshold (top_n=500)
- Label mode: touch_vwap
- min_rr (post-filter): 0.0
- Slippage: 0.01

## Recommended stop_atr (by max total_net_pnl)

- stop_atr = **0.50**

## Results (sorted by total_net_pnl)

|   stop_atr | model_kind   | selection   |   rf_threshold | regression_target   |   n_trades |   n_trades_executed |   win_rate |      rr |   total_net_pnl |   total_return_pct |   avg_net_pnl_per_trade |   avg_risk_dollars |
|-----------:|:-------------|:------------|---------------:|:--------------------|-----------:|--------------------:|-----------:|--------:|----------------:|-------------------:|------------------------:|-------------------:|
|       0.5  | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.236 | 3.02355 |        -87.0654 |        -0.00870654 |               -0.174131 |           118.822  |
|       0.6  | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.278 | 2.51962 |      -1560.09   |        -0.156009   |               -3.12017  |           155.431  |
|       0.25 | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.128 | 6.0471  |      -2527.16   |        -0.252716   |               -5.05433  |            60.7854 |
|       0.35 | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.166 | 4.31936 |      -4272.2    |        -0.42722    |               -8.54439  |            82.9698 |
|       0.75 | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.322 | 2.0157  |      -4824.04   |        -0.482404   |               -9.64808  |           187.71   |
|       0.4  | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.168 | 3.77944 |      -7321.12   |        -0.732112   |              -14.6422   |            96.4105 |
|       1.5  | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.38  | 1.00785 |     -19223.9    |        -1.92239    |              -38.4478   |           284.866  |
|       1    | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.316 | 1.51177 |     -20833.4    |        -2.08334    |              -41.6667   |           221.782  |
|       1.25 | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.33  | 1.20942 |     -26966      |        -2.6966     |              -53.932    |           251.182  |
