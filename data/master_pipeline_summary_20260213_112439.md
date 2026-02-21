# Master pipeline summary (20260213_112439)

- Test year: 2024
- Model kind: regressor
- Regression target: net_r
- Selection mode: top (top_n=500)
- Label mode: touch_vwap
- min_rr (post-filter): 0.0
- Slippage: 0.01

## Recommended stop_atr (by max total_net_pnl)

- stop_atr = **0.75**

## Results (sorted by total_net_pnl)

|   stop_atr | model_kind   | selection   |   rf_threshold | regression_target   |   n_trades |   n_trades_executed |   win_rate |      rr |   total_net_pnl |   total_return_pct |   avg_net_pnl_per_trade |   avg_risk_dollars |
|-----------:|:-------------|:------------|---------------:|:--------------------|-----------:|--------------------:|-----------:|--------:|----------------:|-------------------:|------------------------:|-------------------:|
|       0.75 | regressor    | top_500     |            nan | net_r               |        500 |                 228 |      0.324 | 2.0157  |       2449.84   |         0.244984   |               10.7449   |           161.067  |
|       0.5  | regressor    | top_500     |            nan | net_r               |        500 |                 223 |      0.236 | 3.02355 |        588.768  |         0.0588768  |                2.64022  |           101.312  |
|       0.4  | regressor    | top_500     |            nan | net_r               |        500 |                 258 |      0.19  | 3.77944 |        577.378  |         0.0577378  |                2.2379   |            88.6877 |
|       0.25 | regressor    | top_500     |            nan | net_r               |        500 |                 306 |      0.118 | 6.0471  |        171.296  |         0.0171296  |                0.559792 |            54.3106 |
|       1    | regressor    | top_500     |            nan | net_r               |        500 |                 228 |      0.328 | 1.51177 |         93.4727 |         0.00934727 |                0.409968 |           204.284  |
|       1.25 | regressor    | top_500     |            nan | net_r               |        500 |                 237 |      0.302 | 1.20942 |      -1875.65   |        -0.187565   |               -7.91413  |           228.148  |
|       0.6  | regressor    | top_500     |            nan | net_r               |        500 |                 211 |      0.27  | 2.51962 |      -2631.98   |        -0.263198   |              -12.4738   |           124.972  |
|       0.35 | regressor    | top_500     |            nan | net_r               |        500 |                 289 |      0.142 | 4.31936 |      -3063.04   |        -0.306304   |              -10.5988   |            74.0351 |
|       1.5  | regressor    | top_500     |            nan | net_r               |        500 |                 219 |      0.278 | 1.00785 |      -4970.19   |        -0.497019   |              -22.6949   |           258.949  |
