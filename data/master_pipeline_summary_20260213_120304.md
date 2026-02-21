# Master pipeline summary (20260213_120304)

- Test year: 2024
- Model kind: regressor
- Regression target: net_r
- Selection mode: top (top_n=500)
- Label mode: touch_vwap
- min_rr (post-filter): 0.0
- Slippage: 0.01

## Recommended stop_atr (by max total_net_pnl)

- stop_atr = **1.00**

## Results (sorted by total_net_pnl)

|   stop_atr | model_kind   | selection   |   rf_threshold | regression_target   |   n_trades |   n_trades_executed |   win_rate |      rr |   total_net_pnl |   total_return_pct |   avg_net_pnl_per_trade |   avg_risk_dollars |
|-----------:|:-------------|:------------|---------------:|:--------------------|-----------:|--------------------:|-----------:|--------:|----------------:|-------------------:|------------------------:|-------------------:|
|       1    | regressor    | top_500     |            nan | net_r               |        500 |                 198 |      0.398 | 1.51177 |        5794.79  |          0.579479  |                29.2666  |           211.67   |
|       1.5  | regressor    | top_500     |            nan | net_r               |        500 |                 243 |      0.39  | 1.00785 |        3412.55  |          0.341255  |                14.0434  |           298.752  |
|       0.75 | regressor    | top_500     |            nan | net_r               |        500 |                 187 |      0.348 | 2.0157  |        3405.03  |          0.340503  |                18.2087  |           152.051  |
|       0.25 | regressor    | top_500     |            nan | net_r               |        500 |                 289 |      0.14  | 6.0471  |        3033.04  |          0.303304  |                10.4949  |            54.7163 |
|       0.5  | regressor    | top_500     |            nan | net_r               |        500 |                 236 |      0.25  | 3.02355 |        1825.48  |          0.182548  |                 7.73509 |           107.164  |
|       0.6  | regressor    | top_500     |            nan | net_r               |        500 |                 208 |      0.282 | 2.51962 |         396.95  |          0.039695  |                 1.90841 |           134.393  |
|       0.35 | regressor    | top_500     |            nan | net_r               |        500 |                 275 |      0.178 | 4.31936 |        -842.215 |         -0.0842215 |                -3.0626  |            76.4651 |
|       1.25 | regressor    | top_500     |            nan | net_r               |        500 |                 205 |      0.37  | 1.20942 |       -2030.74  |         -0.203074  |                -9.90607 |           253.998  |
|       0.4  | regressor    | top_500     |            nan | net_r               |        500 |                 272 |      0.204 | 3.77944 |       -2195.26  |         -0.219526  |                -8.07083 |            85.8187 |
