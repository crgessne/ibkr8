# Master pipeline summary (20260213_100358)

- Test year: 2024
- Model kind: regressor
- Regression target: net_r
- Selection mode: top (top_n=500)
- Label mode: touch_vwap
- min_rr (post-filter): 0.0
- Slippage: 0.01

## Recommended stop_atr (by max total_net_pnl)

- stop_atr = **1.25**

## Results (sorted by total_net_pnl)

|   stop_atr | model_kind   | selection   |   rf_threshold | regression_target   |   n_trades |   n_trades_executed |   win_rate |      rr |   total_net_pnl |   total_return_pct |   avg_net_pnl_per_trade |   avg_risk_dollars |
|-----------:|:-------------|:------------|---------------:|:--------------------|-----------:|--------------------:|-----------:|--------:|----------------:|-------------------:|------------------------:|-------------------:|
|       1.25 | regressor    | top_500     |            nan | net_r               |        500 |                 218 |      0.432 | 1.20942 |        3963.19  |          0.396319  |               18.1798   |           245.235  |
|       0.6  | regressor    | top_500     |            nan | net_r               |        500 |                 215 |      0.3   | 2.51962 |        2528.04  |          0.252804  |               11.7583   |           131.369  |
|       1    | regressor    | top_500     |            nan | net_r               |        500 |                 178 |      0.404 | 1.51177 |        1468.74  |          0.146874  |                8.25135  |           205.625  |
|       1.5  | regressor    | top_500     |            nan | net_r               |        500 |                 236 |      0.412 | 1.00785 |         738.436 |          0.0738436 |                3.12897  |           298.035  |
|       0.75 | regressor    | top_500     |            nan | net_r               |        500 |                 189 |      0.356 | 2.0157  |         734.315 |          0.0734315 |                3.88527  |           158.324  |
|       0.25 | regressor    | top_500     |            nan | net_r               |        500 |                 306 |      0.118 | 6.0471  |         171.296 |          0.0171296 |                0.559792 |            54.3106 |
|       0.4  | regressor    | top_500     |            nan | net_r               |        500 |                 267 |      0.174 | 3.77944 |        -692.341 |         -0.0692341 |               -2.59304  |            86.268  |
|       0.5  | regressor    | top_500     |            nan | net_r               |        500 |                 218 |      0.246 | 3.02355 |       -1114.28  |         -0.111428  |               -5.11137  |           102.154  |
|       0.35 | regressor    | top_500     |            nan | net_r               |        500 |                 281 |      0.15  | 4.31936 |       -1427.25  |         -0.142725  |               -5.07919  |            71.9972 |
