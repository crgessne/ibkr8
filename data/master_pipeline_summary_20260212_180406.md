# Master pipeline summary (20260212_180406)

- Test year: 2024
- Model kind: regressor
- Regression target: net_pnl
- Selection mode: threshold (top_n=500)
- Label mode: touch_vwap
- min_rr (post-filter): 0.0
- Slippage: 0.01

## Recommended stop_atr (by max total_net_pnl)

- stop_atr = **0.25**

## Results (sorted by total_net_pnl)

|   stop_atr | model_kind   | selection   |   rf_threshold | regression_target   |   n_trades |   n_trades_executed |   win_rate |      rr |   total_net_pnl |   total_return_pct |   avg_net_pnl_per_trade |   avg_risk_dollars |
|-----------:|:-------------|:------------|---------------:|:--------------------|-----------:|--------------------:|-----------:|--------:|----------------:|-------------------:|------------------------:|-------------------:|
|       0.25 | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.138 | 6.0471  |         -577.37 |          -0.057737 |                -1.15474 |            63.6848 |
|       0.6  | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.266 | 2.51962 |        -1301.04 |          -0.130104 |                -2.60209 |           156.242  |
|       0.35 | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.158 | 4.31936 |        -4018.74 |          -0.401874 |                -8.03748 |            80.3795 |
|       0.5  | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.214 | 3.02355 |        -4365.82 |          -0.436582 |                -8.73163 |           115.889  |
|       0.4  | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.168 | 3.77944 |        -6678.13 |          -0.667813 |               -13.3563  |            91.8914 |
|       0.75 | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.3   | 2.0157  |        -9858.95 |          -0.985895 |               -19.7179  |           178.026  |
|       1.25 | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.354 | 1.20942 |       -16147.9  |          -1.61479  |               -32.2958  |           247.538  |
|       1    | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.308 | 1.51177 |       -21828.7  |          -2.18287  |               -43.6573  |           222.858  |
|       1.5  | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.356 | 1.00785 |       -23004.4  |          -2.30044  |               -46.0087  |           271.061  |
