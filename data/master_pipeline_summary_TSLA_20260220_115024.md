# Master pipeline summary — TSLA (20260220_115024)

- Ticker: TSLA
- Data file: data\tsla_5min_10years.csv
- Train years: <2024
- Test years: 2024+
- Model kind: nn_pnl
- RF thresholds: [0.5, 0.55, 0.6, 0.65]
- Label mode: net_positive
- min_rr (post-filter): 0.0
- Slippage: 0.1

## Recommended stop_atr (by max total_net_pnl)

- stop_atr = **0.75**

## Results (sorted by total_net_pnl)

|   stop_atr | model_kind   |   selection |   rf_threshold | regression_target   |   n_trades |   n_trades_executed |   win_rate |     rr |   total_net_pnl |   total_return_pct |   avg_net_pnl_per_trade |   avg_risk_dollars |
|-----------:|:-------------|------------:|---------------:|:--------------------|-----------:|--------------------:|-----------:|-------:|----------------:|-------------------:|------------------------:|-------------------:|
|       0.75 | classifier   |        0.65 |           0.65 |                     |       5069 |                5069 |   0.374433 | 2.0157 |         -138965 |           -13.8965 |                -27.4147 |            154.145 |
|       0.75 | classifier   |        0.6  |           0.6  |                     |       5267 |                5267 |   0.372318 | 2.0157 |         -148059 |           -14.8059 |                -28.1106 |            152.89  |
|       0.75 | classifier   |        0.55 |           0.55 |                     |       5447 |                5447 |   0.371214 | 2.0157 |         -155256 |           -15.5256 |                -28.503  |            151.356 |
|       0.75 | classifier   |        0.5  |           0.5  |                     |       5617 |                5617 |   0.36888  | 2.0157 |         -164397 |           -16.4397 |                -29.2678 |            150.133 |
|       0.75 | classifier   |        0    |           0    |                     |       7024 |                7024 |   0.349658 | 2.0157 |         -243447 |           -24.3447 |                -34.6593 |            139.859 |

## Total P&L on Test Years

Test period: **2024+**

|   stop_atr |   n_trades_executed | win_rate   | total_net_pnl   | avg_net_pnl_per_trade   | total_return_pct   |
|-----------:|--------------------:|:-----------|:----------------|:------------------------|:-------------------|
|       0.75 |                5617 | 36.9%      | $-164,397       | $-29                    | -16.4%             |

**Grand total across all stops**: $-164,397 (5,617 total trades)

**Best stop**: 0.75 ATR -> $-164,397 (5,617 trades, 36.9% WR)


## Data Metrics

|   stop_atr |   n_total_bars |   n_setup_bars |   pct_kept |   n_train |   n_test |   base_wr_train |   base_wr_test | train_start   | train_end   | test_start   | test_end   |
|-----------:|---------------:|---------------:|-----------:|----------:|---------:|----------------:|---------------:|:--------------|:------------|:-------------|:-----------|
|       0.75 |         194858 |          34515 |       17.7 |     27491 |     7024 |           18.86 |          34.97 | 2015-12-30    | 2023-12-29  | 2024-01-02   | 2026-02-06 |

## RF Model Value-Add

all_WR = blind win rate of all setup bars. sel_WR = win rate of model-selected bars.

|   stop_atr |   base_wr_train |   rf_wr_train |   lift_train |   rf_n_train |   base_wr_test |   rf_wr_test |   lift_test |   rf_n_test |   f1_train |   f1_test |   auc_train |   auc_test |
|-----------:|----------------:|--------------:|-------------:|-------------:|---------------:|-------------:|------------:|------------:|-----------:|----------:|------------:|-----------:|
|       0.75 |           18.86 |          43.4 |        24.54 |         5000 |          34.97 |        34.97 |           0 |        7024 |      0.501 |     0.513 |       0.598 |      0.542 |

## Feature Importance Rankings

### Average across all stops

| Rank | Feature | Avg Importance |
|-----:|:--------|---------------:|
| 1 | is_long_setup | 12.1325 |
| 2 | pct_of_day_range | 10.8682 |
| 3 | dist_vah_atr | 8.2103 |
| 4 | vwap_stretch_zscore | 8.1488 |
| 5 | rel_vol | 6.6735 |
| 6 | vwap_width_atr | 6.3191 |
| 7 | sr_rejection_score | 4.8656 |
| 8 | dist_or_low_atr | 3.6032 |
| 9 | extension_speed | 2.5473 |
| 10 | vol_ratio | 2.3246 |
| 11 | momentum_6bar_atr | 2.1584 |
| 12 | session_phase | 2.1424 |
| 13 | price_to_vwap_atr | 2.0765 |
| 14 | momentum_3bar_atr | 1.8382 |
| 15 | dist_poc_atr | 1.7142 |
| 16 | dist_day_low_atr | 1.3728 |
| 17 | sr_trigger_score | 1.2074 |
| 18 | nearest_sr_atr | 1.1431 |
| 19 | dist_prior_low_atr | 1.0289 |
| 20 | sr_levels_between | 0.8842 |
| 21 | prior_bar_toward_vwap | 0.7906 |
| 22 | reversal_quality | 0.7132 |
| 23 | sr_location_score | 0.4773 |
| 24 | dist_open_atr | 0.2802 |
| 25 | dist_prior_vwap_atr | 0.0905 |
| 26 | rsi | 0.0392 |
| 27 | bars_from_vwap | 0.0350 |
| 28 | consecutive_same_side | 0.0319 |
| 29 | minute | 0.0118 |
| 30 | rsi_slope | -0.0012 |
| 31 | dist_swing_low_atr | -0.0560 |
| 32 | minutes_into_session | -0.0646 |
| 33 | open_vs_vwap_atr | -0.1110 |
| 34 | dist_prior_close_atr | -0.1112 |
| 35 | dist_swing_high_atr | -0.2653 |
| 36 | dist_prior_high_atr | -0.5592 |
| 37 | vwap_crosses_today | -0.5851 |
| 38 | vwap_in_day_range | -0.6777 |
| 39 | dist_to_nearest_vwap_band | -1.0422 |
| 40 | bar_reverting | -1.2068 |
| 41 | swept_key_level | -1.5649 |
| 42 | beyond_vwap_2sigma | -1.7066 |
| 43 | rsi_extreme | -1.7136 |
| 44 | close_position | -1.7926 |
| 45 | hour | -2.1383 |
| 46 | dist_day_high_atr | -2.4895 |
| 47 | beyond_vwap_1sigma | -3.8034 |
| 48 | outside_or | -4.0750 |
| 49 | ema20_slope_atr | -4.2720 |
| 50 | vwap_helping | -4.6806 |
| 51 | day_range_atr | -4.7322 |
| 52 | crossed_vwap | -7.1300 |
| 53 | vwap_slope | -7.1954 |
| 54 | dist_or_high_atr | -7.7535 |
| 55 | nearest_swing_atr | -7.9213 |
| 56 | at_session_extreme | -7.9471 |
| 57 | vol_at_extension | -8.8674 |
| 58 | dist_val_atr | -9.0461 |
| 59 | vwap_slope_5 | -9.6165 |
| 60 | outside_value_area | -11.1883 |
| 61 | vol_pct_complete | -12.3386 |
| 62 | bar_range_atr | -19.0967 |

### Per-stop breakdown

| feature                   |   stop_0.75 |      avg |
|:--------------------------|------------:|---------:|
| is_long_setup             |     12.1325 |  12.1325 |
| pct_of_day_range          |     10.8682 |  10.8682 |
| dist_vah_atr              |      8.2103 |   8.2103 |
| vwap_stretch_zscore       |      8.1488 |   8.1488 |
| rel_vol                   |      6.6735 |   6.6735 |
| vwap_width_atr            |      6.3191 |   6.3191 |
| sr_rejection_score        |      4.8656 |   4.8656 |
| dist_or_low_atr           |      3.6032 |   3.6032 |
| extension_speed           |      2.5473 |   2.5473 |
| vol_ratio                 |      2.3246 |   2.3246 |
| momentum_6bar_atr         |      2.1584 |   2.1584 |
| session_phase             |      2.1424 |   2.1424 |
| price_to_vwap_atr         |      2.0765 |   2.0765 |
| momentum_3bar_atr         |      1.8382 |   1.8382 |
| dist_poc_atr              |      1.7142 |   1.7142 |
| dist_day_low_atr          |      1.3728 |   1.3728 |
| sr_trigger_score          |      1.2074 |   1.2074 |
| nearest_sr_atr            |      1.1431 |   1.1431 |
| dist_prior_low_atr        |      1.0289 |   1.0289 |
| sr_levels_between         |      0.8842 |   0.8842 |
| prior_bar_toward_vwap     |      0.7906 |   0.7906 |
| reversal_quality          |      0.7132 |   0.7132 |
| sr_location_score         |      0.4773 |   0.4773 |
| dist_open_atr             |      0.2802 |   0.2802 |
| dist_prior_vwap_atr       |      0.0905 |   0.0905 |
| rsi                       |      0.0392 |   0.0392 |
| bars_from_vwap            |      0.035  |   0.035  |
| consecutive_same_side     |      0.0319 |   0.0319 |
| minute                    |      0.0118 |   0.0118 |
| rsi_slope                 |     -0.0012 |  -0.0012 |
| dist_swing_low_atr        |     -0.056  |  -0.056  |
| minutes_into_session      |     -0.0646 |  -0.0646 |
| open_vs_vwap_atr          |     -0.111  |  -0.111  |
| dist_prior_close_atr      |     -0.1112 |  -0.1112 |
| dist_swing_high_atr       |     -0.2653 |  -0.2653 |
| dist_prior_high_atr       |     -0.5592 |  -0.5592 |
| vwap_crosses_today        |     -0.5851 |  -0.5851 |
| vwap_in_day_range         |     -0.6777 |  -0.6777 |
| dist_to_nearest_vwap_band |     -1.0422 |  -1.0422 |
| bar_reverting             |     -1.2068 |  -1.2068 |
| swept_key_level           |     -1.5649 |  -1.5649 |
| beyond_vwap_2sigma        |     -1.7066 |  -1.7066 |
| rsi_extreme               |     -1.7136 |  -1.7136 |
| close_position            |     -1.7926 |  -1.7926 |
| hour                      |     -2.1383 |  -2.1383 |
| dist_day_high_atr         |     -2.4895 |  -2.4895 |
| beyond_vwap_1sigma        |     -3.8034 |  -3.8034 |
| outside_or                |     -4.075  |  -4.075  |
| ema20_slope_atr           |     -4.272  |  -4.272  |
| vwap_helping              |     -4.6806 |  -4.6806 |
| day_range_atr             |     -4.7322 |  -4.7322 |
| crossed_vwap              |     -7.13   |  -7.13   |
| vwap_slope                |     -7.1954 |  -7.1954 |
| dist_or_high_atr          |     -7.7535 |  -7.7535 |
| nearest_swing_atr         |     -7.9213 |  -7.9213 |
| at_session_extreme        |     -7.9471 |  -7.9471 |
| vol_at_extension          |     -8.8674 |  -8.8674 |
| dist_val_atr              |     -9.0461 |  -9.0461 |
| vwap_slope_5              |     -9.6165 |  -9.6165 |
| outside_value_area        |    -11.1883 | -11.1883 |
| vol_pct_complete          |    -12.3386 | -12.3386 |
| bar_range_atr             |    -19.0967 | -19.0967 |
