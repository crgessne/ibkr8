"""Sweep top-N values for the best stops to find optimal selection size."""
import sys, warnings
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from master_pipeline import (
    load_and_validate_data, calculate_core_indicators, get_feature_columns,
    apply_setup_filter, train_rf_model, DATA_FILE, TEST_YEAR,
    SHARES_PER_TRADE, COMMISSION_PER_SHARE, SLIPPAGE_PER_SHARE,
)
from label_generator import LabelConfig, generate_labels

print("Loading data...")
df = load_and_validate_data(DATA_FILE)
df = calculate_core_indicators(df, verbose=False)
features = get_feature_columns(df)

stops_to_test = [0.6, 0.75, 1.0, 1.25]
config = LabelConfig(stop_atrs=stops_to_test)
df = generate_labels(df, config)

costs_per_trade = 2.0 * (COMMISSION_PER_SHARE + SLIPPAGE_PER_SHARE) * SHARES_PER_TRADE
top_ns = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]

print(f"\n{'='*100}")
print(f"  TOP-N SWEEP: Which selection size maximizes P&L?")
print(f"{'='*100}")
print(f"\n  {'Stop':>5s}  {'TopN':>5s}  {'Trades':>6s}  {'WR':>6s}  {'GrossPnL':>10s}  {'NetPnL':>10s}  {'Avg/Tr':>8s}  {'AvgPred':>8s}")
print(f"  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")

for stop_atr in stops_to_test:
    result = train_rf_model(
        df, stop_atr, features, TEST_YEAR,
        model_kind='regressor', regression_target='net_r',
        setup_filter=True, min_dist_atr=0.5, min_minutes_session=15,
        max_minutes_session=360, min_rr_setup=1.0,
    )
    if result is None:
        print(f"  {stop_atr:>5.2f}  -- no result --")
        continue

    label_col = f"label_s{stop_atr}".replace(".", "_")
    test_idx = result['test_index']
    df_test = df.loc[test_idx].copy()
    scores = np.asarray(result['proba_test'])
    y_raw = result['y_test_raw'].values

    order = np.argsort(scores)[::-1]

    for top_n in top_ns:
        if top_n > len(order):
            continue
        top_idx = order[:top_n]

        y_sel = y_raw[top_idx]
        atr_sel = df_test.iloc[top_idx]['atr'].values
        vwap_dist_sel = df_test.iloc[top_idx]['vwap_width_atr'].values
        scores_sel = scores[top_idx]

        reward = vwap_dist_sel * atr_sel * SHARES_PER_TRADE
        risk = stop_atr * atr_sel * SHARES_PER_TRADE
        gross = np.where(y_sel == 1, reward, -risk)
        net = gross - costs_per_trade

        n = len(y_sel)
        wr = y_sel.mean() * 100
        total_gross = gross.sum()
        total_net = net.sum()
        avg_per = total_net / n if n > 0 else 0
        avg_pred = scores_sel.mean()

        marker = " ***" if total_net == max(
            [net_val for nn in top_ns if nn <= len(order)
             for net_val in [np.where(y_raw[order[:nn]] == 1,
                                       df_test.iloc[order[:nn]]['vwap_width_atr'].values * df_test.iloc[order[:nn]]['atr'].values * SHARES_PER_TRADE,
                                       -stop_atr * df_test.iloc[order[:nn]]['atr'].values * SHARES_PER_TRADE).sum() - costs_per_trade * nn]]
        ) else ""

        print(f"  {stop_atr:>5.2f}  {top_n:>5d}  {n:>6d}  {wr:>5.1f}%  ${total_gross:>9,.0f}  ${total_net:>9,.0f}  ${avg_per:>7,.0f}  {avg_pred:>7.3f}{marker}")

    print()
