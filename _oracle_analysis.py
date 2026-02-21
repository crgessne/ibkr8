"""Oracle analysis: what's the theoretical max P&L with perfect foresight?

Shows:
1. Oracle (take every winning trade, skip every loser) — upper bound
2. Blind (take every setup-filtered bar) — baseline without RF
3. RF top-N — what the model actually achieves
4. RF lift = how much value the RF adds over blind
"""
import sys, warnings, io

# Tee stdout to file
_out_file = open("_oracle_out.txt", "w", encoding="utf-8")
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()
sys.stdout = Tee(sys.__stdout__, _out_file)

sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from master_pipeline import (
    load_and_validate_data, calculate_core_indicators, get_feature_columns,
    apply_setup_filter, train_rf_model, DATA_FILE, TEST_YEAR,
    SHARES_PER_TRADE, COMMISSION_PER_SHARE, SLIPPAGE_PER_SHARE, STOP_ATRS,
)
from label_generator import LabelConfig, generate_labels

print("Loading data...")
df = load_and_validate_data(DATA_FILE)
df = calculate_core_indicators(df, verbose=False)
features = get_feature_columns(df)

all_stops = [0.25, 0.35, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5]
config = LabelConfig(stop_atrs=all_stops)
df = generate_labels(df, config)

costs_per_trade = 2.0 * (COMMISSION_PER_SHARE + SLIPPAGE_PER_SHARE) * SHARES_PER_TRADE

print(f"\n{'='*120}")
print(f"  ORACLE ANALYSIS: Setup-Filtered Test Set (>= {TEST_YEAR})")
print(f"  Costs/trade: ${costs_per_trade:.2f} | Shares: {SHARES_PER_TRADE}")
print(f"{'='*120}")

header = (
    f"  {'Stop':>5s}  {'Bars':>6s}  "
    f"{'OracleWR':>8s}  {'BlindWR':>7s}  "
    f"{'OracleNet':>11s}  {'BlindNet':>11s}  {'RF500Net':>11s}  "
    f"{'BlindAvg':>9s}  {'RF500Avg':>9s}  "
    f"{'RFLift$':>9s}  {'RF/Blind':>8s}"
)
print(header)
print(f"  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*11}  {'-'*11}  {'-'*11}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*8}")

for stop_atr in all_stops:
    label_col = f"label_s{stop_atr}".replace(".", "_")

    # Get setup-filtered test bars
    valid = df[label_col].notna()
    df_v = df[valid].copy()
    mask = apply_setup_filter(df_v, stop_atr=stop_atr, min_dist_atr=0.5,
                               min_minutes_session=15, max_minutes_session=360, min_rr_setup=1.0)
    df_s = df_v[mask].copy()
    df_s['year'] = pd.to_datetime(df_s['datetime']).dt.year
    df_test = df_s[df_s['year'] >= TEST_YEAR].copy()

    n_bars = len(df_test)
    if n_bars == 0:
        print(f"  {stop_atr:>5.2f}  {0:>6d}  -- no bars --")
        continue

    y = df_test[label_col].astype(int).values
    atr = df_test['atr'].values
    vwap_dist = df_test['vwap_width_atr'].values

    reward = vwap_dist * atr * SHARES_PER_TRADE
    risk = stop_atr * atr * SHARES_PER_TRADE
    gross = np.where(y == 1, reward, -risk)
    net = gross - costs_per_trade

    # Oracle: only take winners
    oracle_wr = y.mean()
    oracle_net = net[y == 1].sum()  # only winning trades, still pay costs
    oracle_n = y.sum()

    # Blind: take every bar
    blind_wr = y.mean()
    blind_net = net.sum()
    blind_avg = blind_net / n_bars

    # RF top-500
    result = train_rf_model(
        df, stop_atr, features, TEST_YEAR,
        model_kind='regressor', regression_target='net_r',
        setup_filter=True, min_dist_atr=0.5, min_minutes_session=15,
        max_minutes_session=360, min_rr_setup=1.0,
    )

    if result is not None:
        scores = np.asarray(result['proba_test'])
        y_raw = result['y_test_raw'].values
        test_idx = result['test_index']
        df_rf_test = df.loc[test_idx].copy()

        order = np.argsort(scores)[::-1]
        top_n = min(500, len(order))
        top_idx = order[:top_n]

        y_sel = y_raw[top_idx]
        atr_sel = df_rf_test.iloc[top_idx]['atr'].values
        vwap_dist_sel = df_rf_test.iloc[top_idx]['vwap_width_atr'].values
        reward_sel = vwap_dist_sel * atr_sel * SHARES_PER_TRADE
        risk_sel = stop_atr * atr_sel * SHARES_PER_TRADE
        gross_sel = np.where(y_sel == 1, reward_sel, -risk_sel)
        net_sel = gross_sel - costs_per_trade

        rf_net = net_sel.sum()
        rf_avg = rf_net / top_n
        rf_wr = y_sel.mean()
        rf_lift = rf_net - (blind_avg * top_n)  # vs taking 500 random setup bars
        rf_ratio = rf_avg / blind_avg if blind_avg != 0 else float('nan')
    else:
        rf_net = float('nan')
        rf_avg = float('nan')
        rf_lift = float('nan')
        rf_ratio = float('nan')

    print(
        f"  {stop_atr:>5.2f}  {n_bars:>6,}  "
        f"{oracle_wr*100:>7.1f}%  {blind_wr*100:>6.1f}%  "
        f"${oracle_net:>10,.0f}  ${blind_net:>10,.0f}  ${rf_net:>10,.0f}  "
        f"${blind_avg:>8,.1f}  ${rf_avg:>8,.1f}  "
        f"${rf_lift:>8,.0f}  {rf_ratio:>7.2f}x"
    )

print()
print("Legend:")
print("  OracleWR  = win rate on setup-filtered bars (same as BlindWR — it's the base rate)")
print("  OracleNet = net P&L if you ONLY took winning trades (perfect foresight, still pay costs)")
print("  BlindNet  = net P&L taking EVERY setup-filtered bar (no model)")
print("  RF500Net  = net P&L from RF regressor selecting top-500 predicted payoff")
print("  RFLift$   = RF500Net minus (BlindAvg * 500) — value added by RF selection")
print("  RF/Blind  = ratio of RF avg/trade to Blind avg/trade — >1 means RF helps")

# ── Also show blind P&L by stop for the "no model" baseline ──
print(f"\n{'='*80}")
print(f"  BLIND BASELINE (no RF, take every setup bar)")
print(f"{'='*80}")
print(f"  {'Stop':>5s}  {'Bars':>6s}  {'WR':>6s}  {'AvgWin$':>9s}  {'AvgLoss$':>9s}  {'NetPnL':>11s}  {'Avg/Trade':>10s}")
print(f"  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*11}  {'-'*10}")

for stop_atr in all_stops:
    label_col = f"label_s{stop_atr}".replace(".", "_")
    valid = df[label_col].notna()
    df_v = df[valid].copy()
    mask = apply_setup_filter(df_v, stop_atr=stop_atr, min_dist_atr=0.5,
                               min_minutes_session=15, max_minutes_session=360, min_rr_setup=1.0)
    df_s = df_v[mask].copy()
    df_s['year'] = pd.to_datetime(df_s['datetime']).dt.year
    df_test = df_s[df_s['year'] >= TEST_YEAR].copy()
    n = len(df_test)
    if n == 0:
        continue

    y = df_test[label_col].astype(int).values
    atr = df_test['atr'].values
    vwap_dist = df_test['vwap_width_atr'].values
    reward = vwap_dist * atr * SHARES_PER_TRADE
    risk = stop_atr * atr * SHARES_PER_TRADE
    gross = np.where(y == 1, reward, -risk)
    net = gross - costs_per_trade

    wr = y.mean()
    avg_win = net[y == 1].mean() if y.sum() > 0 else 0
    avg_loss = net[y == 0].mean() if (n - y.sum()) > 0 else 0
    total_net = net.sum()
    avg_per = total_net / n

    print(
        f"  {stop_atr:>5.2f}  {n:>6,}  {wr*100:>5.1f}%  "
        f"${avg_win:>8,.0f}  ${avg_loss:>8,.0f}  "
        f"${total_net:>10,.0f}  ${avg_per:>9,.1f}"
    )
