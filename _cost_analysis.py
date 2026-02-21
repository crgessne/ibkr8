"""Analyze NN training signal economics at different slippage levels."""
import numpy as np

SHARES = 100
COMM = 0.005
ATR_TSLA_AVG = 2.5  # avg 5-min ATR for TSLA

print("=== Cost vs reward at different slippages and vwap_width_atr levels ===")
print()
for slip in [0.01, 0.05, 0.10, 0.18]:
    cost_total = 2 * (COMM + slip) * SHARES
    min_vwap_atr_for_breakeven = cost_total / (ATR_TSLA_AVG * SHARES)
    print(f"slippage={slip:.2f}:")
    print(f"  round-trip cost = ${cost_total:.2f} (100 shares)")
    print(f"  breakeven vwap_width_atr = {min_vwap_atr_for_breakeven:.4f} ATR")
    print(f"  = ${min_vwap_atr_for_breakeven * ATR_TSLA_AVG:.4f}/share minimum target")
    for vwap_atr in [0.5, 0.75, 1.0, 1.5, 2.0]:
        reward = vwap_atr * ATR_TSLA_AVG * SHARES
        net = reward - cost_total
        label = "WIN " if net > 0 else "LOSS"
        print(f"    vwap_width_atr={vwap_atr:.2f}: reward=${reward:.1f}, net=${net:.1f}  [{label}]")
    print()

print()
print("=== KEY ISSUE: with SHARES=100, even slippage=0.01 only costs $3 round-trip ===")
print("=== So vwap_width_atr=0.5 ATR gives reward=$125 >> cost=$3 ===")
print("=== The NN sees almost ALL setups as net-positive and cannot learn to prefer wider ones ===")
print()
print("=== FIX: use slippage that makes the cost-to-reward ratio realistic ===")
print("=== Or equivalently: use a per-share cost that equals the observed live slippage ===")
print()

# The real-world ratio we want: 
# Live observed: avg target_dist ~$1.10, avg slippage ~$0.18/share
# So slip/target_dist ratio = 0.18/1.10 = 16%
# Round-trip cost/reward = 2*0.18 / 1.10 = 33% — this is meaningful signal!
# 
# But with SHARES=100 and ATR=$2.5:
# At vwap_width_atr=0.5: reward=$125, need slip=$0.625/share to eat 50%
# 
# Solution: use NORMALIZED cost/reward ratio in the pnl signal
# i.e., normalize net_pnl by reward so the gradient is about RATIO not absolute dollars
print("=== NORMALIZED approach: use net_r (net R multiple) as training signal ===")
print("=== net_r = (reward - cost) / risk ===")
print("=== This is dimensionless and shows the TRUE cost-to-reward ratio ===")
print()
for slip in [0.01, 0.05, 0.10, 0.18]:
    cost_per_share = 2 * (COMM + slip)
    print(f"slippage={slip:.2f} (cost={cost_per_share:.3f}/share):")
    for stop_atr in [0.75]:
        risk_per_share = stop_atr * ATR_TSLA_AVG
        for vwap_atr in [0.5, 0.75, 1.0, 1.5, 2.0]:
            reward_per_share = vwap_atr * ATR_TSLA_AVG
            net_per_share = reward_per_share - cost_per_share
            net_r = net_per_share / risk_per_share
            label = "WIN " if net_r > 0 else "LOSS"
            print(f"    vwap_atr={vwap_atr:.2f}: net_r={net_r:.3f}  [{label}]")
    print()
