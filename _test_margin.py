"""Quick smoke test for ibkr_margin_cost function."""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'scripts')
out = open('_test_margin_out.txt', 'w')
def print(msg='', **kw):
    out.write(str(msg) + '\n')
    out.flush()
from master_pipeline import ibkr_margin_cost, IBKR_MARGIN_RATE_TIERS

# Test 1: No borrowing
c = ibkr_margin_cost(0, 1.0)
print(f"No borrowing: ${c:.4f}")

# Test 2: $500K borrowed for 1 hour
c = ibkr_margin_cost(500_000, 1.0)
print(f"$500K for 1hr: ${c:.2f}")

# Test 3: $1.5M borrowed for 2 hours (crosses tier boundary)  
c = ibkr_margin_cost(1_500_000, 2.0)
print(f"$1.5M for 2hr: ${c:.2f}")

# Test 4: Typical trade: $2.4M notional, $1M capital = $1.4M borrowed, 30 min hold
c = ibkr_margin_cost(1_400_000, 0.5)
print(f"$1.4M for 30min: ${c:.2f}")

# Test 5: Small position no margin needed
c = ibkr_margin_cost(-500_000, 1.0)
print(f"Negative (no margin): ${c:.4f}")

# Show tiers
print("\nMargin rate tiers:")
for ub, rate in IBKR_MARGIN_RATE_TIERS:
    if ub < 1e12:
        print(f"  <= ${ub:>15,.0f}: {rate*100:.2f}%")
    else:
        print(f"  > $200,000,000: {rate*100:.2f}%")

# Manual sanity:
# First $100K at 6.83%, next $900K at 6.33%, next $400K at 6.08%
# Per hour: (100000*0.0683 + 900000*0.0633 + 400000*0.0608) / (360*24) = 
manual = (100_000 * 0.0683 + 900_000 * 0.0633 + 400_000 * 0.0608) / (360 * 24) * 0.5
print(f"\nManual calc $1.4M for 30min: ${manual:.2f}")
print(f"Function result:             ${ibkr_margin_cost(1_400_000, 0.5):.2f}")
assert abs(manual - ibkr_margin_cost(1_400_000, 0.5)) < 0.01, "Mismatch!"
print("PASS - all tests OK")
