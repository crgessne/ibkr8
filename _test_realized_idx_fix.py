"""Quick unit test: verify realized_net_pnl index alignment fix in train_rf_model.

This simulates the exact index flow that was broken (KeyError) and verifies
the fix (using df_valid_sim['bar_index'] instead of df_valid_sim.index).
"""
import pandas as pd
import numpy as np

def test_index_alignment():
    # Simulate df_valid with original sparse index labels (like real data)
    original_labels = [68158, 118408, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000]
    df_valid = pd.DataFrame({
        'val': range(10),
        'datetime': pd.date_range('2020-01-01', periods=10, freq='h'),
    }, index=original_labels)

    X = pd.DataFrame({'feat1': np.random.randn(10), 'feat2': np.random.randn(10)}, index=original_labels)
    y_raw = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], index=original_labels)

    # Step 1: reset_index -> 'bar_index' has original labels, new index is 0..N
    df_valid_sim = df_valid.reset_index().rename(columns={'index': 'bar_index'})
    assert list(df_valid_sim.index) == list(range(10)), "After reset_index, index should be 0..N"
    assert list(df_valid_sim['bar_index']) == original_labels, "bar_index should hold original labels"

    # Step 2: Downsample
    df_valid_sim = df_valid_sim.sample(n=5, random_state=42)

    # OLD (BROKEN): keep_idx = df_valid_sim.index.to_numpy()
    # This gives e.g. [6, 3, 7, 8, 0] - positional ints from 0..N, NOT original labels!
    broken_idx = df_valid_sim.index.to_numpy()

    # NEW (FIXED): keep_idx = df_valid_sim['bar_index'].astype(int).to_numpy()
    # This gives e.g. [400000, 250000, 450000, 500000, 68158] - original labels!
    fixed_idx = df_valid_sim['bar_index'].astype(int).to_numpy()

    print(f"Original labels: {original_labels}")
    print(f"Broken idx (0..N positional): {broken_idx}")
    print(f"Fixed idx (original labels):  {fixed_idx}")

    # Verify broken path fails
    realized_vals = [10.0, 20.0, 30.0, 40.0, 50.0]

    try:
        broken_series = pd.Series(realized_vals, index=broken_idx)
        X.loc[broken_series.index]  # This SHOULD fail - broken_idx has ints like 6,3,7... not in X.index
        print("ERROR: broken path did NOT raise KeyError (unexpected)")
        return False
    except KeyError:
        print("OK: broken path correctly raises KeyError (this was the bug)")

    # Verify fixed path works
    fixed_series = pd.Series(realized_vals, index=fixed_idx)
    try:
        X_aligned = X.loc[fixed_series.index]
        y_aligned = y_raw.loc[fixed_series.index]
        df_aligned = df_valid.loc[fixed_series.index]
        assert len(X_aligned) == 5
        assert len(y_aligned) == 5
        assert len(df_aligned) == 5
        assert set(X_aligned.index) == set(fixed_idx)
        print("OK: fixed path aligns correctly")
    except Exception as e:
        print(f"ERROR: fixed path failed: {e}")
        return False

    print("\nALL TESTS PASSED - index fix is correct")
    return True

if __name__ == '__main__':
    success = test_index_alignment()
    exit(0 if success else 1)
