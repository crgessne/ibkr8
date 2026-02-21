"""Check what features the model actually sees."""
import traceback
try:
    import sys, warnings
    warnings.filterwarnings('ignore')
    sys.path.insert(0, 'src')
    sys.path.insert(0, 'scripts')

    import pandas as pd
    import numpy as np
    print("Imports OK", flush=True)
    
    from master_pipeline import calculate_core_indicators, get_feature_columns, DEFAULT_STOP_ATR
    print("Pipeline imports OK", flush=True)

    df = pd.read_csv('data/tsla_5min_10years.csv')
    print(f"Loaded {len(df)} bars", flush=True)
    
    df['datetime'] = pd.to_datetime(df['time'], utc=True)
    df['date'] = df['datetime'].dt.date
    print("Computing indicators...", flush=True)
    
    df = calculate_core_indicators(df, verbose=False)
    print(f"Indicators done. Columns: {len(df.columns)}", flush=True)
    
    feats = get_feature_columns(df)
    print(f"\nTotal features model sees: {len(feats)}", flush=True)

    # RR features
    rr_feats = [f for f in feats if 'rr' in f.lower() or 'setup_rr' in f or 'vwap_width' in f]
    print(f"\nRR features IN model ({len(rr_feats)}):", flush=True)
    for f in sorted(rr_feats):
        print(f"  + {f}", flush=True)

    # Overextension features  
    ext_feats = [f for f in feats if any(k in f for k in ['stretch', 'sigma', 'extension', 'rsi', 'bb_z', 'stoch'])]
    print(f"\nOverextension features IN model ({len(ext_feats)}):", flush=True)
    for f in sorted(ext_feats):
        print(f"  + {f}", flush=True)

    # What rr_ columns exist in the DataFrame?
    all_rr = [c for c in df.columns if c.startswith('rr_')]
    in_feats = [c for c in all_rr if c in feats]
    excluded = [c for c in all_rr if c not in feats]
    print(f"\nrr_* columns in DataFrame: {all_rr}", flush=True)
    print(f"rr_* columns IN features:  {in_feats}", flush=True)
    print(f"rr_* columns EXCLUDED:     {excluded}", flush=True)

    # Key individual checks
    for key in ['setup_rr', 'vwap_width_atr', 'price_to_vwap_atr', 'vwap_stretch_zscore',
                'vwap_sigma', 'bb_z_score', 'rsi', 'rsi_extreme', 'stoch_k']:
        status = "IN MODEL" if key in feats else "EXCLUDED"
        print(f"  {key:30s} -> {status}", flush=True)

    # Show all features
    print(f"\nALL FEATURES:", flush=True)
    for i, f in enumerate(feats, 1):
        print(f"  {i:3d}. {f}", flush=True)

except Exception as e:
    traceback.print_exc()
    print(f"ERROR: {e}", flush=True)
