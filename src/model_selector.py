"""
Model Selection Utilities

Helper functions to find and select the best model based on various criteria.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from src.model_persistence import list_saved_models, load_model


def find_models_by_stop_atr(models_dir: str = "models", stop_atr: float = None,
                            model_type: str = None) -> List[Dict]:
    """
    Find all models matching a specific stop ATR and optionally model type.
    
    Args:
        models_dir: Directory containing saved models
        stop_atr: Stop ATR to filter by (e.g., 0.5). If None, returns all models.
        model_type: Filter by model_type metadata (e.g., 'PnLModelWrapper', 'MLPClassifier').
                    If None, no type filtering is applied.
    
    Returns:
        List of model info dictionaries matching the criteria
    """
    all_models = list_saved_models(models_dir)
    
    if stop_atr is not None:
        all_models = [m for m in all_models if abs(m['stop_atr'] - stop_atr) < 0.001]
    
    if model_type is not None:
        all_models = [m for m in all_models if m.get('model_type') == model_type]
    
    return all_models


def get_latest_model_by_stop(models_dir: str = "models", stop_atr: float = 0.5,
                             model_type: str = None) -> Optional[str]:
    """
    Get the most recently trained model for a specific stop width.
    
    Args:
        models_dir: Directory containing saved models
        stop_atr: Stop ATR to find (e.g., 0.5)
        model_type: Filter by model_type metadata (e.g., 'PnLModelWrapper').
    
    Returns:
        Path to the latest model, or None if not found
    """
    models = find_models_by_stop_atr(models_dir, stop_atr, model_type=model_type)
    
    if len(models) == 0:
        return None
    
    # Sort by saved_at timestamp (most recent first)
    models_sorted = sorted(models, key=lambda m: m['saved_at'], reverse=True)
    
    return models_sorted[0]['filepath']


def get_best_model(models_dir: str = "models", criterion: str = "pnl") -> Optional[Tuple[str, Dict]]:
    """
    Find the best performing model across all stop widths.
    
    Args:
        models_dir: Directory containing saved models
        criterion: Metric to optimize ('pnl', 'ev', 'winrate', 'trades')
    
    Returns:
        Tuple of (filepath, metadata) for best model, or None if no models found
    """
    all_models = list_saved_models(models_dir)
    
    if len(all_models) == 0:
        return None
    
    # Load each model's metadata and extract performance metrics
    models_with_metrics = []
    
    for model_info in all_models:
        try:
            _, metadata = load_model(model_info['filepath'])
            
            # Extract test performance metrics at RF threshold 0.5
            test_metrics = metadata.get('test_stats', {}).get('rf_threshold_0.5', {})
            
            models_with_metrics.append({
                'filepath': model_info['filepath'],
                'stop_atr': metadata['stop_atr'],
                'pnl': test_metrics.get('total_net_pnl', 0),
                'ev': test_metrics.get('ev', 0),
                'winrate': test_metrics.get('win_rate', 0),
                'trades': test_metrics.get('n_trades', 0),
                'metadata': metadata
            })
        except Exception as e:
            print(f"⚠ Could not load {model_info['filepath']}: {e}")
            continue
    
    if len(models_with_metrics) == 0:
        return None
    
    # Sort by chosen criterion
    criterion_map = {
        'pnl': lambda m: m['pnl'],
        'ev': lambda m: m['ev'],
        'winrate': lambda m: m['winrate'],
        'trades': lambda m: m['trades']
    }
    
    if criterion not in criterion_map:
        raise ValueError(f"Unknown criterion '{criterion}'. Use: pnl, ev, winrate, trades")
    
    best = max(models_with_metrics, key=criterion_map[criterion])
    
    return best['filepath'], best['metadata']


def list_all_models_summary(models_dir: str = "models") -> None:
    """
    Print a summary table of all available models.
    
    Args:
        models_dir: Directory containing saved models
    """
    all_models = list_saved_models(models_dir)
    
    if len(all_models) == 0:
        print("No models found.")
        return
    
    print(f"\n{'='*100}")
    print(f"AVAILABLE MODELS ({len(all_models)} total)")
    print(f"{'='*100}\n")
    
    # Group by stop ATR
    by_stop = {}
    for model_info in all_models:
        stop = model_info['stop_atr']
        if stop not in by_stop:
            by_stop[stop] = []
        by_stop[stop].append(model_info)
    
    for stop_atr in sorted(by_stop.keys()):
        models = by_stop[stop_atr]
        print(f"\n📊 Stop {stop_atr:.2f} ATR ({len(models)} model{'s' if len(models) > 1 else ''})")
        print("-" * 100)
        
        for m in sorted(models, key=lambda x: x['saved_at'], reverse=True):
            filepath = Path(m['filepath'])
            print(f"  • {filepath.name}")
            print(f"    Saved: {m['saved_at'][:19]}")
            print(f"    Features: {m['features_count']}")
            
            # Try to load and show performance
            try:
                _, metadata = load_model(m['filepath'])
                test_metrics = metadata.get('test_stats', {}).get('rf_threshold_0.5', {})
                print(f"    Performance: WR={test_metrics.get('win_rate', 0)*100:.1f}%, "
                      f"EV={test_metrics.get('ev', 0):+.3f}, "
                      f"P&L=${test_metrics.get('total_net_pnl', 0):,.0f}, "
                      f"Trades={test_metrics.get('n_trades', 0):,}")
            except:
                pass
            print()


def load_model_for_stop(stop_atr: float, models_dir: str = "models", latest: bool = True,
                        model_type: str = None):
    """
    Convenient function to load a model for a specific stop width.
    
    Args:
        stop_atr: Stop ATR to load (e.g., 0.5)
        models_dir: Directory containing saved models
        latest: If True, loads most recent model. If False, loads best performing.
        model_type: Filter by model_type metadata (e.g., 'PnLModelWrapper' for nn_pnl,
                    'MLPClassifier' for sklearn NN, 'RandomForestClassifier' for RF).
                    If None, no type filtering is applied.
    
    Returns:
        Tuple of (model, metadata)
    
    Example:
        model, metadata = load_model_for_stop(0.5, model_type='PnLModelWrapper')
        prob = model.predict_proba(features)[0, 1]
    """
    if latest:
        filepath = get_latest_model_by_stop(models_dir, stop_atr, model_type=model_type)
        if filepath is None:
            type_msg = f" with model_type='{model_type}'" if model_type else ""
            raise FileNotFoundError(f"No model found for stop {stop_atr} ATR{type_msg} in {models_dir}")
        return load_model(filepath)
    else:
        # Load best performing model for this stop
        models = find_models_by_stop_atr(models_dir, stop_atr, model_type=model_type)
        if len(models) == 0:
            type_msg = f" with model_type='{model_type}'" if model_type else ""
            raise FileNotFoundError(f"No model found for stop {stop_atr} ATR{type_msg} in {models_dir}")
        
        best_pnl = None
        best_filepath = None
        
        for m in models:
            _, metadata = load_model(m['filepath'])
            test_metrics = metadata.get('test_stats', {}).get('rf_threshold_0.5', {})
            pnl = test_metrics.get('total_net_pnl', 0)
            
            if best_pnl is None or pnl > best_pnl:
                best_pnl = pnl
                best_filepath = m['filepath']
        
        return load_model(best_filepath)


if __name__ == "__main__":
    # Example usage
    print("Model Selection Utilities\n")
    
    # List all models
    list_all_models_summary("models")
    
    # Find models by stop width
    print("\n" + "="*100)
    print("FINDING MODELS FOR SPECIFIC STOP WIDTH")
    print("="*100)
    
    for stop in [0.25, 0.5, 0.75, 1.0]:
        models = find_models_by_stop_atr("models", stop)
        print(f"\nStop {stop:.2f} ATR: {len(models)} model(s) found")
    
    # Get latest model
    print("\n" + "="*100)
    print("LOADING LATEST MODEL FOR STOP 0.5 ATR")
    print("="*100)
    
    try:
        latest_path = get_latest_model_by_stop("models", 0.5)
        if latest_path:
            print(f"\nLatest model: {Path(latest_path).name}")
            model, metadata = load_model(latest_path)
            print(f"Stop: {metadata['stop_atr']} ATR")
            print(f"R:R: {metadata['rr']:.2f}")
            print(f"Features: {len(metadata['features'])}")
        else:
            print("\nNo model found for stop 0.5 ATR")
    except Exception as e:
        print(f"\nError: {e}")
    
    # Get best model overall
    print("\n" + "="*100)
    print("FINDING BEST MODEL (BY P&L)")
    print("="*100)
    
    try:
        best_filepath, best_metadata = get_best_model("models", criterion="pnl")
        if best_filepath:
            print(f"\nBest model: {Path(best_filepath).name}")
            print(f"Stop: {best_metadata['stop_atr']} ATR")
            test_metrics = best_metadata['test_stats']['rf_threshold_0.5']
            print(f"P&L: ${test_metrics['total_net_pnl']:,.0f}")
            print(f"Win Rate: {test_metrics['win_rate']*100:.1f}%")
            print(f"EV: {test_metrics['ev']:+.3f}")
        else:
            print("\nNo models found")
    except Exception as e:
        print(f"\nError: {e}")
