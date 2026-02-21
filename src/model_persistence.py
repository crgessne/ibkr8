"""
Model Persistence for RF VWAP Reversion Strategy

This module provides functions to save and load trained RandomForest models
along with their metadata, feature configurations, and training statistics.

Usage:
    # Saving a model
    save_model(
        model=rf_classifier,
        filepath="models/rf_vwap_stop0.5.pkl",
        metadata={
            'stop_atr': 0.5,
            'features': feature_list,
            'train_date_range': ('2020-01-01', '2023-12-31'),
            'test_metrics': {...}
        }
    )
    
    # Loading a model
    model, metadata = load_model("models/rf_vwap_stop0.5.pkl")
"""

import pickle
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import json


class _ModelUnpickler(pickle.Unpickler):
    """Custom unpickler that resolves PnLNet/PnLModelWrapper from nn_pnl_models.

    Models saved from master_pipeline.py (as __main__) store class references as
    '__main__.PnLModelWrapper' / '__main__.PnLNet'. When loading from a different
    script, pickle can't find them. This unpickler redirects to the shared module.
    Also handles 'scripts.master_pipeline' references if models were saved that way.
    """

    _REDIRECT_CLASSES = {'PnLNet', 'PnLModelWrapper'}
    _REDIRECT_MODULES = {'__main__', 'scripts.master_pipeline', 'master_pipeline'}

    def find_class(self, module: str, name: str):
        if name in self._REDIRECT_CLASSES and module in self._REDIRECT_MODULES:
            # Import from the shared nn_pnl_models module
            try:
                from src.nn_pnl_models import PnLNet, PnLModelWrapper
            except ImportError:
                from nn_pnl_models import PnLNet, PnLModelWrapper
            return {'PnLNet': PnLNet, 'PnLModelWrapper': PnLModelWrapper}[name]
        return super().find_class(module, name)


def save_model(
    model: Any,
    filepath: str,
    metadata: Dict[str, Any],
) -> Path:
    """
    Save a trained model with metadata to disk.
    
    Args:
        model: Trained sklearn model (e.g., RandomForestClassifier)
        filepath: Path where model will be saved (e.g., "models/rf_stop0.5.pkl")
        metadata: Dictionary containing:
            - stop_atr: Stop width in ATR multiples
            - features: List of feature column names
            - rr: Risk:reward ratio
            - train_stats: Training set statistics (n_samples, date_range, etc.)
            - test_stats: Test set statistics and metrics
            - rf_params: RandomForest hyperparameters
            - Any other relevant configuration
    
    Returns:
        Path object of saved file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Add save timestamp to metadata
    metadata['saved_at'] = datetime.now().isoformat()
    metadata['model_type'] = type(model).__name__
    
    # Package model and metadata together
    package = {
        'model': model,
        'metadata': metadata,
        'version': '1.0'  # For future compatibility
    }
      # Save using pickle (supports sklearn models)
    with open(filepath, 'wb') as f:
        pickle.dump(package, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Saved model to: {filepath}")
    print(f"  - Model type: {metadata['model_type']}")
    print(f"  - Stop ATR: {metadata.get('stop_atr', 'N/A')}")
    print(f"  - Features: {len(metadata.get('features', []))}")
    
    # Also save a human-readable metadata file
    metadata_path = filepath.with_suffix('.json')
    metadata_serializable = {k: v for k, v in metadata.items() if k != 'model'}
    # Convert non-serializable objects to strings
    for key, value in metadata_serializable.items():
        if hasattr(value, 'tolist'):  # numpy arrays
            metadata_serializable[key] = value.tolist()
        elif not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            metadata_serializable[key] = str(value)
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_serializable, f, indent=2)    
    print(f"Saved metadata to: {metadata_path}")
    
    return filepath


def load_model(filepath: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a trained model and its metadata from disk.
    
    Args:
        filepath: Path to saved model file
    
    Returns:
        Tuple of (model, metadata)
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model file is corrupted or incompatible
    """
    filepath = Path(filepath)
    
    if not filepath.exists():        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            package = _ModelUnpickler(f).load()
        # Validate package structure
        if not isinstance(package, dict):
            raise ValueError("Invalid model file format")
        
        if 'model' not in package or 'metadata' not in package:
            raise ValueError("Model file missing required components")
        
        model = package['model']
        metadata = package['metadata']
        
        print(f"Loaded model from: {filepath}")
        print(f"  - Model type: {metadata.get('model_type', 'Unknown')}")
        print(f"  - Saved at: {metadata.get('saved_at', 'Unknown')}")
        print(f"  - Stop ATR: {metadata.get('stop_atr', 'N/A')}")
        print(f"  - Features: {len(metadata.get('features', []))}")
        
        return model, metadata
    
    except Exception as e:
        raise ValueError(f"Failed to load model from {filepath}: {str(e)}")


def list_saved_models(models_dir: str = "models") -> List[Dict[str, Any]]:
    """
    List all saved models in a directory.
    
    Args:
        models_dir: Directory containing saved models
    
    Returns:
        List of dictionaries with model information
    """
    models_dir = Path(models_dir)
    
    if not models_dir.exists():
        return []
    
    models_info = []
    
    for model_file in models_dir.glob("*.pkl"):
        try:
            _, metadata = load_model(model_file)
            models_info.append({
                'filepath': str(model_file),
                'stop_atr': metadata.get('stop_atr'),
                'saved_at': metadata.get('saved_at'),
                'features_count': len(metadata.get('features', [])),
                'model_type': metadata.get('model_type'),
            })
        except Exception as e:
            print(f"Warning: Could not load {model_file}: {e}")
    
    return models_info


def get_model_summary(filepath: str) -> Dict[str, Any]:
    """
    Get a summary of a saved model without fully loading it.
    
    Args:
        filepath: Path to model file
    
    Returns:
        Dictionary with model summary information
    """
    # Try to load just the metadata JSON file first
    json_path = Path(filepath).with_suffix('.json')
    
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Fall back to loading the full pickle file
    _, metadata = load_model(filepath)
    return metadata


def validate_model_compatibility(
    model_metadata: Dict[str, Any],
    required_features: List[str],
    stop_atr: Optional[float] = None
) -> Tuple[bool, List[str]]:
    """
    Validate that a saved model is compatible with current requirements.
    
    Args:
        model_metadata: Metadata from loaded model
        required_features: Features expected in current data
        stop_atr: Expected stop ATR (optional)
    
    Returns:
        Tuple of (is_compatible, list_of_issues)
    """
    issues = []
    
    # Check features
    model_features = model_metadata.get('features', [])
    if set(model_features) != set(required_features):
        missing = set(required_features) - set(model_features)
        extra = set(model_features) - set(required_features)
        if missing:
            issues.append(f"Missing features: {missing}")
        if extra:
            issues.append(f"Extra features in model: {extra}")
    
    # Check stop ATR
    if stop_atr is not None:
        model_stop = model_metadata.get('stop_atr')
        if model_stop != stop_atr:
            issues.append(f"Stop ATR mismatch: model={model_stop}, expected={stop_atr}")
    
    is_compatible = len(issues) == 0
    return is_compatible, issues


if __name__ == "__main__":
    # Example usage and testing
    print("Model Persistence Module")
    print("=" * 80)
    print("\nThis module provides save/load functionality for trained models.")
    print("\nExample usage:")
    print("""
from model_persistence import save_model, load_model

# Save a trained model
save_model(
    model=my_rf_model,
    filepath="models/rf_vwap_stop0.5.pkl",
    metadata={
        'stop_atr': 0.5,
        'features': ['vwap_pct', 'rsi', 'bb_width', ...],
        'rr': 2.5,
        'train_stats': {
            'n_samples': 50000,
            'date_range': ('2020-01-01', '2023-12-31'),
            'win_rate': 0.65
        }
    }
)

# Load a saved model
model, metadata = load_model("models/rf_vwap_stop0.5.pkl")
predictions = model.predict_proba(X_new)[:, 1]
""")
