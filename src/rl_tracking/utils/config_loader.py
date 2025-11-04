"""Configuration loader utility for training."""
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_nested_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a nested value from a config dictionary using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the key (e.g., "model.lr")
        default: Default value if key is not found
        
    Returns:
        The value at the key path, or default if not found
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value

