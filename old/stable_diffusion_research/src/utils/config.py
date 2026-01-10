"""
Configuration loading and management.

Supports:
- YAML configuration files
- Configuration inheritance (_base_ key)
- Command-line overrides
- Environment variable expansion
"""

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load a configuration file with inheritance support.
    
    Configuration files can inherit from base configs using the _base_ key:
    ```yaml
    _base_: "../base.yaml"
    
    training:
        learning_rate: 1e-4
    ```
    
    Args:
        config_path: Path to configuration file
        overrides: Optional dictionary of overrides
    
    Returns:
        Merged configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Handle inheritance
    if "_base_" in config:
        base_path = config_path.parent / config["_base_"]
        base_config = load_config(base_path)
        
        # Remove _base_ from current config
        del config["_base_"]
        
        # Merge configs (current overrides base)
        config = merge_configs(base_config, config)
    
    # Apply overrides
    if overrides:
        config = merge_configs(config, overrides)
    
    # Expand environment variables
    config = expand_env_vars(config)
    
    return config


def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.
    
    Override values take precedence over base values.
    
    Args:
        base: Base configuration
        override: Override configuration
    
    Returns:
        Merged configuration
    """
    result = deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = merge_configs(result[key], value)
        else:
            # Override value
            result[key] = deepcopy(value)
    
    return result


def expand_env_vars(config: Any) -> Any:
    """
    Recursively expand environment variables in config values.
    
    Supports ${VAR} and ${VAR:-default} syntax.
    
    Args:
        config: Configuration (dict, list, or value)
    
    Returns:
        Configuration with expanded env vars
    """
    if isinstance(config, dict):
        return {k: expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(v) for v in config]
    elif isinstance(config, str):
        # Expand ${VAR} and ${VAR:-default}
        import re
        
        def replace_env(match):
            var = match.group(1)
            if ":-" in var:
                var_name, default = var.split(":-", 1)
                return os.environ.get(var_name, default)
            else:
                return os.environ.get(var, match.group(0))
        
        return re.sub(r'\$\{([^}]+)\}', replace_env, config)
    else:
        return config


def save_config(
    config: Dict[str, Any],
    save_path: Union[str, Path],
):
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save to
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def config_to_flat_dict(
    config: Dict[str, Any],
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Flatten a nested config dict for logging.
    
    Example:
        {"training": {"lr": 1e-4}} -> {"training.lr": 1e-4}
    
    Args:
        config: Nested configuration dictionary
        prefix: Prefix for keys
    
    Returns:
        Flattened dictionary
    """
    result = {}
    
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            result.update(config_to_flat_dict(value, full_key))
        else:
            result[full_key] = value
    
    return result


def parse_cli_overrides(args: List[str]) -> Dict[str, Any]:
    """
    Parse command-line overrides in key=value format.
    
    Supports nested keys with dot notation: training.lr=1e-4
    
    Args:
        args: List of key=value strings
    
    Returns:
        Nested dictionary of overrides
    """
    overrides = {}
    
    for arg in args:
        if "=" not in arg:
            continue
        
        key, value = arg.split("=", 1)
        
        # Parse value
        value = parse_value(value)
        
        # Handle nested keys
        keys = key.split(".")
        current = overrides
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    return overrides


def parse_value(value: str) -> Any:
    """
    Parse a string value to the appropriate Python type.
    
    Args:
        value: String value
    
    Returns:
        Parsed value (int, float, bool, or str)
    """
    # Try boolean
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null" or value.lower() == "none":
        return None
    
    # Try int
    try:
        return int(value)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value)
    except ValueError:
        pass
    
    # Try list (comma-separated)
    if "," in value:
        return [parse_value(v.strip()) for v in value.split(",")]
    
    # Return as string
    return value
