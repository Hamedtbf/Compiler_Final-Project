import os
import copy

try:
    import yaml
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False

DEFAULT_CONFIG = {
    "weights": {
        "token": 0.3,
        "ast": 0.4,
        "cfg": 0.3
    },
    "ast": {
        "method": "levenshtein"  # "levenshtein" or "zhang_shasha"
    },
    "cfg": {
        "ged_max_nodes": 60,
        "ged_timeout": 5.0,
        "greedy_fallback": True
    },
    "token": {
        "ignore_literal_values": True
    }
}


def _deep_update(dest, src):
    """
    Recursively update dict dest with values from src in-place and return dest.
    """
    if src is None:
        return dest
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dest.get(k), dict):
            _deep_update(dest[k], v)
        else:
            dest[k] = copy.deepcopy(v)
    return dest


def load_config(path=None):
    """
    Load YAML config and deep-merge with defaults.
    Returns a fresh dict (deep copy of DEFAULT_CONFIG updated with user values).
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return cfg
    if not os.path.exists(path):
        print(f"[config] config file {path} not found — using defaults")
        return cfg
    if not YAML_AVAILABLE:
        raise RuntimeError("PyYAML not available. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as fh:
        user_cfg = yaml.safe_load(fh) or {}
    _deep_update(cfg, user_cfg)
    return cfg
