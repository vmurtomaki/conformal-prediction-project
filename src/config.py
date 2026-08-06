from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "hyperparameters.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    """Loads centralized hyperparameter defaults (alpha, gamma, bootstrap_estimators)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)