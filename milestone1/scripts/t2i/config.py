from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # milestone1/


def load_config(path: str | Path = ROOT / "configs" / "default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the milestone folder."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
