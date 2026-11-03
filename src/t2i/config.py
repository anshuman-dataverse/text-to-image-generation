from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # repository root
CONFIGS = ROOT / "configs"
OUTPUTS = ROOT / "outputs"


def load_config(path: str | Path = CONFIGS / "milestone1.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def output_dir(milestone: str) -> Path:
    """outputs/<milestone>/, with results/, figures/ and logs/ inside."""
    out = OUTPUTS / milestone
    for sub in ("results", "figures", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out
