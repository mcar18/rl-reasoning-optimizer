"""Configuration loading and defaults."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_project_root() -> Path:
    """Return project root (parent of src)."""
    return Path(__file__).resolve().parents[3]


def get_data_path() -> Path:
    """Return path to data directory."""
    return get_project_root() / "data"


def get_runs_path() -> Path:
    """Return path to runs directory."""
    return get_project_root() / "runs"


def get_config_path() -> Path:
    """Return path to configs directory."""
    return get_project_root() / "configs"


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Shallow merge override into base."""
    out = dict(base)
    for k, v in override.items():
        out[k] = v
    return out


def load_config(
    config_name: str = "default",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load config by name (e.g. 'default', 'local_ollama', 'api_model')."""
    root = get_project_root()
    configs_dir = root / "configs"
    path = configs_dir / f"{config_name}.yaml"
    cfg = load_yaml(path)
    if overrides:
        cfg = merge_config(cfg, overrides)
    # Ensure paths are absolute where needed
    if "data_path" in cfg and not Path(cfg["data_path"]).is_absolute():
        cfg["data_path"] = str(root / cfg["data_path"])
    if "runs_path" in cfg and not Path(cfg["runs_path"]).is_absolute():
        cfg["runs_path"] = str(root / cfg["runs_path"])
    return cfg


def get_env(key: str, default: str = "") -> str:
    """Get environment variable safely."""
    return os.environ.get(key, default)
