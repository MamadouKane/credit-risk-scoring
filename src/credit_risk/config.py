"""Read configs/config.yaml and resolve its paths against the repository root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Return the project configuration as a nested dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(key: str, config: dict[str, Any] | None = None) -> Path:
    """Absolute path for an entry of the `paths` section, e.g. project_path("raw")."""
    config = config or load_config()
    return ROOT / config["paths"][key]
