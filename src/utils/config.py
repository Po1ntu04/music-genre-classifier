from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    project_root = config_path.parent.parent
    resolved_paths = {}
    for key, value in config.get("paths", {}).items():
        resolved_paths[key] = (project_root / value).resolve()

    config["paths"] = resolved_paths
    config.setdefault("project", {})
    config["project"]["root_dir"] = project_root
    config["project"]["config_path"] = config_path
    return config
