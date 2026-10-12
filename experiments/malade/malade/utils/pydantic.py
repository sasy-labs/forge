from pydantic import BaseModel
import json
from pathlib import Path
from typing import Any

# Base directory for relative paths (malade package root)
_MALADE_DIR = Path(__file__).parent.parent.parent


def _resolve_path(path: str) -> Path:
    """Resolve a path relative to the malade directory if not absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return _MALADE_DIR / p


def load(model: type[BaseModel], path: str) -> Any:
    def load_fn() -> model:
        resolved = _resolve_path(path)
        with open(resolved, "r") as f:
            return model.model_validate_json(f.read())

    return load_fn


def save(model: type[BaseModel], path: str):
    def save_fn(instance: model) -> None:
        resolved = _resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved, "w") as f:
            return json.dump(instance.model_dump(), f)

    return save_fn
