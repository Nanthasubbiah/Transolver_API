"""
Load and cache the model registry from models.json.
"""
import json
from pathlib import Path
from ..config import MODELS_JSON

_registry = None


def get_registry() -> list[dict]:
    global _registry
    if _registry is None:
        p = Path(MODELS_JSON)
        if not p.exists():
            raise FileNotFoundError(f"Model registry not found: {p}")
        _registry = json.loads(p.read_text())
    return _registry


def get_model_entry(model_id: str) -> dict:
    for m in get_registry():
        if m["id"] == model_id:
            return m
    raise KeyError(f"Unknown model_id: '{model_id}'")
