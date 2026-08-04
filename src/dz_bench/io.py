"""Small JSON and public-schema loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .models import GroundTruth, Manifest, Predictions, Report

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"
SCHEMA_FILES = {
    "common": "common.schema.json",
    "manifest": "manifest.schema.json",
    "ground-truth": "ground-truth.schema.json",
    "prediction": "prediction.schema.json",
    "report": "report.schema.json",
}


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object and reject arrays/scalars at the contract boundary."""

    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return payload


def schema_path(name: str) -> Path:
    filename = SCHEMA_FILES.get(name, name if name.endswith(".schema.json") else "")
    if not filename:
        raise ValueError(f"unknown public schema: {name}")
    candidates = (SCHEMA_DIR / filename, Path.cwd() / "schemas" / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"public schema not found: {filename}")


def load_schema(name: str) -> dict[str, Any]:
    schema = load_json(schema_path(name))
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ValueError(f"{name} is not a JSON Schema draft 2020-12 document")
    return schema


def load_model[ModelT: BaseModel](path: str | Path, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate(load_json(path))


def load_manifest(path: str | Path) -> Manifest:
    return load_model(path, Manifest)


def load_ground_truth(path: str | Path) -> GroundTruth:
    return load_model(path, GroundTruth)


def load_predictions(path: str | Path) -> Predictions:
    return load_model(path, Predictions)


def load_report(path: str | Path) -> Report:
    return load_model(path, Report)


def write_json(payload: BaseModel | dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return target
