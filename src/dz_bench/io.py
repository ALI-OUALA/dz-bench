"""Small JSON and public-schema loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from pydantic import BaseModel, JsonValue
from referencing import Registry
from referencing.jsonschema import DRAFT202012

from .models import AssetIndex, GroundTruth, Manifest, Predictions, Report

PACKAGE_SCHEMA_DIR = Path(__file__).parent / "schemas"
SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"
SCHEMA_FILES = {
    "common": "common.schema.json",
    "manifest": "manifest.schema.json",
    "ground-truth": "ground-truth.schema.json",
    "prediction": "prediction.schema.json",
    "report": "report.schema.json",
    "assets": "assets.schema.json",
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
    candidates = (
        PACKAGE_SCHEMA_DIR / filename,
        SCHEMA_DIR / filename,
        Path.cwd() / "schemas" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"public schema not found: {filename}")


def load_schema(name: str) -> dict[str, Any]:
    schema = load_json(schema_path(name))
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ValueError(f"{name} is not a JSON Schema draft 2020-12 document")
    return schema


def validate_instance(payload: JsonValue, name: str) -> None:
    """Validate an artifact against the packaged public Draft 2020-12 contract."""

    schemas = [load_schema(value) for value in SCHEMA_FILES]
    registry = Registry().with_resources(
        (schema["$id"], DRAFT202012.create_resource(schema)) for schema in schemas
    )
    Draft202012Validator(load_schema(name), registry=registry).validate(payload)


def load_model[ModelT: BaseModel](path: str | Path, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate(load_json(path))


def load_manifest(path: str | Path) -> Manifest:
    payload = load_json(path)
    validate_instance(payload, "manifest")
    return Manifest.model_validate(payload)


def load_ground_truth(path: str | Path) -> GroundTruth:
    payload = load_json(path)
    validate_instance(payload, "ground-truth")
    return GroundTruth.model_validate(payload)


def load_predictions(path: str | Path) -> Predictions:
    payload = load_json(path)
    validate_instance(payload, "prediction")
    return Predictions.model_validate(payload)


def load_report(path: str | Path) -> Report:
    payload = load_json(path)
    validate_instance(payload, "report")
    return Report.model_validate(payload)


def load_asset_index(path: str | Path) -> AssetIndex:
    payload = load_json(path)
    validate_instance(payload, "assets")
    return AssetIndex.model_validate(payload)


def write_json(payload: BaseModel | object, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return target
