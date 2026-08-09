"""Stateless public evaluation API; benchmark data remains caller-owned."""

from __future__ import annotations

import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile

from .io import load_ground_truth, load_manifest, load_predictions
from .scoring import score_files

MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
app = FastAPI(title="DZ-Bench API", version="1.0.0", redoc_url=None)


async def _save(upload: UploadFile, path: Path) -> None:
    data = await upload.read(MAX_ARTIFACT_BYTES + 1)
    if len(data) > MAX_ARTIFACT_BYTES:
        raise HTTPException(413, "benchmark artifact is too large")
    path.write_bytes(data)


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/score")
async def score(
    manifest: UploadFile = File(),
    ground_truth: UploadFile = File(),
    predictions: UploadFile = File(),
):
    with tempfile.TemporaryDirectory(prefix="dz-bench-") as directory:
        root = Path(directory)
        paths = [root / "manifest.json", root / "ground-truth.json", root / "predictions.json"]
        for upload, path in zip((manifest, ground_truth, predictions), paths, strict=True):
            await _save(upload, path)
        try:
            load_manifest(paths[0])
            load_ground_truth(paths[1])
            load_predictions(paths[2])
            report = score_files(paths[0], paths[1], paths[2], root / "report.json")
        except (ValueError, OSError) as exc:
            raise HTTPException(422, "invalid benchmark artifact") from exc
        return report.model_dump(mode="json")


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8001, proxy_headers=True)
