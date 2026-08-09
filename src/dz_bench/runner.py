"""Sandbox-friendly black-box runner for any OCR engine public contract."""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .io import load_manifest, load_predictions, write_json
from .models import (
    CoordinateSystem,
    Predictions,
    PredictionSample,
    ProcessingError,
    RunMetadata,
    SystemMetadata,
)


@dataclass(frozen=True, slots=True)
class RunnerLimits:
    timeout_seconds: float = 600
    max_prediction_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.max_prediction_bytes <= 0:
            raise ValueError("runner limits must be positive")


@dataclass(frozen=True, slots=True)
class RunResult:
    predictions_path: Path
    return_code: int | None
    status: str
    duration_ms: float


def run_system(
    command: list[str],
    *,
    bundle_dir: Path,
    output_path: Path,
    system_name: str,
    system_version: str,
    limits: RunnerLimits = RunnerLimits(),
) -> RunResult:
    """Execute without a shell; placeholders are the only path interpolation."""

    if not command or any(not value for value in command):
        raise ValueError("command must contain non-empty arguments")
    bundle = bundle_dir.resolve(strict=True)
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("bundle must contain manifest.json")
    output = output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        value.replace("{bundle}", str(bundle)).replace("{predictions}", str(output))
        for value in command
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "DZ_BENCH_BUNDLE": str(bundle),
            "DZ_BENCH_PREDICTIONS": str(output),
        }
    )
    started = time.perf_counter()
    status = "crashed"
    return_code: int | None = None
    try:
        process = subprocess.run(
            argv,
            cwd=bundle,
            env=environment,
            shell=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=limits.timeout_seconds,
            check=False,
        )
        return_code = process.returncode
        status = "success" if process.returncode == 0 else "crashed"
    except subprocess.TimeoutExpired:
        status = "timeout"
    duration_ms = (time.perf_counter() - started) * 1000
    valid = False
    if (
        status == "success"
        and output.is_file()
        and output.stat().st_size <= limits.max_prediction_bytes
    ):
        try:
            load_predictions(output)
            valid = True
        except (ValueError, OSError):
            status = "invalid"
    elif status == "success":
        status = "invalid"
    if not valid:
        _write_failure_predictions(
            manifest_path,
            output,
            system_name=system_name,
            system_version=system_version,
            status="timeout" if status == "timeout" else "crashed",
            code=f"runner_{status}",
            duration_ms=duration_ms,
            command=argv,
        )
    return RunResult(output, return_code, status, duration_ms)


def _write_failure_predictions(
    manifest_path: Path,
    output: Path,
    *,
    system_name: str,
    system_version: str,
    status: Literal["crashed", "timeout"],
    code: str,
    duration_ms: float,
    command: list[str],
) -> None:
    manifest = load_manifest(manifest_path)
    samples = [
        PredictionSample(
            document_id=document.document_id,
            page_id=page.page_id,
            status=status,
            error=ProcessingError(
                code=code,
                message="black-box system did not produce a valid prediction artifact",
                retryable=status == "timeout",
            ),
        )
        for document in manifest.documents
        for page in document.pages
    ]
    rendered_command = " ".join(command)
    predictions = Predictions(
        dataset_revision=manifest.dataset_revision,
        coordinate_system=CoordinateSystem(),
        system=SystemMetadata(
            name=system_name,
            version=system_version,
            adapter_name="dz-bench-command",
            adapter_version="1.0.0",
            execution_provider="external",
            command=rendered_command[:2000],
        ),
        run=RunMetadata(
            run_id="run-" + hashlib.sha256(rendered_command.encode()).hexdigest()[:16],
            duration_ms=duration_ms,
            command=rendered_command[:2000],
        ),
        samples=samples,
    )
    write_json(predictions, output)
