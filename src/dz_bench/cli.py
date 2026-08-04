"""Typer CLI for contract validation, synthetic generation, and scoring."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import typer

from .adapters.fake import FakeSystemAdapter
from .io import (
    load_ground_truth,
    load_manifest,
    load_predictions,
    load_report,
    load_schema,
    write_json,
)
from .scoring import score_files, write_report_markdown
from .synthetic import generate_corpus, write_bac_corpus, write_corpus

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("synthetic")
def synthetic_command(
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="Directory for generated JSON artifacts."
    ),
    seed: int = typer.Option(17, help="Deterministic generator seed."),
    documents: int = typer.Option(4, min=1, help="Number of original synthetic documents."),
    pages: int = typer.Option(1, min=1, help="Pages per synthetic document."),
    records: bool = typer.Option(False, "--records", help="Also emit plain JSON page records."),
) -> None:
    outputs = write_corpus(output_dir, seed, documents, pages, records)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("bac-synthetic")
def bac_synthetic_command(
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="Directory for the records-only BAC-like corpus."
    ),
    seed: int = typer.Option(17, help="Deterministic generator seed."),
    repeats: int = typer.Option(1, min=1, help="Repeat the balanced scenario set."),
) -> None:
    outputs = write_bac_corpus(output_dir, seed, repeats)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("bac-images")
def bac_images_command(
    font: Path = typer.Option(
        ...,
        "--font",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="External TTF/OTF font used for rendering; no font is bundled.",
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="Directory for the PNG evaluation bundle."
    ),
    seed: int = typer.Option(17, help="Deterministic generator seed."),
    repeats: int = typer.Option(1, min=1, help="Repeat the balanced scenario set."),
) -> None:
    from .raster import write_bac_images

    try:
        outputs = write_bac_images(output_dir, font, seed, repeats)
    except (FileNotFoundError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("smoke")
def smoke_command(
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for smoke artifacts."),
    seed: int = typer.Option(17, help="Deterministic generator seed."),
) -> None:
    """Generate synthetic data, fake predictions, and a scored report."""

    corpus = generate_corpus(seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_json(corpus.manifest, output_dir / "manifest.json")
    truth_path = write_json(corpus.ground_truth, output_dir / "ground-truth.json")
    prediction_path = write_json(
        FakeSystemAdapter().predict(corpus.ground_truth), output_dir / "predictions.json"
    )
    report = score_files(manifest_path, truth_path, prediction_path, output_dir / "report.json")
    markdown_path = write_report_markdown(report, output_dir / "report.md")
    for path in (
        manifest_path,
        truth_path,
        prediction_path,
        output_dir / "report.json",
        markdown_path,
    ):
        typer.echo(str(path))


@app.command("score")
def score_command(
    manifest: Path = typer.Option(..., help="Manifest JSON path."),
    ground_truth: Path = typer.Option(..., "--ground-truth", help="Ground-truth JSON path."),
    predictions: Path = typer.Option(..., help="Prediction JSON path."),
    output: Path = typer.Option(..., help="Structured report JSON path."),
) -> None:
    report = score_files(manifest, ground_truth, predictions, output)
    markdown = write_report_markdown(report, output.with_suffix(".md"))
    typer.echo(f"report: {output}")
    typer.echo(f"markdown: {markdown}")


@app.command("validate")
def validate_command(
    path: Path = typer.Argument(..., exists=True, readable=True),
    kind: str = typer.Option(..., "--kind", help="manifest, ground-truth, prediction, or report."),
) -> None:
    loaders: dict[str, Callable[[Path], object]] = {
        "manifest": load_manifest,
        "ground-truth": load_ground_truth,
        "prediction": load_predictions,
        "report": load_report,
    }
    loader = loaders.get(kind)
    if loader is None:
        raise typer.BadParameter(f"unknown contract kind: {kind}", param_hint="--kind")
    loader(path)
    typer.echo(f"valid {kind}: {path}")


@app.command("schema")
def schema_command(
    name: str = typer.Argument(..., help="manifest, ground-truth, prediction, report, or common."),
) -> None:
    typer.echo(json.dumps(load_schema(name), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
