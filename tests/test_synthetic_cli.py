from pathlib import Path

from typer.testing import CliRunner

from dz_bench.cli import app
from dz_bench.io import load_ground_truth, load_manifest, write_json
from dz_bench.models import Predictions, PredictionSample, RunMetadata, SystemMetadata
from dz_bench.synthetic import generate_corpus


def test_synthetic_generation_is_deterministic() -> None:
    first = generate_corpus(seed=22, document_count=3, pages_per_document=2)
    second = generate_corpus(seed=22, document_count=3, pages_per_document=2)
    other = generate_corpus(seed=23, document_count=3, pages_per_document=2)
    assert first.manifest.model_dump(mode="json") == second.manifest.model_dump(mode="json")
    assert first.ground_truth.model_dump(mode="json") == second.ground_truth.model_dump(mode="json")
    assert first.records == second.records
    assert first.records != other.records


def test_cli_synthetic_validate_and_score(tmp_path: Path) -> None:
    runner = CliRunner()
    corpus_dir = tmp_path / "synthetic"
    generated = runner.invoke(
        app,
        [
            "synthetic",
            "--output-dir",
            str(corpus_dir),
            "--seed",
            "8",
            "--documents",
            "2",
            "--records",
        ],
    )
    assert generated.exit_code == 0, generated.output
    manifest = load_manifest(corpus_dir / "manifest.json")
    truth = load_ground_truth(corpus_dir / "ground-truth.json")
    predictions = Predictions(
        dataset_revision=manifest.dataset_revision,
        coordinate_system=manifest.coordinate_system,
        system=SystemMetadata(
            name="cli-fixture",
            version="0.1.0",
            adapter_name="fixture-json",
            adapter_version="1.0.0",
            execution_provider="cpu",
        ),
        run=RunMetadata(run_id="cli-run"),
        samples=[
            PredictionSample(
                document_id=document.document_id,
                page_id=page.page_id,
                status="success",
                page=page,
            )
            for document in truth.documents
            for page in document.pages
        ],
    )
    prediction_path = tmp_path / "predictions.json"
    write_json(predictions, prediction_path)
    validation = runner.invoke(
        app,
        ["validate", str(corpus_dir / "manifest.json"), "--kind", "manifest"],
    )
    assert validation.exit_code == 0, validation.output
    output = tmp_path / "report.json"
    scored = runner.invoke(
        app,
        [
            "score",
            "--manifest",
            str(corpus_dir / "manifest.json"),
            "--ground-truth",
            str(corpus_dir / "ground-truth.json"),
            "--predictions",
            str(prediction_path),
            "--output",
            str(output),
        ],
    )
    assert scored.exit_code == 0, scored.output
    assert output.is_file()
    assert output.with_suffix(".md").is_file()
