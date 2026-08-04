from pathlib import Path

import pytest
from pydantic import ValidationError

from dz_bench.io import load_manifest, load_schema
from dz_bench.models import (
    Predictions,
    PredictionSample,
    ProcessingError,
    RunMetadata,
    SystemMetadata,
)
from dz_bench.scoring import score
from dz_bench.synthetic import generate_corpus


def _predictions(corpus, samples) -> Predictions:
    return Predictions(
        dataset_revision=corpus.manifest.dataset_revision,
        coordinate_system=corpus.manifest.coordinate_system,
        system=SystemMetadata(
            name="fixture-system",
            version="0.1.0",
            adapter_name="fixture-json",
            adapter_version="1.0.0",
            execution_provider="cpu",
        ),
        run=RunMetadata(run_id="test-run"),
        samples=samples,
    )


def test_invalid_contract_is_rejected() -> None:
    corpus = generate_corpus(seed=4, document_count=1)
    payload = corpus.manifest.model_dump(mode="json")
    payload["documents"][0]["pages"][0]["checksum"]["value"] = "not-a-sha256"
    with pytest.raises(ValidationError):
        type(corpus.manifest).model_validate(payload)


def test_missing_and_crashed_pages_are_counted() -> None:
    corpus = generate_corpus(seed=4, document_count=3)
    pages = [
        (document.document_id, document.pages[0]) for document in corpus.ground_truth.documents
    ]
    samples = [
        PredictionSample(
            document_id=pages[0][0],
            page_id=pages[0][1].page_id,
            status="success",
            page=pages[0][1],
        ),
        PredictionSample(
            document_id=pages[1][0],
            page_id=pages[1][1].page_id,
            status="crashed",
            error=ProcessingError(code="worker_crashed", message="fixture crash"),
        ),
    ]
    report = score(corpus.manifest, corpus.ground_truth, _predictions(corpus, samples))
    assert report.summary.total_pages == 3
    assert report.summary.scored_pages == 1
    assert report.summary.crashed_pages == 1
    assert report.summary.missing_pages == 1
    assert {failure.status for failure in report.failures} == {"crashed", "missing"}
    assert report.metrics["cer"].sample_count == 1


def test_all_public_schema_files_load() -> None:
    for name in ("manifest", "ground-truth", "prediction", "report"):
        schema = load_schema(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] == f"{name}.schema.json"


def test_reference_only_manifest_has_no_documents() -> None:
    path = Path("datasets/bac/manifests/bac-reference-only-v0.1.json")
    manifest = load_manifest(path)
    assert manifest.documents == []
    assert manifest.reference_sources[0].redistribution == "reference-only"


def test_reading_order_does_not_compare_system_local_ids() -> None:
    corpus = generate_corpus(seed=4, document_count=1)
    page = corpus.ground_truth.documents[0].pages[0]
    hypothesis = page.model_copy(deep=True)
    for index, block in enumerate(hypothesis.blocks):
        block.block_id = f"prediction-block-{index}"
        for line_index, line in enumerate(block.lines):
            line.line_id = f"prediction-line-{index}-{line_index}"
    hypothesis.reading_order = [
        line.line_id
        for block in sorted(hypothesis.blocks, key=lambda value: value.reading_order_index)
        for line in block.lines
    ]
    prediction = PredictionSample(
        document_id=corpus.ground_truth.documents[0].document_id,
        page_id=page.page_id,
        status="success",
        page=hypothesis,
    )
    report = score(corpus.manifest, corpus.ground_truth, _predictions(corpus, [prediction]))
    assert report.metrics["reading_order_sequence_score"].micro == 1.0
