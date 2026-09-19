from pathlib import Path

import pytest
from jsonschema import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError

from dz_bench.io import load_manifest, load_schema, validate_instance
from dz_bench.models import (
    DocumentExtraction,
    Predictions,
    PredictionSample,
    ProcessingError,
    Provenance,
    RunMetadata,
    StructuredField,
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
    for name in ("manifest", "ground-truth", "prediction", "report", "assets"):
        schema = load_schema(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] == f"{name}.schema.json"


def test_generated_artifacts_match_public_json_schemas() -> None:
    corpus = generate_corpus(seed=6, document_count=1)
    predictions = _predictions(
        corpus,
        [
            PredictionSample(
                document_id=corpus.ground_truth.documents[0].document_id,
                page_id=corpus.ground_truth.documents[0].pages[0].page_id,
                status="success",
                page=corpus.ground_truth.documents[0].pages[0],
            )
        ],
    )
    report = score(corpus.manifest, corpus.ground_truth, predictions)
    for kind, artifact in (
        ("manifest", corpus.manifest),
        ("ground-truth", corpus.ground_truth),
        ("prediction", predictions),
        ("report", report),
    ):
        validate_instance(artifact.model_dump(mode="json"), kind)

    invalid = corpus.manifest.model_dump(mode="json")
    invalid.pop("dataset_revision")
    with pytest.raises(JsonSchemaValidationError):
        validate_instance(invalid, "manifest")


def test_structured_extractions_round_trip_through_prediction_schema() -> None:
    corpus = generate_corpus(seed=8, document_count=1)
    document_id = corpus.ground_truth.documents[0].document_id
    extraction = DocumentExtraction(
        document_id=document_id,
        schema_name="invoice-dz",
        schema_version="1.0.0",
        fields=[
            StructuredField(
                field_id="invoice-number",
                field_name="invoice_number",
                value="FA-42",
                normalized_value="FA-42",
                value_type="identifier",
                confidence={"score": 0.9},
                provenance=Provenance(kind="system_prediction", source="fixture"),
            )
        ],
    )
    predictions = _predictions(corpus, [])
    predictions.document_extractions.append(extraction)

    validate_instance(predictions.model_dump(mode="json"), "prediction")


def test_report_scores_confidence_hallucination_and_structured_fields() -> None:
    corpus = generate_corpus(seed=9, document_count=1)
    truth_document = corpus.ground_truth.documents[0]
    extraction = DocumentExtraction(
        document_id=truth_document.document_id,
        schema_name="invoice-dz",
        schema_version="1.0.0",
        fields=[
            StructuredField(
                field_id="total-ttc",
                field_name="total_ttc",
                value="1190.00",
                normalized_value="1190.00",
                value_type="decimal",
                confidence={"score": 1.0, "calibrated": True},
                provenance=Provenance(kind="human_annotation", source="fixture"),
            )
        ],
    )
    truth_document.extractions.append(extraction)
    predictions = _predictions(
        corpus,
        [
            PredictionSample(
                document_id=truth_document.document_id,
                page_id=truth_document.pages[0].page_id,
                status="success",
                page=truth_document.pages[0],
            )
        ],
    )
    predictions.document_extractions.append(extraction)

    report = score(corpus.manifest, corpus.ground_truth, predictions)

    assert report.metrics["confidence_brier"].micro == 0.0
    assert report.metrics["hallucinated_block_rate"].micro == 0.0
    assert report.metrics["structured_field_f1"].micro == 1.0
    assert report.metrics["financial_value_accuracy"].micro == 1.0


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


def test_score_extractions_determinism() -> None:
    corpus = generate_corpus(seed=10, document_count=1)
    truth_document = corpus.ground_truth.documents[0]

    # Create multiple extractions to verify `key` sorting works without dependency on dict order
    ex1 = DocumentExtraction(
        document_id=truth_document.document_id,
        schema_name="invoice-dz",
        schema_version="1.0.0",
        fields=[
            StructuredField(
                field_id="f1",
                field_name="total_ttc",
                value="1190.00",
                normalized_value="1190.00",
                value_type="decimal",
                confidence={"score": 1.0},
                provenance=Provenance(kind="human_annotation", source="fixture"),
            )
        ],
    )
    ex2 = DocumentExtraction(
        document_id=truth_document.document_id,
        schema_name="invoice-fr",
        schema_version="1.0.0",
        fields=[
            StructuredField(
                field_id="f2",
                field_name="total_ht",
                value="1000.00",
                normalized_value="1000.00",
                value_type="decimal",
                confidence={"score": 1.0},
                provenance=Provenance(kind="human_annotation", source="fixture"),
            )
        ],
    )

    truth_document.extractions.append(ex1)
    truth_document.extractions.append(ex2)

    predictions1 = _predictions(corpus, [])
    # order 1
    predictions1.document_extractions.append(ex1)
    predictions1.document_extractions.append(ex2)

    predictions2 = _predictions(corpus, [])
    # order 2
    predictions2.document_extractions.append(ex2)
    predictions2.document_extractions.append(ex1)

    report1 = score(corpus.manifest, corpus.ground_truth, predictions1)
    report2 = score(corpus.manifest, corpus.ground_truth, predictions2)

    assert (
        report1.metrics["financial_value_accuracy"].micro
        == report2.metrics["financial_value_accuracy"].micro
    )
