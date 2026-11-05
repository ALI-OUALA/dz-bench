import pytest

from dz_bench.metrics import (
    confidence_brier_score,
    error_detection_auroc,
    expected_calibration_error,
    structured_field_scores,
)
from dz_bench.models import (
    BoundingBox,
    Confidence,
    DocumentExtraction,
    Provenance,
    StructuredField,
)


def _field(name: str, value: str, *, confidence: float = 1.0, x: float = 10) -> StructuredField:
    return StructuredField(
        field_id=f"field-{name.replace('.', '-')}",
        field_name=name,
        value=value,
        normalized_value=value,
        value_type="decimal" if "total" in name else "string",
        confidence=Confidence(score=confidence, calibrated=False),
        page_id="invoice-page-1",
        bbox=BoundingBox(x=x, y=20, width=100, height=30),
        provenance=Provenance(kind="human_annotation", source="test"),
    )


def test_confidence_metrics_have_hand_computed_behavior() -> None:
    assert confidence_brier_score([1, 0], [0.8, 0.2]) == pytest.approx(0.04)
    assert expected_calibration_error([1, 0], [0.8, 0.2], bins=2) == pytest.approx(0.2)
    assert error_detection_auroc([False, True], [0.1, 0.9]) == 1.0


def test_structured_field_scoring_ignores_missing_financial_fields() -> None:
    from dz_bench.scoring import _extraction_metrics

    reference_without_financial = DocumentExtraction(
        document_id="invoice-1",
        schema_name="invoice-dz",
        schema_version="1.0.0",
        fields=[_field("invoice_number", "FA-2026-0042"), _field("currency", "DZD")],
    )
    hypothesis = reference_without_financial.model_copy()

    # Zero financial fields means we don't report the metric rather than reporting 0.0
    metrics = _extraction_metrics(reference_without_financial, hypothesis)
    assert "financial_value_accuracy" not in metrics
    assert metrics["structured_field_exact_accuracy"].sample_count == 2

    reference_with_financial = DocumentExtraction(
        document_id="invoice-1",
        schema_name="invoice-dz",
        schema_version="1.0.0",
        fields=[
            _field("invoice_number", "FA-2026-0042"),
            _field("total_ht", "100.00"),
            _field("total_ttc", "119.00"),
        ],
    )
    metrics_with_financial = _extraction_metrics(
        reference_with_financial, reference_with_financial.model_copy()
    )
    assert "financial_value_accuracy" in metrics_with_financial
    assert metrics_with_financial["financial_value_accuracy"].sample_count == 2


def test_structured_field_scoring_penalizes_wrong_and_hallucinated_values() -> None:
    reference = DocumentExtraction(
        document_id="invoice-1",
        schema_name="invoice-dz",
        schema_version="1.0.0",
        fields=[_field("invoice_number", "FA-2026-0042"), _field("total_ttc", "1190.00")],
    )
    hypothesis = reference.model_copy(
        update={
            "fields": [
                _field("invoice_number", "FA-2026-0042", confidence=0.9),
                _field("total_ttc", "190.00", confidence=0.8),
                _field("currency", "DZD", confidence=0.7),
            ]
        }
    )

    scores = structured_field_scores(reference, hypothesis)

    assert scores.precision == pytest.approx(1 / 3)
    assert scores.recall == pytest.approx(1 / 2)
    assert scores.exact_accuracy == pytest.approx(1 / 2)
    assert scores.financial_accuracy == 0.0
    assert scores.hallucination_rate == pytest.approx(1 / 3)
    assert scores.coordinate_iou == 1.0


def test_structured_field_scoring_does_not_misclassify_financial_fields() -> None:
    reference = DocumentExtraction(
        document_id="invoice-1",
        schema_name="invoice-dz",
        schema_version="1.0.0",
        fields=[
            _field("shipping_weight", "10kg"),
            _field("flight_number", "AH1004"),
            _field("total_ht", "100.00"),
        ],
    )
    scores = structured_field_scores(reference, reference.model_copy())
    assert scores.financial_count == 1
    assert scores.financial_accuracy == 1.0


def test_structured_field_scoring_is_deterministic() -> None:
    import random

    from dz_bench.metrics import structured_field_scores

    reference_fields = [
        _field("invoice_number", "FA-2026-0042", x=10),
        _field("total_ht", "100.00", x=20),
        _field("total_ttc", "119.00", x=30),
        _field("currency", "DZD", x=40),
        _field("shipping_weight", "10kg", x=50),
        _field("flight_number", "AH1004", x=60),
    ]
    hypothesis_fields = [
        _field("invoice_number", "FA-2026-0042", confidence=0.9, x=15),
        _field("total_ht", "100.00", confidence=0.8, x=25),
        _field("total_ttc", "119.00", confidence=0.9, x=35),
        _field("currency", "DZD", confidence=0.7, x=45),
        _field("shipping_weight", "10kg", confidence=0.9, x=55),
        _field("flight_number", "AH1004", confidence=0.9, x=65),
    ]

    baseline_reference = DocumentExtraction(
        document_id="invoice-1",
        schema_name="invoice-dz",
        schema_version="1.0.0",
        fields=reference_fields,
    )
    baseline_hypothesis = DocumentExtraction(
        document_id="invoice-1",
        schema_name="invoice-dz",
        schema_version="1.0.0",
        fields=hypothesis_fields,
    )

    baseline_scores = structured_field_scores(baseline_reference, baseline_hypothesis)

    for _ in range(20):
        random.shuffle(reference_fields)
        random.shuffle(hypothesis_fields)

        shuffled_reference = DocumentExtraction(
            document_id="invoice-1",
            schema_name="invoice-dz",
            schema_version="1.0.0",
            fields=reference_fields,
        )
        shuffled_hypothesis = DocumentExtraction(
            document_id="invoice-1",
            schema_name="invoice-dz",
            schema_version="1.0.0",
            fields=hypothesis_fields,
        )

        scores = structured_field_scores(shuffled_reference, shuffled_hypothesis)
        assert scores == baseline_scores
