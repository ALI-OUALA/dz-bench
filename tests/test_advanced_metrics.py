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


def _field(name: str, value: str, *, confidence: float = 1.0) -> StructuredField:
    return StructuredField(
        field_id=f"field-{name.replace('.', '-')}",
        field_name=name,
        value=value,
        normalized_value=value,
        value_type="decimal" if "total" in name else "string",
        confidence=Confidence(score=confidence, calibrated=False),
        page_id="invoice-page-1",
        bbox=BoundingBox(x=10, y=20, width=100, height=30),
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


def test_structured_field_scoring_is_deterministic_in_order() -> None:
    from unittest.mock import patch

    reference = DocumentExtraction(
        document_id="doc-1",
        schema_name="schema",
        schema_version="1.0.0",
        fields=[
            _field("z_field", "val"),
            _field("a_field", "val"),
            _field("m_field", "val"),
        ],
    )
    hypothesis = reference.model_copy()

    # Track order of field processing by extracting the x coordinate passed to bounding_box_iou.
    # To do this, we need fields to have different bounding boxes.
    for i, field in enumerate(reference.fields):
        if field.bbox:
            # Recreate bbox to bypass Pydantic frozen mutation
            reference.fields[i] = field.model_copy(
                update={
                    "bbox": BoundingBox(
                        x=float(i), y=field.bbox.y, width=field.bbox.width, height=field.bbox.height
                    )
                }
            )
            hypothesis.fields[i] = reference.fields[i].model_copy()

    # reference.fields[0] is z_field (x=0.0)
    # reference.fields[1] is a_field (x=1.0)
    # reference.fields[2] is m_field (x=2.0)

    observed_xs = []

    def mock_iou(first: BoundingBox, second: BoundingBox) -> float:
        observed_xs.append(first.x)
        return 1.0

    with patch("dz_bench.metrics.bounding_box_iou", mock_iou):
        structured_field_scores(reference, hypothesis)

    # The fields should be processed in alphabetical order: a_field, m_field, z_field
    # So we expect x coordinates: 1.0, 2.0, 0.0
    assert observed_xs == [1.0, 2.0, 0.0]
