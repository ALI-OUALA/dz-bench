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


def test_structured_field_scoring_is_deterministic() -> None:
    # Adding items that would cause different orders across Python runs if sets are unordered.
    # We create a large number of fields to amplify floating-point precision differences
    # if evaluated in a different order.
    fields_ref = [_field(f"field_a_{i}", f"val_{i}") for i in range(50)] + [
        _field(f"total_ht_{i}", f"{i}.00") for i in range(50)
    ]

    fields_hyp = (
        [_field(f"field_a_{i}", f"val_{i}") for i in range(40)]
        + [_field(f"field_a_{i}", f"wrong_{i}") for i in range(40, 50)]
        + [_field(f"total_ht_{i}", f"{i}.00") for i in range(40)]
        + [_field(f"total_ht_{i}", f"wrong_{i}") for i in range(40, 50)]
        + [_field(f"extra_field_{i}", "extra") for i in range(10)]
    )

    reference1 = DocumentExtraction(
        document_id="invoice-1",
        schema_name="invoice-dz",
        schema_version="1.0.0",
        fields=fields_ref,
    )
    hypothesis1 = reference1.model_copy(update={"fields": fields_hyp})

    reference2 = reference1.model_copy(update={"fields": list(reversed(fields_ref))})
    hypothesis2 = hypothesis1.model_copy(update={"fields": list(reversed(fields_hyp))})

    scores1 = structured_field_scores(reference1, hypothesis1)
    scores2 = structured_field_scores(reference2, hypothesis2)

    # We do strict equality (==), NOT pytest.approx, because exact deterministic ordering
    # should produce identically evaluated floating point additions.
    assert scores1.precision == scores2.precision
    assert scores1.recall == scores2.recall
    assert scores1.exact_accuracy == scores2.exact_accuracy
    assert scores1.financial_accuracy == scores2.financial_accuracy
    assert scores1.hallucination_rate == scores2.hallucination_rate
    assert scores1.coordinate_iou == scores2.coordinate_iou
