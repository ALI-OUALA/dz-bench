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


def test_structured_field_scoring_is_deterministic_and_sorted() -> None:
    # Set up fields such that non-deterministic ordering would result in varying
    # floating-point summation outputs when computing coordinate_iou.
    # We verify deterministic sorted evaluation directly via the returned coordinate_iou.
    reference_fields = []
    hypothesis_fields = []
    for i in range(100):
        # We vary the bounding box coordinates based on the index to create different IoU scores
        ref_field = _field(f"field_name_{i}", "value")
        hyp_field = _field(f"field_name_{i}", "value")

        ref_field = ref_field.model_copy(
            update={"bbox": ref_field.bbox.model_copy(update={"y": 30.0 + i * 0.01})}
        )
        hyp_field = hyp_field.model_copy(
            update={"bbox": hyp_field.bbox.model_copy(update={"y": 30.0 + (i % 23) * 0.1})}
        )
        # Modify the height so the sum is non-trivial and prone to precision loss if order differs
        # (Though with sorting it will always be the same precision loss).
        ref_field = ref_field.model_copy(
            update={"bbox": ref_field.bbox.model_copy(update={"height": 30.0 + i * 0.01})}
        )
        hyp_field = hyp_field.model_copy(
            update={"bbox": hyp_field.bbox.model_copy(update={"height": 30.0 + (i % 17) * 0.01})}
        )

        reference_fields.append(ref_field)
        hypothesis_fields.append(hyp_field)

    reference = DocumentExtraction(
        document_id="doc1",
        schema_name="sch",
        schema_version="1.0.0",
        fields=reference_fields,
    )
    hypothesis = DocumentExtraction(
        document_id="doc1",
        schema_name="sch",
        schema_version="1.0.0",
        fields=hypothesis_fields,
    )

    # We just ensure it computes a stable score without throwing.
    # The actual sorting is checked by ensuring fields are processed in sorted order.
    structured_field_scores(reference, hypothesis)

    # Let's mock or patch bounding_box_iou to record the order of calls
    import unittest.mock

    from dz_bench.metrics import bounding_box_iou

    with unittest.mock.patch(
        "dz_bench.metrics.bounding_box_iou", wraps=bounding_box_iou
    ) as mock_iou:
        structured_field_scores(reference, hypothesis)

    called_names = []
    for call in mock_iou.call_args_list:
        # We need to trace back the bounding boxes to the field names.
        # It's easier if we just know we patched it and it was called in order.
        # Actually, since all fields have y = 30.0 + i * 0.01, we can infer `i` from y.
        # i = (y - 30.0) / 0.01
        ref_box = call.args[0]
        i = round((ref_box.y - 30.0) / 0.01)
        called_names.append(f"field_name_{i}")

    # The fields should have been processed in alphabetical order:
    # "field_name_0", "field_name_1", "field_name_10", "field_name_11", ...
    sorted_names = sorted(f"field_name_{i}" for i in range(100))
    assert called_names == sorted_names
